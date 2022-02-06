# PyTorch tools
import argparse
import shutil
import sys
import time
from glob import glob

# Python tools
import numpy as np
import seaborn as sns
import torch
import torchvision
from absl import app
from absl.flags import argparse_flags
from matplotlib import pyplot as plt
# - Metrics
from pytorch_msssim import ms_ssim
from skimage import io
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# Utilities
# - Data
from data.clic_train import CLICTrain
from data.custom import CustomData
from data.kodak import Kodak
from data.mscoco import MSCOCO
# - Network
from darts_network import GoogleHyperPriorCoderDarts
from oct_network import OctGoogleHPCoder
# - Tools
from util.auto_helper import get_gpu_id
from util.log_manage import *
from util.msssim import MultiScaleSSIM
from util.psnr import PSNR, PSNR_np
from util.write_file import BitStreamIO

gpu_id, machine_id = get_gpu_id()
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
gpuCount = 1

np.set_printoptions(threshold=sys.maxsize)

torchvision.set_image_backend('accimage')
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Enable CUDA computation
use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if use_cuda else 'cpu')

Dataset_dir = "/work/gcwscs04/dataset/"
TrainData = "MSCOCO"

def get_max_idx(alist):
    max_v = 0
    max_i = -1
    for i in range(len(alist)):
        if alist[i] > max_v:
            # print(alist[i], ">", max_v)
            max_v = alist[i]
            max_i = i
    
    return max_i

def _concat(xs):
    for x in xs:
        con = x.view(-1)
        break
    for x in xs:
        if x is not None and x.size() != con.size():
            # print(x.size())
            con = torch.cat([con, x.view(-1)])
        #else:
            #print('nonetype')
            #print(x.size())
    return con#torch.cat([x.view(-1) for x in xs])

def get_coder(args):
    # Instantiate model.
    if args.Octave:
        coder = OctGoogleHPCoder(args.num_filters // 2, args.num_features // 2, args.num_hyperpriors // 2)
    else:
        coder = GoogleHyperPriorCoderDarts(args)

    return coder


def train(args):
    model_ID = os.path.basename(args.checkpoint_dir if args.checkpoint_dir[-1] is not '/' else args.checkpoint_dir[:-1])

    # Create input data pipeline.
    data_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_dataset = {"MSCOCO": MSCOCO(Dataset_dir + "COCO/", data_transforms),
                     "CLIC": CLICTrain(Dataset_dir + "CLIC_train/images/", data_transforms)}[TrainData]
    train_dataloader = data.DataLoader(train_dataset, args.batch_size, True, num_workers=16, drop_last=True)
    validate_dataset = Kodak(Dataset_dir + "Kodak/", transforms.ToTensor())
    validate_dataloader = data.DataLoader(validate_dataset, 1, False, num_workers=16)

    log_writer = SummaryWriter(log_dir=args.checkpoint_dir)

    # alphas_conv = torch.autograd.Variable(1e-3*torch.randn(6).cuda(), requires_grad=True).to(DEVICE)
    # alphas_acti = torch.autograd.Variable(1e-3*torch.randn(3).cuda(), requires_grad=True).to(DEVICE)

    coder = get_coder(args)
    coder = nn.DataParallel(coder)

    def evaluate(epoch, eval_lmda=None, is_resume=False):
        coder.eval()

        eval_psnr_list = []
        eval_msssim_list = []
        eval_rate_list = []
        eval_aux_loss_list = []

        eval_counter = 0

        with torch.no_grad():
            for eval_imgs in validate_dataloader:
                eval_counter += 1
                eval_imgs = eval_imgs.to(DEVICE)

                eval_imgs_tilde, eval_rate = coder(eval_imgs)

                if args.pretrain_ae:
                    eval_aux_loss = torch.Tensor([0])
                elif args.Octave:
                    eval_aux_loss = coder.module.entropy_bn_h.aux_loss() + coder.entropy_bn_l.aux_loss()
                else:
                    eval_aux_loss = coder.module.entropy_bottleneck.aux_loss()

                eval_imgs *= 255
                eval_imgs = eval_imgs.clamp(0, 255).round().float() / 255.
                eval_imgs_tilde *= 255
                eval_imgs_tilde = eval_imgs_tilde.clamp(0, 255).round().float() / 255.

                eval_psnr = PSNR(eval_imgs, eval_imgs_tilde, data_range=1.)
                eval_msssim = ms_ssim(eval_imgs, eval_imgs_tilde,
                                      data_range=1., size_average=True, nonnegative_ssim=True).item()

                eval_psnr_list.append(eval_psnr)
                eval_msssim_list.append(eval_msssim)
                eval_rate_list.append(eval_rate.item())
                eval_aux_loss_list.append(eval_aux_loss.item())

                if not is_resume and eval_counter == 7:
                    log_writer.add_image('kodim07.png', eval_imgs_tilde[0], epoch)

                # print("PSNR: {:.4f}, MS-SSIM: {:.4f}, rate: {:.4f}".format(eval_psnr, eval_msssim, eval_rate.item()))

            print("lmda={:.1e}::".format(eval_lmda) if eval_lmda is not None else "",
                  "PSNR: {:.4f}, MS-SSIM: {:.4f}, rate: {:.4f}, aux: {:.4f}".format(np.mean(eval_psnr_list),
                                                                                    np.mean(eval_msssim_list),
                                                                                    np.mean(eval_rate_list),
                                                                                    np.mean(eval_aux_loss_list)))

            if not is_resume:
                log_writer.add_scalar('Loss/aux', np.mean(eval_aux_loss_list), epoch)
                log_writer.add_scalar('Evaluation/PSNR', np.mean(eval_psnr_list), epoch)
                log_writer.add_scalar('Evaluation/MS-SSIM', np.mean(eval_msssim_list), epoch)
                log_writer.add_scalar('Evaluation/est-rate', np.mean(eval_rate_list), epoch)




    # load checkpoint if needed/ wanted
    start_epoch = 0
    if args.resume or args.train_extracted:

        # custom method for loading last checkpoint
        # if args.train_extracted:
        #     model_name = "darts_model.ckpt"
        # else:
        #     model_name = "model.ckpt"
        ckpt = torch.load(os.path.join(args.checkpoint_dir, "model.ckpt"), map_location='cpu')
        print("========================================================================")

        if args.reuse_args:
            args = load_args(args, args.checkpoint_dir)

        if args.resume:
            start_epoch = ckpt['epoch'] + 1

        # ==== only for old version model loading ====
        # coder_ckpt = ckpt['coder']
        # new_ckpt = {}
        # for key, value in coder_ckpt.items():
        #     # contents = key.split('.')
        #     # if contents[-2] in ['conv', 'deconv', 'gdn', 'igdn']:
        #     #     key = '.'.join(contents[:-2]) + '.m.' + '.'.join(contents[-2:])
        #     # key = 'module.' + key

        #     new_ckpt[key] = value

        # ckpt['coder'] = new_ckpt
        # ============================================

        try:
            coder.load_state_dict(ckpt['coder'])
        except RuntimeError as e:
            # Warning(e)
            print(e)
            coder.load_state_dict(ckpt['coder'], strict=False)

        coder = coder.to(DEVICE)

        # alphas_conv = coder.module.alpha_conv
        # alphas_acti = coder.module.alpha_acti

        if args.resume:
            if args.pretrain_ae:
                optim2 = torch.optim.Adam([torch.Tensor(0)])
            elif args.Octave:
                optim2 = torch.optim.Adam([coder.module.entropy_bn_h.quantiles.requires_grad_(),
                                        coder.module.entropy_bn_l.quantiles.requires_grad_()], lr=3e-4)
            else:
                optim2 = torch.optim.Adam([coder.module.entropy_bottleneck.quantiles.requires_grad_()], lr=3e-4)


            alpha_list = []
            for mod in coder.module.analysis.model:
                alpha_list.append(mod.alpha_conv.requires_grad_())
                alpha_list.append(mod.alpha_acti.requires_grad_())
                # for alpha_k in mod.alphas:
                #     alpha_list.append(mod.alphas[alpha_k].requires_grad_())

            for mod in coder.module.synthesis.model:
                alpha_list.append(mod.alpha_deconv.requires_grad_())
                alpha_list.append(mod.alpha_acti.requires_grad_())
                # for alpha_k in mod.alphas:
                #     alpha_list.append(mod.alphas[alpha_k].requires_grad_())

            optim3 = torch.optim.Adam(alpha_list, lr=3e-4)
        
        
            
            # optim3 = torch.optim.Adam([coder.module.alphas['alpha_conv'].requires_grad_(),
            #                         coder.module.alphas['alpha_deconv'].requires_grad_(),
            #                         coder.module.alphas['alpha_acti'].requires_grad_(),
            #                             ], lr=3e-4)


            optim.load_state_dict(ckpt['optim'])
            optim2.load_state_dict(ckpt['optim2'])
            if 'optim3' in ckpt.keys() and (not args.train_extracted):
                optim3.load_state_dict(ckpt['optim3'])

            # if not args.verbose:
            #     evaluate(0, is_resume=True)

        ############train extracted init steps##############################
        def find_chosen():
            softmax = torch.nn.Softmax(dim=0)
            for mod in coder.module.analysis.model:
                conv_alpha_s = softmax(mod.alpha_conv)
                chosen_i = get_max_idx(conv_alpha_s)
                mod.chosen_conv = chosen_i
                print(chosen_i)
                acti_alpha_s = softmax(mod.alpha_acti)
                chosen_i = get_max_idx(acti_alpha_s)
                mod.chosen_acti = chosen_i
                print(chosen_i)
                # for alpha_k in mod.alphas:
                #     alpha_k_s = softmax(mod.alphas[alpha_k])
                #     chosen_i = get_max_idx(softmax(alpha_k_s))
                #     #print(chosen_i)
                #     mod.chosen = chosen_i
                #     print(mod.chosen)
            for mod in coder.module.synthesis.model:
                deconv_alpha_s = softmax(mod.alpha_deconv)
                chosen_i = get_max_idx(deconv_alpha_s)
                mod.chosen_deconv = chosen_i
                print(chosen_i)
                acti_alpha_s = softmax(mod.alpha_acti)
                chosen_i = get_max_idx(acti_alpha_s)
                mod.chosen_acti = chosen_i
                print(chosen_i)
                # for alpha_k in mod.alphas:
                #     alpha_k_s = softmax(mod.alphas[alpha_k])
                #     chosen_i = get_max_idx(softmax(alpha_k_s))
                #     #print(chosen_i)
                #     mod.chosen = chosen_i
                #     print(mod.chosen)
        
        if args.train_extracted:
            for mod in coder.module.analysis.model:
                mod.search = False
            for mod in coder.module.synthesis.model:
                mod.search = False

            find_chosen()

        if args.pretrain_ae:
            optim2 = torch.optim.Adam([torch.Tensor(0)])
        elif args.Octave:
            optim2 = torch.optim.Adam([coder.module.entropy_bn_h.quantiles.requires_grad_(),
                                       coder.module.entropy_bn_l.quantiles.requires_grad_()], lr=3e-4)
        else:
            optim2 = torch.optim.Adam([coder.module.entropy_bottleneck.quantiles.requires_grad_()], lr=3e-4)

        #########################################################################

        print("Latest checkpoint restored, start training at step {}.".format(start_epoch))

    else:
        if args.pretrain_ae:
            optim2 = torch.optim.Adam([torch.Tensor(0)])
        elif args.Octave:
            optim2 = torch.optim.Adam([coder.module.entropy_bn_h.quantiles.requires_grad_(),
                                       coder.module.entropy_bn_l.quantiles.requires_grad_()], lr=3e-4)
        else:
            optim2 = torch.optim.Adam([coder.module.entropy_bottleneck.quantiles.requires_grad_()], lr=3e-4)
        
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        coder = coder.to(DEVICE) 

    optim = torch.optim.Adam(coder.parameters(), lr=3e-5)

    if not args.train_extracted:
        alpha_list = []
        for mod in coder.module.analysis.model:
            alpha_list.append(mod.alpha_conv.requires_grad_())
            alpha_list.append(mod.alpha_acti.requires_grad_())
            # for alpha_k in mod.alphas:
            #     alpha_list.append(mod.alphas[alpha_k].requires_grad_())

        for mod in coder.module.synthesis.model:
            alpha_list.append(mod.alpha_deconv.requires_grad_())
            alpha_list.append(mod.alpha_acti.requires_grad_())
            # for alpha_k in mod.alphas:
            #     alpha_list.append(mod.alphas[alpha_k].requires_grad_())

        optim3 = torch.optim.Adam(alpha_list, lr=3e-4)
    # optim3 = torch.optim.Adam([
    #         coder.module.alphas['alpha_conv'].requires_grad_(),
    #         coder.module.alphas['alpha_deconv'].requires_grad_(),
    #         coder.module.alphas['alpha_acti'].requires_grad_()
    #                             ], lr=3e-4)

    # def weight_require_grad(require=True):
    #     for mod in coder.module.analysis.model:
    #         mod.requires_grad = require
    #     for mod in coder.module.synthesis.model:
    #         mod.requires_grad = require
    #     coder.module.hyper_analysis.requires_grad = require
    #     coder.module.hyper_analysis.requires_grad = require
    #     coder.module.conditional_bottleneck.requires_grad = require
    #     coder.module.entropy_bottleneck.requires_grad = require
        

    # def alpha_require_grad(require = True):
    #     for mod in coder.module.analysis.model:
    #         for alpha_k in mod.alphas:
    #             mod.alphas[alpha_k].requires_grad = require

    #     for mod in coder.module.synthesis.model:
    #         for alpha_k in mod.alphas:
    #             mod.alphas[alpha_k].requires_grad = require

    def coder_forward(imgs, model=coder):
        imgs_tilde, rate = model(imgs)

        distortion = torch.mean(torch.pow(imgs - imgs_tilde, 2), dim=[1, 2, 3])
        distortion *= (255 ** 2)
        distortion = torch.clamp_max(distortion, 1e8)

        rate = torch.squeeze(rate)
        distortion = torch.squeeze(distortion)

        rd_loss = torch.mean(args.lmda * distortion + rate)
        return rd_loss, distortion, rate

    for epoch in range(start_epoch, args.max_epochs):
        coder.train()

        writer_record = {
            'rd_loss': [],
            'rate': [],
            'distortion': [],
            'alpha_loss': [],
            #'morph_loss': []
        }

        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        start_time = time.time()

        # for loop going through dataset
        for idx, imgs in enumerate(pbar):

            if not args.train_extracted:
                if idx%2 == 0:
                    eval_imgs = imgs.to(DEVICE)
                    continue
            imgs = imgs.to(DEVICE)

            # # save origin w
            # torch.save({
            # 'coder': coder.state_dict(),
            # 'optim': optim.state_dict(),
            # 'optim2': optim2.state_dict(),
            # 'optim3': optim3.state_dict()
            # }, os.path.join(args.checkpoint_dir, "tempw.ckpt"))

            # print("training weight")
            optim.zero_grad()        

            coder.train()

            # weight_require_grad(True)
            # alpha_require_grad(True)
            
            # coder.module.alphas['alpha_conv'].requires_grad = False
            # coder.module.alphas['alpha_deconv'].requires_grad = False
            # coder.module.alphas['alpha_acti'].requires_grad = False

            # forward pass
            eta = 0.0001
            rd_loss, distortion, rate = coder_forward(imgs)

            rd_loss.backward(retain_graph=True)
            optim.step()
            optim.zero_grad()

            ### get flops ###
            flops = coder.flops(imgs)
            print (flops)

            #update alpha by bi-level optimization
            if not args.train_extracted and args.bilevel == True:
                optim3.zero_grad()
                # print(coder)
                theta = _concat(coder.parameters()).data
                # print(theta.size())
                dtheta_prototype = torch.autograd.grad(rd_loss, coder.parameters(), allow_unused=True)
                dtheta = torch.tensor([])
                dtheta.data = dtheta_prototype[0].view(-1).clone()
                idxx = 0
                for x, y in zip(dtheta_prototype, coder.parameters()):
                    if idxx == 0:
                        idxx += 1
                        continue
                    if x is None:
                        dtheta = torch.cat([dtheta, torch.zeros_like(y.view(-1))])
                    else:
                        dtheta = torch.cat([dtheta, x.view(-1)])
                        

                theta = theta.sub(eta, dtheta)
                unrolled_coder = get_coder(args)
                for x, y in zip(unrolled_coder.analysis.model, coder.module.analysis.model):
                    x.alpha_conv.data.copy_(y.alpha_conv.data)
                    x.alpha_acti.data.copy_(y.alpha_acti.data)
                    # for ax, ay in zip(x.alphas, y.alphas):
                    #     x.alphas[ax].data.copy_(y.alphas[ay].data)
                    x.search = y.search
                    x.chosen_conv = y.chosen_conv
                    x.chosen_acti = y.chosen_acti
                for x, y in zip(unrolled_coder.synthesis.model, coder.module.synthesis.model):
                    x.alpha_deconv.data.copy_(y.alpha_deconv.data)
                    x.alpha_acti.data.copy_(y.alpha_acti.data)
                    # for ax, ay in zip(x.alphas, y.alphas):
                    #     x.alphas[ax].data.copy_(y.alphas[ay].data)
                    x.search = y.search
                    x.chosen_deconv = y.chosen_deconv
                    x.chosen_acti = y.chosen_acti

                model_dict = coder.state_dict()

                params, offset = {}, 0
                for k, v in coder.named_parameters():
                    v_length = np.prod(v.size())
                    params[k] = theta[offset: offset+v_length].view(v.size())
                    offset += v_length

                assert offset == len(theta)
                model_dict.update(params)
                unrolled_coder = unrolled_coder.to(DEVICE)
                unrolled_coder = nn.DataParallel(unrolled_coder)
                unrolled_coder.load_state_dict(model_dict)
                
                unroll_loss, _, _ = coder_forward(eval_imgs, unrolled_coder)
                
                unroll_loss.backward()

                dalpha = []
                for m in unrolled_coder.module.analysis.model:
                    dalpha.append(m.alpha_conv.grad)
                    dalpha.append(m.alpha_acti.grad)
                    # for key in m.alphas:
                    #     dalpha.append(m.alphas[key].grad)
                for m in unrolled_coder.module.synthesis.model:
                    dalpha.append(m.alpha_deconv.grad)
                    dalpha.append(m.alpha_acti.grad)    
                    # for key in m.alphas:
                    #     dalpha.append(m.alphas[key].grad)

                #dalpha = [v.grad for v in unrolled_coder.module.alphas()]
                vector = []
                for v in unrolled_coder.parameters():
                    if v.grad is not None:
                        vector.append(v.grad.data)
                    else:
                        vector.append(torch.zeros_like(v))
                # vector = [v.grad.data for v in unrolled_coder.parameters()]


                R = 0.01 / _concat(vector).norm()
                for p, v in zip(unrolled_coder.parameters(), vector):
                    p.data.add_(R, v)
                loss, _, _ = coder_forward(imgs, unrolled_coder)
                grads_p = torch.autograd.grad(loss, unrolled_coder.module.arch_parameters())

                for p, v in zip(unrolled_coder.parameters(), vector):
                    p.data.sub_(2*R, v)
                loss, _, _ = coder_forward(imgs, unrolled_coder)
                grads_n = torch.autograd.grad(loss, unrolled_coder.module.arch_parameters())

                for p, v in zip(unrolled_coder.parameters(), vector):
                    p.data.add_(R, v)

                implicit_grads = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

                

                for g, ig in zip(dalpha, implicit_grads):
                    g.data.sub_(eta, ig.data)

                for v, g in zip(coder.module.arch_parameters(), dalpha):
                    if v.grad is None:
                        v.grad = torch.autograd.Variable(g.data)
                    else:
                        v.grad.data.copy_(g.data)


                # print('encoder grad')
                # for m in unrolled_coder.module.analysis.model:
                #     for key in m.alphas:
                #         print(m.alphas[key].grad)
                # print('decoder grad')
                # for m in unrolled_coder.module.synthesis.model:
                #     for key in m.alphas:
                #         print(m.alphas[key].grad)

                
                optim.step()
                optim3.step()
                # morph_loss = coder.module.morph_loss()
                # rd_loss += 2 * morph_loss;

                # w -> w'
                # rd_loss.backward(retain_graph=True)
                # optim.step()

                # print("rd_loss: ", rd_loss)

            # not to use bi-level optimization for alpha update
            elif not args.train_extracted and not args.bilevel:
                optim3.zero_grad()

                eval_loss, _, _ = coder_forward(eval_imgs)

                eval_loss.backward()
                optim3.step()
                optim3.zero_grad()

            if not args.pretrain_ae:

                optim2.zero_grad()

                if args.Octave:
                    aux_loss = coder.module.entropy_bn_h.aux_loss() + coder.entropy_bn_l.aux_loss()
                else:
                    aux_loss = coder.module.entropy_bottleneck.aux_loss()

                aux_loss.backward(retain_graph=True)
                optim2.step()

            prepare_time = start_time - time.time()
            
            softmax = torch.nn.Softmax(dim=0)

            if not args.train_extracted:
                print("encoder Alphas:")
                for mod in coder.module.analysis.model:
                    print("alpha conv: ", softmax(mod.alpha_conv))
                    print("alpha acti: ", softmax(mod.alpha_acti))

                print("decoder Alphas:")
                for mod in coder.module.synthesis.model:
                    print("alpha deconv: ", softmax(mod.alpha_deconv))
                    print("alpha acti: ", softmax(mod.alpha_acti))

            # print("Alphas:")
            # print(softmax(coder.module.alphas['alpha_conv']))
            # print(softmax(coder.module.alphas['alpha_deconv']))
            # print(softmax(coder.module.alphas['alpha_acti']))

            # compute computation time and *compute_efficiency*
            softmax = nn.Softmax(dim=0)


            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}, dis: {:.3f}, rate: {:.3f}, epoch: {}/{}:".format(
                process_time / (process_time + prepare_time), distortion.mean(), rate.mean(), epoch, args.max_epochs))

            writer_record['rd_loss'].append(rd_loss.cpu().item())
            writer_record['rate'].append(rate.mean().cpu().item())
            writer_record['distortion'].append(distortion.mean().cpu().item())
            #writer_record['alpha_loss'].append(alpha_loss.cpu().item())
            #writer_record['morph_loss'].append(morph_loss.cpu().item())

            start_time = time.time()

        log_writer.add_scalar('Loss/total', np.mean(writer_record['rd_loss']), epoch)
        log_writer.add_scalar('Loss/distortion', np.mean(writer_record['distortion']), epoch)
        log_writer.add_scalar('Loss/rate', np.mean(writer_record['rate']), epoch)
        #log_writer.add_scalar('Loss/alpha', np.mean(writer_record['alpha_loss']), epoch)
        #log_writer.add_scalar('Loss/morph', np.mean(writer_record['morph_loss']), epoch)

        softmax = torch.nn.Softmax(dim=0)

        #if(args.resume):
        #    find_chosen()
        # Save model
        if args.train_extracted is True:
            torch.save({
                'epoch': epoch,
                'coder': coder.state_dict(),
                'optim': optim.state_dict(),
                'optim2': optim2.state_dict(),
            }, os.path.join(args.checkpoint_dir, "model.ckpt"))
        else:
            torch.save({
                'epoch': epoch,
                'coder': coder.state_dict(),
                'optim': optim.state_dict(),
                'optim2': optim2.state_dict(),
                'optim3': optim3.state_dict()
            }, os.path.join(args.checkpoint_dir, "model.ckpt"))

        # Testing
        if not args.verbose:
            print("Model ID:: {}".format(model_ID))
            evaluate(epoch)
        #coder.module.morph_status()
        

    #coder.module.demorphize()
    #torch.save(coder, os.path.join(args.checkpoint_dir, "darts_model.ckpt"))


def compress(args):
    model_ID = os.path.basename(args.checkpoint_dir[:-1])

    coder = get_coder(args)
    coder = nn.DataParallel(coder)

    # if args.train_extracted:
    #     model_name = "darts_model.ckpt"
    # else:
    #     model_name = "model.ckpt"

    # custom method for loading last checkpoint
    ckpt = torch.load(os.path.join(args.checkpoint_dir, "model.ckpt"), map_location='cpu')
    print("========================================================================\n"
          "Loading model checkpoint at directory: ", args.checkpoint_dir,
          "\n========================================================================")

    # ==== only for old version model loading ====
    # coder_ckpt = ckpt['coder']
    # new_ckpt = {}
    # for key, value in coder_ckpt.items():
    #     contents = key.split('.')
    #     if contents[-2] in ['conv', 'deconv', 'gdn', 'igdn']:
    #         key = '.'.join(contents[:-2]) + '.m.' + '.'.join(contents[-2:])
    #
    #     new_ckpt[key] = value
    #
    # ckpt['coder'] = new_ckpt
    # ============================================

    try:
        coder.load_state_dict(ckpt['coder'])
    except RuntimeError as e:
        # Warning(e)
        print(e)
        coder.load_state_dict(ckpt['coder'], strict=False)

    coder = coder.to(DEVICE)

    softmax = torch.nn.Softmax(dim=0)
    for mod in coder.module.analysis.model:
        conv_alpha_s = softmax(mod.alpha_conv)
        chosen_i = get_max_idx(conv_alpha_s)
        mod.chosen_conv = chosen_i
        mod.search = False
        acti_alpha_s = softmax(mod.alpha_acti)
        chosen_i = get_max_idx(acti_alpha_s)
        mod.chosen_acti = chosen_i
    for mod in coder.module.synthesis.model:
        deconv_alpha_s = softmax(mod.alpha_deconv)
        chosen_i = get_max_idx(deconv_alpha_s)
        mod.chosen_deconv = chosen_i
        mod.search = False
        acti_alpha_s = softmax(mod.alpha_acti)
        chosen_i = get_max_idx(acti_alpha_s)
        mod.chosen_acti = chosen_i

    print("Model ID:: {}".format(model_ID))

    # Create input data pipeline.
    test_dataset = CustomData(args.source_dir, transforms.ToTensor())
    test_dataloader = data.DataLoader(test_dataset, 1, False, num_workers=16)

    coder.eval()

    os.makedirs(args.target_dir, exist_ok=True)

    est_rate_list = []
    rate_list = []

    with torch.no_grad():
        for eval_img, img_path in test_dataloader:
            eval_img = eval_img.to(DEVICE)

            img_name = os.path.basename(img_path[0])
            stream_io = BitStreamIO(os.path.join(args.target_dir, img_name + ".ifc"))

            if args.Blur is not False:
                stream_list, shape_list, blur_scale = coder.module.compress(eval_img)
            else:
                stream_list, shape_list = coder.module.compress(eval_img)

            stream_io.prepare_shape(shape_list)
            stream_io.streams = stream_list
            stream_io.prepare_strings()
            stream_io.write_file()

            if args.Blur is not False:
                sns.heatmap(blur_scale, vmax=2., vmin=0., cbar=True, square=True, xticklabels=False, yticklabels=False)
                plt.savefig(args.target_dir + '/' + img_name[:-4] + '_ht.png')
                plt.close()

            eval_rate = os.path.getsize(stream_io.path) * 8 / (eval_img.size(2) * eval_img.size(3))

            _, eval_est_rate = coder(eval_img)

            print("{}:: est.rate: {:.4f} rate: {:.4f}".format(img_name, eval_est_rate.item(), eval_rate))

            est_rate_list.append(eval_est_rate.item())
            rate_list.append(eval_rate)

        print("==========avg. performance==========")
        print("est.rate: {:.4f} rate: {:.4f}".format(
            np.mean(est_rate_list),
            np.mean(rate_list)
        ))


def decompress(args):
    model_ID = os.path.basename(args.checkpoint_dir[:-1])

    coder = get_coder(args)
    coder = nn.DataParallel(coder)

    # if args.train_extracted:
    #     model_name = "darts_model.ckpt"
    # else:
    #     model_name = "model.ckpt"

    # custom method for loading last checkpoint
    ckpt = torch.load(os.path.join(args.checkpoint_dir, "model.ckpt"), map_location='cpu')
    print("========================================================================\n"
          "Loading model checkpoint at directory: ", args.checkpoint_dir,
          "\n========================================================================")

    try:
        coder.load_state_dict(ckpt['coder'])
    except RuntimeError as e:
        # Warning(e)
        print(e)
        coder.load_state_dict(ckpt['coder'], strict=False)

    coder = coder.to(DEVICE)

    softmax = torch.nn.Softmax(dim=0)
    for mod in coder.module.analysis.model:
        conv_alpha_s = softmax(mod.alpha_conv)
        chosen_i = get_max_idx(conv_alpha_s)
        mod.chosen_conv = chosen_i
        mod.search = False
        acti_alpha_s = softmax(mod.alpha_acti)
        chosen_i = get_max_idx(acti_alpha_s)
        mod.chosen_acti = chosen_i
    for mod in coder.module.synthesis.model:
        deconv_alpha_s = softmax(mod.alpha_deconv)
        chosen_i = get_max_idx(deconv_alpha_s)
        mod.chosen_deconv = chosen_i
        mod.search = False
        acti_alpha_s = softmax(mod.alpha_acti)
        chosen_i = get_max_idx(acti_alpha_s)
        mod.chosen_acti = chosen_i

    print("Model ID:: {}".format(model_ID))

    coder.eval()

    os.makedirs(args.target_dir, exist_ok=True)
    file_name_list = sorted(glob(os.path.join(args.source_dir, "*.ifc")))

    with torch.no_grad():
        eval_psnr_list = []
        eval_msssim_list = []
        eval_rate_list = []

        for file_name in file_name_list:
            img_name = os.path.basename(file_name)[:-4]

            stream_io = BitStreamIO(file_name)
            stream_io.read_file()
            stream_io.extract_strings()

            eval_img_tilde = coder.module.decompress(stream_io.streams, stream_io.shape_list)
            eval_img_tilde *= 255
            eval_img_tilde = eval_img_tilde.clamp(0., 255.).round().permute(0, 2, 3, 1).cpu()[0].numpy()

            io.imsave(os.path.join(args.target_dir, img_name), eval_img_tilde.astype(np.uint8))

            if args.eval:
                eval_img = io.imread(os.path.join(args.original_dir, img_name))

                eval_psnr = PSNR_np(eval_img, eval_img_tilde, data_range=255.)
                eval_msssim = MultiScaleSSIM(eval_img[None], eval_img_tilde[None])
                eval_rate = os.path.getsize(stream_io.path) * 8 / (eval_img.shape[0] * eval_img.shape[1])

                print("{}:: PSNR: {:.4f}, MS-SSIM: {:.4f}, rate: {:.4f}".format(
                    img_name, eval_psnr, eval_msssim, eval_rate
                ))

                eval_psnr_list.append(eval_psnr)
                eval_msssim_list.append(eval_msssim)
                eval_rate_list.append(eval_rate)

        if args.eval:
            print("==========avg. performance==========")
            print("PSNR: {:.4f}, MS-SSIM: {:.4f}, rate: {:.4f}".format(
                np.mean(eval_psnr_list),
                np.mean(eval_msssim_list),
                np.mean(eval_rate_list),
            ))


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report bitrate and distortion when training or compressing.")
    parser.add_argument(
        "--num_features", "-NF", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--num_filters", "-NFL", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--num_hyperpriors", "-NHP", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--checkpoint_dir", "-ckpt", default=None,
        help="Directory where to save/load model checkpoints.")
    parser.add_argument(
        "--Mean", "-M", action="store_true",
        help="Enable hyper-decoder to output predicted mean or not.")
    parser.add_argument(
        "--Octave", "-O", action="store_true",
        help="Enable Octave feature or not.")
    parser.add_argument(
        "--Blur", "-B", default=False,
        choices=['False', 'normal', 'scaleAE', 'scalePyramid', 'distance', 'duoPyramid', 'kernelPyramid'],
        help="The Gaussian blur scheme apply on latent feature.")
    parser.add_argument(
        "--Blur_Mode", "-BM", default=None, choices=['single', 'multi'],
        help="The Gaussian blur mode apply on latent feature.")
    parser.add_argument(
        "--darts_warmup_epoch", "-DWE", type=int, default=15,
        help="Train network weights for warm up before searching for structure.")

    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: \n"
             "'train' loads training data and trains (or continues to train) a new model.\n"
             "'compress' reads an image file (lossless PNG format) and writes a compressed binary file.\n"
             "'decompress' reads a binary file and reconstructs the image (in PNG format).\n"
             "input and output filenames need to be provided for the latter two options.\n\n"
             "Invoke '<command> -h' for more information.")

    # 'train' sub-command.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model.")
    train_cmd.add_argument(
        "--batch_size", type=int, default=30,
        help="Batch size for training.")
    train_cmd.add_argument(
        "--patch_size", type=int, default=128,
        help="Size of image patches for training.")
    train_cmd.add_argument(
        "--lambda", type=float, default=0.01, dest="lmda",
        help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument(
        "--max_epochs", type=int, default=200,
        help="Train up to this number of steps.")
    train_cmd.add_argument(
        "--use_radam", action="store_true",
        help="Use RAdam or not.")
    train_cmd.add_argument(
        "--resume", action="store_true",
        help="Whether to resume on previous checkpoint")
    train_cmd.add_argument(
        "--train_extracted", action="store_true",
        help="Whether to train an extracted structure")
    train_cmd.add_argument(
        "--reuse_ckpt", action="store_true",
        help="Whether to reuse the specified checkpoint, if not, "
             "we'll resume the current model in the new created directory.")
    train_cmd.add_argument(
        "--reuse_args", action="store_true",
        help="Whether to reuse the original arguments")
    train_cmd.add_argument(
        "--fixed_base", action="store_true",
        help="To fixed the baseline model and only the rest can change.")
    train_cmd.add_argument(
        "--fixed_enc", action="store_true",
        help="To fixed the baseline encoder and only the rest can change.")
    train_cmd.add_argument(
        "--pretrain_ae", action="store_true",
        help="To fixed the baseline encoder and only the rest can change.")
    train_cmd.add_argument(
        "--bilevel", default=True,
        help="Whether to use bi-level optimization for DARTS")

    # 'compress' sub-command.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compress images with a trained model.")
    compress_cmd.add_argument(
        "--source_dir", "-SD",
        help="The directory of the images that are expected to compress.")
    compress_cmd.add_argument(
        "--target_dir", "-TD",
        help="The directory where the compressed files are expected to store at.")

    # 'decompress' sub-command.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Decompress bitstreams with a trained model.")
    decompress_cmd.add_argument(
        "--source_dir", "-SD",
        help="The directory of the compressed files that are expected to decompress.")
    decompress_cmd.add_argument(
        "--target_dir", "-TD",
        help="The directory where the images are expected to store at.")
    decompress_cmd.add_argument(
        "--eval", action="store_true",
        help="Evaluate decompressed images with original ones.")
    decompress_cmd.add_argument(
        "--original_dir", "-OD", nargs="?",
        help="The directory where the original images are expected to store at.")

    # 'val' sub-command.
    val_cmd = subparsers.add_parser(
        "val",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Validate a trained model.")
    val_cmd.add_argument(
        "--source_dir", "-SD",
        help="The directory of the images that are expected to compress.")
    val_cmd.add_argument(
        "--bitstream_dir", "-BD",
        help="The directory where the compressed files are expected to store at.")
    val_cmd.add_argument(
        "--target_dir", "-TD",
        help="The directory where the decoded images are expected to store at.")
    val_cmd.add_argument(
        "--original_dir", nargs="?",
        help="This is a placeholder for the program to pass argument to decompress module.")
    val_cmd.add_argument(
        "--eval", action="store_true", default=True,
        help="This is a placeholder for the program to pass argument to decompress module.")

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    if args.command == "train":
        if args.checkpoint_dir is None:
            args.checkpoint_dir = gen_log_folder_name()

        if (not args.resume) and (not args.train_extracted):
            assert args.checkpoint_dir is not None
            print("========================================================================\n"
                  "Creating model checkpoint at directory: ", args.checkpoint_dir,
                  "\n========================================================================")
            os.makedirs(args.checkpoint_dir, exist_ok=False)

            # Config dump
            dump_args(args, args.checkpoint_dir)

        else:
            assert args.checkpoint_dir is not None, "Checkpoint directory must be specified. [-ckpt=path/to/your/model]"

            if not args.reuse_ckpt:
                old_ckpt_dir = args.checkpoint_dir
                args.checkpoint_dir = gen_log_folder_name()
                os.makedirs(args.checkpoint_dir, exist_ok=False)

                print("========================================================================\n"
                      "Creating model checkpoint at directory: ", args.checkpoint_dir,
                      "\nCopying model checkpoint from ", old_ckpt_dir)

                assert os.path.exists(os.path.join(old_ckpt_dir, "model.ckpt")), \
                    os.path.join(old_ckpt_dir, "model.ckpt") + " not exist"
                shutil.copy(os.path.join(old_ckpt_dir, "model.ckpt"), os.path.join(args.checkpoint_dir, "model.ckpt"))
                # if args.train_extracted:
                #     assert os.path.exists(os.path.join(old_ckpt_dir, "darts_model.ckpt")), \
                #     os.path.join(old_ckpt_dir, "darts_model.ckpt") + " not exist"
                #     shutil.copy(os.path.join(old_ckpt_dir, "darts_model.ckpt"), os.path.join(args.checkpoint_dir, "darts_model.ckpt"))
            else:
                print("========================================================================\n")

        train(args)
    elif args.command == "compress":
        compress(args)
    elif args.command == "decompress":
        decompress(args)
    elif args.command == "val":
        input_dir = args.source_dir
        stream_dir = args.bitstream_dir
        output_dir = args.target_dir

        args.target_dir = stream_dir
        compress(args)

        args.source_dir = stream_dir
        args.target_dir = output_dir
        args.original_dir = input_dir
        args.eval = True
        decompress(args)


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)

