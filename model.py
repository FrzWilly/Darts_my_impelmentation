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
from network import GoogleHyperPriorCoder
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

Dataset_dir = "/work/dataset/"
TrainData = "MSCOCO"


def get_coder(args):
    # Instantiate model.
    if args.Octave:
        coder = OctGoogleHPCoder(args.num_filters // 2, args.num_features // 2, args.num_hyperpriors // 2)
    else:
        coder = GoogleHyperPriorCoder(args)

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
    if args.resume:

        # custom method for loading last checkpoint
        ckpt = torch.load(os.path.join(args.checkpoint_dir, "model.ckpt"), map_location='cpu')
        print("========================================================================")

        if args.reuse_args:
            args = load_args(args, args.checkpoint_dir)

        start_epoch = ckpt['epoch'] + 1

        # ==== only for old version model loading ====
        coder_ckpt = ckpt['coder']
        new_ckpt = {}
        for key, value in coder_ckpt.items():
            # contents = key.split('.')
            # if contents[-2] in ['conv', 'deconv', 'gdn', 'igdn']:
            #     key = '.'.join(contents[:-2]) + '.m.' + '.'.join(contents[-2:])
            # key = 'module.' + key

            new_ckpt[key] = value

        ckpt['coder'] = new_ckpt
        # ============================================

        try:
            coder.load_state_dict(ckpt['coder'])
        except RuntimeError as e:
            # Warning(e)
            print(e)
            coder.load_state_dict(ckpt['coder'], strict=False)

        coder.module.morphize()
        coder = coder.to(DEVICE)

        if args.pretrain_ae:
            optim2 = torch.optim.Adam([torch.Tensor(0)])
        elif args.Octave:
            optim2 = torch.optim.Adam([coder.module.entropy_bn_h.quantiles.requires_grad_(),
                                       coder.module.entropy_bn_l.quantiles.requires_grad_()], lr=3e-4)
        else:
            optim2 = torch.optim.Adam([coder.module.entropy_bottleneck.quantiles.requires_grad_()], lr=3e-4)


        # optim.load_state_dict(ckpt['optim'])
        optim2.load_state_dict(ckpt['optim2'])

        if not args.verbose:
            evaluate(0, is_resume=True)

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

    for epoch in range(start_epoch, args.max_epochs):
        coder.train()

        writer_record = {
            'rd_loss': [],
            'rate': [],
            'distortion': [],
            'morph_loss': []
        }

        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        start_time = time.time()

        # for loop going through dataset
        for idx, imgs in enumerate(pbar):
            imgs = imgs.to(DEVICE)

            optim.zero_grad()

            imgs_tilde, rate = coder(imgs)

            distortion = torch.mean(torch.pow(imgs - imgs_tilde, 2), dim=[1, 2, 3])
            distortion *= (255 ** 2)
            distortion = torch.clamp_max(distortion, 1e8)

            rate = torch.squeeze(rate)
            distortion = torch.squeeze(distortion)

            rd_loss = torch.mean(args.lmda * distortion + rate)

            morph_loss = coder.module.morph_loss()
            rd_loss += 2 * morph_loss;

            prepare_time = start_time - time.time()

            rd_loss.backward()
            optim.step()

            if not args.pretrain_ae:

                optim2.zero_grad()

                if args.Octave:
                    aux_loss = coder.module.entropy_bn_h.aux_loss() + coder.entropy_bn_l.aux_loss()
                else:
                    aux_loss = coder.module.entropy_bottleneck.aux_loss()

                aux_loss.backward()
                optim2.step()

            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}, dis: {:.3f}, rate: {:.3f}, morph: {:.3f} epoch: {}/{}:".format(
                process_time / (process_time + prepare_time), distortion.mean(), rate.mean(), morph_loss.item(), epoch, args.max_epochs))

            writer_record['rd_loss'].append(rd_loss.cpu().item())
            writer_record['rate'].append(rate.mean().cpu().item())
            writer_record['distortion'].append(distortion.mean().cpu().item())
            writer_record['morph_loss'].append(morph_loss.cpu().item())

            start_time = time.time()

        log_writer.add_scalar('Loss/total', np.mean(writer_record['rd_loss']), epoch)
        log_writer.add_scalar('Loss/distortion', np.mean(writer_record['distortion']), epoch)
        log_writer.add_scalar('Loss/rate', np.mean(writer_record['rate']), epoch)
        log_writer.add_scalar('Loss/morph', np.mean(writer_record['morph_loss']), epoch)

        # Save model
        torch.save({
            'epoch': epoch,
            'coder': coder.state_dict(),
            'optim': optim.state_dict(),
            'optim2': optim2.state_dict()
        }, os.path.join(args.checkpoint_dir, "model.ckpt"))

        # Testing
        if not args.verbose:
            print("Model ID:: {}".format(model_ID))
            evaluate(epoch)
        coder.module.morph_status()
        

    coder.module.demorphize()
    torch.save(coder, os.path.join(args.checkpoint_dir, "morphized_model.ckpt"))


def compress(args):
    model_ID = os.path.basename(args.checkpoint_dir[:-1])

    coder = get_coder(args)
    coder = nn.DataParallel(coder)

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

        if not args.resume:
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

