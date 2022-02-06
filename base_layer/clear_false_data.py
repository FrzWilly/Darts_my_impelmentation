from glob import glob
import os

img_dir = 'COCO/'

ori_yuv_list = sorted(glob(img_dir + 'yuv/*.yuv'))
bin_list = sorted(glob(img_dir + 'bits/*.bin'))
yuv_list = sorted(glob(img_dir + 'qp_45/*.yuv'))

for idx, path in enumerate(ori_yuv_list):
    bin_name = img_dir + 'bits/' + os.path.splitext(os.path.basename(path))[0] + '.bin'
    yuv_name = img_dir + 'qp_45/' + os.path.splitext(os.path.basename(path))[0] + '.yuv'

    if os.path.exists(bin_name) and os.path.getsize(bin_name) == 0:
        print('deleting', os.path.basename(bin_name))
        os.remove(bin_name)

    if os.path.exists(yuv_name) and os.path.getsize(yuv_name) == 0:
        print('deleting', os.path.basename(yuv_name))
        os.remove(yuv_name)
