import os
from glob import glob
from skimage import io
from subprocess import DEVNULL, STDOUT, Popen
import subprocess
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

img_dir = "COCO/"
img_ext = "jpg"
qp = 30
num_cores = multiprocessing.cpu_count()

image_list = sorted(glob(img_dir + "*." + img_ext))

os.makedirs("{}yuv/".format(img_dir), exist_ok=True)
os.makedirs("{}qp_{:02d}/".format(img_dir, qp), exist_ok=True)
os.makedirs("{}bits/".format(img_dir), exist_ok=True)


def compress_single(image_path):
    img = io.imread(image_path)

    img_shape = img.shape

    if len(img_shape) == 3:
        h, w, _ = img_shape
    else:
        h, w = img_shape

    basename = os.path.splitext(os.path.basename(image_path))[0]
    command = "ffmpeg -i {}{}.{} -pix_fmt yuv444p -hide_banner -loglevel panic -r 1 {}yuv/{}.yuv".format(img_dir,
                                                                                                         basename,
                                                                                                         img_ext,
                                                                                                         img_dir,
                                                                                                         basename)

    if not os.path.exists(img_dir + "yuv/" + basename + ".yuv"):
        Popen(command.split(' '), stdout=DEVNULL, stderr=STDOUT)

    command = "EncoderApp -i {}yuv/{}.yuv -c i_frame.cfg -fr 1 -f 1 -wdt {:d} -hgt {:d} --InputChromaFormat=444 -o {}qp_{:02d}/{}.yuv -b {}bits/{}.bin -v 9999 -q {:d}".format(
        img_dir, basename, w, h, img_dir, qp, basename, img_dir, basename, qp)

    if not (os.path.exists(img_dir + "qp_" + str(qp) + "/" + basename + ".yuv") and os.path.exists(
            img_dir + "bits/" + basename + ".bin")):
        p = Popen(command.split(' '), stdout=DEVNULL, stderr=STDOUT)

        try:
            p.wait(timeout=60 * 30)
        except subprocess.TimeoutExpired:
            p.kill()

    command = "ffmpeg -s {:d}x{:d} -r 1 -pix_fmt yuv444p -hide_banner -loglevel panic -i {}qp_{:02d}/{}.yuv {}qp_{:02d}/{}.png".format(
        w, h, img_dir, qp, basename, img_dir, qp, basename)

    if not os.path.exists(img_dir + "qp_" + str(qp) + "/" + basename + ".png"):
        Popen(command.split(' '), stdout=DEVNULL, stderr=STDOUT)


if __name__ == '__main__':
    Parallel(n_jobs=num_cores)(delayed(compress_single)(img_path) for img_path in tqdm(image_list, total=len(image_list)))
