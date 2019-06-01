import wget
import os
import tarfile
import shutil

def tar_open(filename):
  tar = tarfile.open(filename)
  tar.extractall()
  tar.close()


mobilenet_v2_ckpt_url = "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz"
mobilenet_v2_tar_file="mobilenet_v2_1.0_224.tgz"
mobilenet_v2_ckpt_dir="mobilenet_v2_1.0_224"

if not os.path.isfile(mobilenet_v2_tar_file):
  image_tar_file = wget.download(mobilenet_v2_ckpt_url)
  print("file download : %", mobilenet_v2_tar_file)

if not os.path.isdir(mobilenet_v2_ckpt_dir):
  os.mkdir(mobilenet_v2_ckpt_dir)
  shutil.move(mobilenet_v2_tar_file, mobilenet_v2_ckpt_dir)
  os.chdir(mobilenet_v2_ckpt_dir)

if not os.path.isdir(mobilenet_v2_tar_file):
    tar_open(mobilenet_v2_tar_file)

