import wget
import os
import tarfile

def tar_open(filename):
  tar = tarfile.open(filename)
  tar.extractall()
  tar.close()


image_url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
annotations_url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
image_tar_file="images.tar.gz"
annotations_tar_file="annotations.tar.gz"
image_dir="images.tar.gz"
annotations_dir="annotations.tar.gz"

if not os.path.isfile(image_tar_file):
  image_tar_file = wget.download(image_url)
  print("file download : %", image_tar_file)

if not os.path.isfile(annotations_tar_file):
  annotations_tar_file = wget.download(annotations_url)
  print("file download : %", annotations_tar_file)

if not os.path.isdir(image_dir):
  tar_open(image_tar_file)
if not os.path.isdir(annotations_dir):
  tar_open(annotations_tar_file)

