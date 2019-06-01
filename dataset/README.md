# PET dataset

### script
```bash
pip3 install wget # if you use docker env, you don't use it.
cd datset/original
bash download_pet_dataset.sh
python3 removeNotRGB.sh
cd ..
bash split_train_val_images.sh
```
 - download_pet_dataset.sh
    - download pet dataset and extract tar.gz
    
 - python3 removeNotRGB.sh
  - remove files which are mat type file
  - remove RGBA and gray color space files.
  
 - split_train_val_images.sh
   - 90% train data
   - 10% validation data 

### Requirements
 - wget

### Reference
 - https://www.robots.ox.ac.uk/~vgg/data/pets/
