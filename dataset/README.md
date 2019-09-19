# PET dataset

### classes
```bash
Abyssinian
Bengal
Birman
Bombay
Egyptian_Mau
Maine_Coon
Persian
Ragdoll
Russian_Blue
Sphynx
american_pit_bull_terrier
basset_hound
beagle
boxer
chihuahua
english_cocker_spaniel
english_setter
german_shorthaired
great_pyrenees
havanese
japanese_chin
keeshond
leonberger
miniature_pinscher
newfoundland
pomeranian
pug
saint_bernard
samoyed
scottish_terrier
shiba_inu
staffordshire_bull_terrier
wheaten_terrier
yorkshire_terrier
```

### script
```bash
pip3 install wget # if you use docker env, you don't use it.
cd datset/original
bash download_pet_dataset.sh
python3 removeNotRGB.py
cd ..
bash split_train_val_images.sh
```
 - download_pet_dataset.sh
    - download pet dataset and extract tar.gz
    
 - python3 removeNotRGB.py
  - remove files which are mat type file
  - remove RGBA and gray color space files.
  
 - split_train_val_images.sh
   - 90% train data
   - 10% validation data 
   
### Folder Structure
```bash 
$tree -d
.
├── original
│   ├── annotations
│   │   ├── trimaps
│   │   └── xmls
│   └── images
├── tfrecords
├── train
│   ├── Abyssinian
│   ├── Bengal
│   ├── Birman
................................
│   ├── shiba_inu
│   ├── staffordshire_bull_terrier
│   ├── wheaten_terrier
│   └── yorkshire_terrier
└── validation
    ├── Abyssinian
    ├── Bengal
    ├── Birman
................................
    ├── shiba_inu
    ├── staffordshire_bull_terrier
    ├── wheaten_terrier
    └── yorkshire_terrier

```

### Requirements
 - wget

### Reference
 - https://www.robots.ox.ac.uk/~vgg/data/pets/
