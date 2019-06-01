import os
import shutil
import random

# print (os.getcwd()) #현재 디렉토리의
# print (os.path.realpath(__file__))#파일
# print (os.path.dirname(os.path.realpath(__file__)) )#파일이 위치한 디렉토리

# cur_path = os.path.dirname(os.path.realpath(__file__))
cur_path = '/home/jamin/projects/Code/Classification/dataset'

lable_file = os.path.join(cur_path, "labels.txt")
train_dir = os.path.join(cur_path, "train")
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(cur_path, "validation")
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

label_file = open(lable_file, "r")
lines = label_file.readlines()
for idx, line in enumerate(lines):
    lines[idx] = line.rstrip("\n")

#make label directorys
for line in lines:
    target_path = os.path.join(train_dir, line)
    os.mkdir(target_path)
for line in lines:
    target_path = os.path.join(validation_dir, line)
    os.mkdir(target_path)

src = os.path.join(cur_path, "original/images/")
dst_train = os.path.join(cur_path, "train/")
dst_validation = os.path.join(cur_path, "validation/")

lable_file_list = {}
for line in lines:
    lable_file_list[line] = []

total_file_list = os.listdir(src)
for idx in total_file_list:
    for line in lines:
        if line in idx:
            lable_file_list[line].append(idx)

#copy label dataset
for line in lines:
    random.shuffle(lable_file_list[line])
    val_num = len(lable_file_list[line]) * 0.1
    val_num = int(val_num)
    file_list_train = lable_file_list[line][val_num:-1]
    file_list_validation = lable_file_list[line][0:val_num]
    for file_list in file_list_train:
        shutil.copy2(os.path.join(src, file_list), os.path.join(dst_train, line))
    for file_list in file_list_validation:
        shutil.copy2(os.path.join(src, file_list), os.path.join(dst_validation, line))