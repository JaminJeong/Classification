find ./original/images -type f | grep -v jpg | xargs rm 

python3 split_train_val_images.py
