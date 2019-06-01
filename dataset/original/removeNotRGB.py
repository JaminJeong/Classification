from PIL import Image     
import os       
import shutil       
path = os.path.join(os.getcwd(), "images")
for file in os.listdir(path):      
  extension = file.split('.')[-1]
  if extension == 'jpg':
    fileLoc = path+"/"+file
    img = Image.open(fileLoc)
    if img.mode != 'RGB':
      print(file+', '+img.mode)
      img.close()
      print('remove file : ' + file+', '+img.mode)
      os.remove(fileLoc)
