import os
import random

a = os.listdir('./')

random.shuffle(a)

lt = a[:int(len(a)*0.2)]

for i in lt:
  print(i)
