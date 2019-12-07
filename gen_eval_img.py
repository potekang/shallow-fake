# The code randomly sample 1000 from dataset and output to /evaluation/[language] file
import numpy as np
import os
from PIL import Image
import glob
import cv2
import pylib as py
import imlib as im

language = 'japanese'

train_dir = "./dataset/" + language
output_dir = "./evaluation/training"
file_list = os.listdir(train_dir)
imgs = []
for file in file_list:
    img = cv2.imread(os.path.join(train_dir, file), 0)
    imgs.append(img)
np.random.shuffle(imgs)
#print(np.shape(imgs[1]))
for i in range(0,1000):
    cv2.imwrite(py.join(output_dir, language + '%03d.jpg' %i), imgs[i])
print(np.shape(imgs))
