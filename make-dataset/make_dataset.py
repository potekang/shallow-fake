#The code based on https://github.com/marcin7Cd/variant-of-CPPN-GAN/blob/master/chinese.py 
#Take the first arguement as target forder
#The forder should have [name]_fonts directory with pure fonts and [name]_text file
# an example: python3 make_dataset.py japanese.
#coding=utf-8
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys

font_size=60
size_x =64
size_y =64
#language = "japanese"
language = sys.argv[1]
lang_path =  './'+language+'/'
output_path = './dataset/' + language + '/'

#read font list from font path
font_path = lang_path + language + '_fonts/'
fonts = []
font_list = os.listdir(font_path)
for f in font_list:
    temp_dir = font_path+f
    temp = ImageFont.truetype(temp_dir, font_size, 0)
    fonts.append(temp)


#read target text from file
with open (lang_path + language + '_text', encoding = "utf-8") as textfile:
    chars = textfile.read().replace('\n', '')
charset = list(chars)


def block(image, mode, size):
    if mode == 0:# move the character block
        which = np.array([0,0,0,0])
        which[np.random.randint(0, 3)] = np.random.randint(0, 4)
        which[np.random.randint(0, 3)] = np.random.randint(0, 4)
        image = cv2.copyMakeBorder(image,which[0],which[0],which[0],which[0],cv2.BORDER_CONSTANT,value=0)
        image = cv2.resize(image,size)
    if mode == 1: #random rescaling
        size_xy = np.max(size_x,size_y)
        scale = np.random.randint(size_xy,int(size_xy*1.2))
        center = [scale/2,scale/2]
        image = cv2.resize(image, (scale,scale))
        image = image[int(center[0]-size_xy//2):int(center[0]+size_xy//2),int(center[1]-size_xy//2):int(center[1]+size_xy//2)]
    return image

char_id = np.random.randint(0,len(charset))
font_id = np.random.randint(0,len(fonts))

def generate_image(char_id, font_id):
    image = np.zeros(shape=(size_x,size_y), dtype=np.uint8)
    start = np.random.randint(-1,2, size=(2))
    x = Image.fromarray(image)
    draw = ImageDraw.Draw(x)
    draw.text(start,charset[char_id],(255),font=fonts[font_id])
    p = np.array(x)
    image = p
    mode = np.random.randint(0,1)
    size = (size_x,size_y)
    image = block(image, 0, size)


    return image

def generate_images(number = 81,part_taken=0.9999, to_file = False, file = ''):
    images = []
    for i in range(number):
        char_id = np.random.randint(len(charset)*part_taken,len(charset))
        font_id = np.random.randint(0,len(fonts))
        cv2.imwrite(file+language+str(i)+'.png', 
                        generate_image(char_id,font_id))

if __name__ == "__main__":
    directory = os.path.dirname(output_path)
    #creat output directory if not exist
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    generate_images(number = 16,part_taken=0.014, to_file=True, file=output_path)
      

