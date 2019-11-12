# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:08:18 2019

@author: hello
"""

from image_mod_gen_utils import *


path="D:\\project\\analyzeNN\\pics\\roads\\tree.jpg"

im=Image.open(path)

img=scale_img(im,0.2)

img.show()