# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:08:18 2019

@author: hello
"""

from image_mod_gen_utils import *

from update_library import *

from PIL import Image, ImageEnhance

'''
path="D:\\project\\analyzeNN\\pics\\roads\\tree.jpg"

im=Image.open(path)

img=scale_img(im,0.2)

img.show()

library=update_library()
'''
trapezoid=bb([350,400], [400,400], [400,300], [350,300])
scale = namedtuple('scale', ['front', 'back'])    #对应的分别是前进

scale = scale(front=1,back=0.9)

lib = update_library(trapezoid,scale)
library=lib


def modify_image_bscc(image_data, brightness, sharpness, contrast, color):
    brightness_mod = ImageEnhance.Brightness(image_data)
    image_data = brightness_mod.enhance(brightness)

    sharpness_mod = ImageEnhance.Sharpness(image_data)
    image_data = sharpness_mod.enhance(sharpness)

    contrast_mod = ImageEnhance.Contrast(image_data)
    image_data = contrast_mod.enhance(contrast)

    color_mod = ImageEnhance.Color(image_data)
    image_data = color_mod.enhance(color)

    return image_data


def scale_img(img, scale):
    return img.resize((np.array(img.size) * scale).astype(int))

#这个函数可得到,对应的图片以及
def scale_get_loc(img, scale, centroid):
    scaled_img = scale_img(img, scale)
    top_right_loc = (np.array(centroid) - \
                    np.array(scaled_img.size) * 0.5).astype(int)
    return scaled_img, top_right_loc

def scale_image(originalObject, scale):
    scaledData = originalObject.data.resize([int(x) for x in np.multiply(originalObject.data.size,np.full((1,2),scale)[0])])
    if originalObject.componentType == 'Road':
        updatedVP = scaleCoord(originalObject.vp, np.full((1,2),scale)[0])
        updatedMIN_X = scaleCoord(originalObject.min_x, np.full((1,2),scale)[0])
        updatedMAX_X = scaleCoord(originalObject.max_x, np.full((1,2),scale)[0])
        updatedLANES = []
        for i in range(len(originalObject.lanes)):
            updatedLANES.append(scaleCoord(originalObject.lanes[i], np.full((1,2),scale)[0]))
        return comp.road(ImageFile(scaledData, originalObject.description), updatedVP, updatedMIN_X, updatedMAX_X, updatedLANES)

    elif originalObject.componentType =='Car':
        return comp.car(ImageFile(scaledData, originalObject.description))
    
def box_2_kitti_format(box):
    '''Transform box for KITTI label format'''
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    left = int(x - w/2)
    right = int(x + w/2)
    top = int(y - h/2)
    bot = int(y + h/2)
    return [left,top,right,bot]

def save_label(ground_boxes, file_name, path_data_set):
    '''Save label'''
#这个函数是ground_boxes的函数，这个函数对应的是车辆的位置信息
    f = open(path_data_set + 'labels/' + file_name + '.txt', 'w')

    if len(ground_boxes) > 0:
        for box in ground_boxes[:-1]:
            label = [0,0,0] + box_2_kitti_format(box) + [0,0,0,0,0,0,0]
            label = list(map(str, label))
            label = " ".join(label)
            label  = "Car " + label + "\n"
            f.write(label)  # python will convert \n to os.linesep
        label = [0,0,0] + box_2_kitti_format(ground_boxes[-1]) + [0,0,0,0,0,0,0]
        label = list(map(str, label))
        label = " ".join(label)
        label  = "Car " + label
        f.write(label)  # python will convert \n to os.linesep
    f.close()

def save_image(img, file_name, path_data_set):
    '''Save image and label'''

    img_file_name = path_data_set + 'images/' + file_name + '.png'
    img.save(img_file_name)
    
    
import random
def random_sample(typ, domain, n_samples):

    sample = []
    for _ in range(n_samples):
        if typ == 'float':
            r = random.random()*(domain[1]-domain[0])+domain[0]
        elif typ == 'int':
            r = random.choice(range(domain[0],domain[1]))
        else:
            print('Error')
        sample.append(r)
    return sample    
    
    
    
    
    
def random_config(domains, n_cars):
#     print("domains is : ", domains)# [[52, 72], [1, 19], [0, 1], [0.35, 1], [0.8, 1.2], [0.7, 1]]
#     print("n_cars is : ", n_cars)
    # Scene and cars
    sample = [random_sample('int',domains[0],1)]
    sample += [random_sample('int',domains[1],n_cars)]
    # Modifications
    x_sample = []
    y_sample = []
    if n_cars > 0:
        step_x = float(domains[2][1])/n_cars
        step_y = float(domains[3][1])/n_cars

        base_x = 0
        base_y = 0
        for _ in range(n_cars):
            x_sample.append(random.uniform(base_x, base_x+step_x))
            y_sample.append(random.uniform(base_y, base_y+step_y))
            base_x += step_x
            base_y += step_y

    random.shuffle(x_sample)
    y_sample.sort(reverse=True)
    sample += [x_sample]
    sample += [y_sample]

    sample += [random_sample('float',domains[4],1)]
    sample += [random_sample('float',domains[5],1)]

    return sample  
    
    
    
    
    
def gen_image(sample):
    fg = []
    for s in range(len(sample[1])):
        fg += [fg_obj(fg_id=sample[1][s], x=sample[2][s], y=sample[3][s])]
    
    #return gen_comp_img(lib, fg, bg_id=sample[0][0], brightness=sample[4][0], sharpness=sample[5][0], contrast=sample[5][0], color=sample[5][0])

    return gen_comp_img(lib, fg, bg_id=0, brightness=sample[4][0], sharpness=sample[5][0], contrast=sample[5][0], color=sample[5][0])




def gen_comp_img(library, fg_objects, bg_id=0, brightness=1., sharpness=1.,\
                 contrast= 1., color=1.):
    background = library.background_objects[bg_id]
    scaling_factor = background.scaling

    background_copy = background.image.copy()

    # remove alpha channel from background (if present)
    if background_copy.mode in ('RGBA', 'LA') or \
            (background_copy.mode == 'P' and
                     'transparency' in background_copy.info):
        background_no_alpha = \
            Image.new("RGB", background_copy.size, (255, 255, 255))
        background_no_alpha.paste(background_copy,
                                  mask=background_copy.split()[3])
                                # 3 is the alpha channel
    else:
        background_no_alpha = background_copy

    pic_dict = background.add_details.copy()
    pic_dict['brightness_sample'] = brightness
    pic_dict['sharpness_sample'] = sharpness
    pic_dict['contrast_sample'] = contrast
    pic_dict['color_sample'] = color

    # Add foreground images
    boxes = []
    for i, fg_i in zip(range(len(fg_objects)), fg_objects):
        x, y, fg = fg_i.x, fg_i.y, fg_i.fg_id
        fg=0
        scale_fg = y * (scaling_factor.back - scaling_factor.front) +scaling_factor.front
        sample_conv_space = ld_to_bb_sample(sample=[x,y],
                                            h=background.homography_h)

        foreground = library.foreground_objects[fg]

        scaled_img, top_right_loc = scale_get_loc(foreground.image, scale_fg, \
                                                  sample_conv_space)

        # paste car
        background_no_alpha.paste(scaled_img, tuple(top_right_loc), scaled_img)

        # store labels,这部分是存储标签数据
        int_centroid = list(sample_conv_space.astype(int))
        list_size = list(scaled_img.size)

        boxes.append(int_centroid+list_size)
        pic_dict['foreground' + str(i) + '_x'] = x
        pic_dict['foreground' + str(i) + '_y'] = y
        pic_dict['foreground' + str(i) + '_height'] = scaled_img.size[0]
        pic_dict['foreground' + str(i) + '_width'] = scaled_img.size[0]
        for k in foreground.add_details:
            pic_dict['foreground' + str(i) + k] = foreground.add_details[k]

    modif_img= modify_image_bscc(image_data=background_no_alpha,
                             brightness=brightness, sharpness=sharpness,
                             contrast=contrast, color=color)

    return modif_img, boxes, pic_dict




if __name__=="__main__":
    domains=[[52, 72], [1, 19], [0, 1], [0.35, 1], [0.8, 1.2], [0.7, 1]]
    n_cars=1
    sample=random_config(domains, n_cars)
    a=gen_image(sample)
   
    modif_img, boxes, pic_dict=gen_comp_img(library, fg_objects, bg_id=0, brightness=1., sharpness=1.,\
                 contrast= 1., color=1.)