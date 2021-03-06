'''
This file has all the basic image modification primitives
'''

from PIL import Image, ImageEnhance
from image_generation_utils import *

#这个函数主要是修改图片的比例，其中img是图片信息，scale是缩放或者是扩大的
def scale_img(img, scale):
    return img.resize((np.array(img.size) * scale).astype(int))


#这个函数的主要目的就是对图片进行缩放和修改，估计是后文中对car的处理的问题
def scale_get_loc(img, scale, centroid):
    scaled_img = scale_img(img, scale)
    top_right_loc = (np.array(centroid) - \
                    np.array(scaled_img.size) * 0.5).astype(int)
    return scaled_img, top_right_loc


# Update with brightness, sharpness, contrast and color
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

# Get samples and generate image
# x is the lateral displacement
# y is the vertical displacement
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

    # Add foreground images，从这里可以看出，增加的是前面的背景图片，
    #前景主要是车辆信息，这部分涉及到车辆的缩小，x的位置移植和y的位置的移植
    #还有就是前面的采样和渲染的数据
    boxes = []
    for i, fg_i in zip(range(len(fg_objects)), fg_objects):
        x, y, fg = fg_i.x, fg_i.y, fg_i.fg_id
        scale_fg = y * (scaling_factor.back - scaling_factor.front) + \
                   scaling_factor.front
        sample_conv_space = ld_to_bb_sample(sample=[x,y],
                                            h=background.homography_h)

        foreground = library.foreground_objects[fg]

        scaled_img, top_right_loc = scale_get_loc(foreground.image, scale_fg, \
                                                  sample_conv_space)

        # paste car  把车辆信息放里边去
        background_no_alpha.paste(scaled_img, tuple(top_right_loc), scaled_img)

        # store labels  存储的是车辆的ground truth信息
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
