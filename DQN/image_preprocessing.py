#import opencv as cv
import numpy as np
import skimage


def to_Ychannel(image):
    """
    Reduces the dimentionnality of the input image for dqn by converting it to
    greyscale, trimming to and bottom and performing a maxpooling to reduce the
    image size.
    """
    image = np.array(image)
    to_Y = np.array([0.299, 0.587, 0.114])
    image = np.dot(image, to_Y)
    image = image[34:194]
    image = skimage.measure.block_reduce(image, (2, 2), np.max)
    #image = (image-80) / (255-80) 
    image = np.expand_dims(image, axis=-1)
    return np.array(image)
