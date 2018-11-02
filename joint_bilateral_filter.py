import cv2 
import numpy as np 

def joint_bilateral_filter(src, joint, sigma_space, sigma_color, border_type=cv2.BORDER_DEFAULT): 
    window_shape = np.array((2 * np.ceil(3 * sigma_space) + 1,) * 2).astype(np.int64) 