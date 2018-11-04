import cv2 
import numpy as np 

# image are normalized to [0, 255] 
# sigma_space: [0, 1] 
# sigma_color: [0, 1] 
class JointBilateralFilter: 
    def __init__(self, sigma_space, sigma_color, padding_border_type=cv2.BORDER_DEFAULT): 
        self.sigma_space = sigma_space 
        self.sigma_color = sigma_color 
        self.padding_border_type = padding_border_type 

    def pad(self, src, radius): 
        return cv2.copyMakeBorder(src, top=radius, bottom=radius, left=radius, right=radius, borderType=self.padding_border_type) 

    def to_patches(self, src, patch_size):
        src_h, src_w, src_channels = np.atleast_3d(src).shape 
        patch_h, patch_w = patch_size 
        return np.lib.stride_tricks.as_strided(
            src, 
            shape=(src_h-patch_h+1, src_w-patch_w+1, patch_h, patch_w, src_channels), 
            strides=src.itemsize*np.array((src_w*src_channels, src_channels, src_w*src_channels, src_channels, 1)), 
            writeable=False
        ) 

    def get_spatial_kernel(self, src_size):  
        sigma_pixels = int(min(src_size) * self.sigma_space)   
        _ = np.exp(-np.arange(-3*sigma_pixels, 3*sigma_pixels+1)**2 / (2 * sigma_pixels**2))
        return np.outer(_, _) 

    def filter(self, src, guide): 
        src_h, src_w, src_channels = np.atleast_3d(src).shape 
        guide_h, guide_w, guide_channels = np.atleast_3d(guide).shape 
        
        src_size = (src_h, src_w)
        spatial_kernel = self.get_spatial_kernel(src_size) 

        d, d = spatial_kernel.shape 
        r = d // 2 
        src_padded = self.pad(src, radius=r)
        guide_padded = self.pad(guide, radius=r)

        src_patches = self.to_patches(src_padded, patch_size=spatial_kernel.shape) 
        guide_patches = self.to_patches(guide_padded, patch_size=spatial_kernel.shape) 

        range_kernels = np.exp(-np.sum((guide_patches-guide_patches[:, :, r:r+1, r:r+1])**2, axis=-1, keepdims=True) / (2 * self.sigma_color**2))

        kernels = np.atleast_3d(spatial_kernel) * range_kernels 

        filtered = np.sum(kernels * src_patches, axis=(-3, -2)) / np.atleast_3d(np.sum(kernels, axis=(-3, -2, -1))) 

        return np.squeeze(filtered) 

