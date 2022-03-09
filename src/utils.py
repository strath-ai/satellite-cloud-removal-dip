import rasterio as rio
import numpy as np

def rgb(image):
    if image.shape[-1] > 12:
        return image[...,(3,2,1)]
    elif image.shape[-1] == 3:        
        return image
    # S1 -> VV, VH
    elif image.shape[-1] == 2:
        return np.concatenate((np.zeros((*image.shape[:-1],1)), image), axis = -1)
    
def s2_preprocess(path, val_range = [0, 1e4], scale = 2e3, channels = [3,2,1]):
    
    s2_file = rio.open(path)
    all_channels = s2_file.read()
    s2 = all_channels[channels,:,:]
    
    s2clip = s2.clip(*val_range)
    s2clip /= scale
    
    return s2clip.transpose(1,2,0)

def s1_preprocess(path, max_val = 2.0, to_db = False):
    s1_file= rio.open(path)
    s1vv, s1vh, s1m = s1_file.read() # s1m is the mean
    
    s1_stack = np.stack([s1vv, s1vh], axis = -1)
    
    # 1. Apply dB scale
    if to_db:
        s1dB = 10*np.log10(s1_stack)
    else:
        s1dB = s1_stack

    # 2. Clip
    # "The clipping range for the Sentinel-1 VV and VH is [−25,0] and [−32.5,0]"
    # 3. Contain within [0, max_val]
    s1dB[...,0] = max_val*(s1dB[...,0].clip(-25, 0) + 25) / 25
    s1dB[...,1] = max_val*(s1dB[...,1].clip(-32.5, 0) + 32.5) / 32.5  

    return s1dB