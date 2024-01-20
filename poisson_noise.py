# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:12:04 2024

@author: nadja
"""
import numpy as np
    


photon_count=3000 # 
attenuation_factor=2.76 # corresponds to absorption of 50%



def apply_noise(img, photon_count):
    opt = dict(dtype=np.float32)
    img = np.exp(-img, **opt)
    # Add poisson noise and retain scale by dividing by photon_count
    img = np.random.poisson(img * photon_count)
    img[img == 0] = 1
    img = img / photon_count
    # Redo log transform and scale img to range [0, img_max] +- some noise.
    img = -np.log(img, **opt)
    return img