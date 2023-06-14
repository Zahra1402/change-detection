#Import the necessary libraries
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def cloudshadow(image1):
    b1   = image1[:, :,11]
    b2  = image1[:, :, 2]
    b3   = image1[:, :, 5]

    ratiob = (b2 - b1) / (b2 + b1)
    # _____ creates ndsi image
    ndsi = (b3 - b1) / (b3 + b1)
    #Threshold the NDSI and ratiob to identify cloud and shadow pixels
    cloud_mask  = (ndsi>0.95) & (ratiob > 0.93)
    shadow_mask = (ndsi>0.92) & (ratiob >  0.93)

    #Combine the cloud and shadow masks
    mask = ~(cloud_mask | shadow_mask)
       
    # Apply the mask to the original image
    cldshw1 = image1 * mask[..., np.newaxis]

    return cldshw1, ndsi , ratiob
