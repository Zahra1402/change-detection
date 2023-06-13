#Import the necessary libraries
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def cloudshadow(image1):
    # _____ creates ndvi image
    B4 = image1[:, :, 3]
    B8 = image1[:, :, 7]
    ndvi = (B8 - B4) / (B8 + B4)
    # _____ creates ndsi image
    swir1   = image1[:, :,11]
    green   = image1[:, :, 2]
    ndsi = (green - swir1) / (green + swir1)

    #Threshold the NDSI and NDVI to identify cloud and shadow pixels
    cloud_mask  = (ndsi < 0.95) & (ndvi <  0.3)
    shadow_mask = (ndsi > 0.92) & (ndvi <  0.1)

    #Combine the cloud and shadow masks
    mask = ~(cloud_mask | shadow_mask)
   
    # Apply the mask to the original image
    cldshw1 = image1 * mask[..., np.newaxis]

    return cldshw1




