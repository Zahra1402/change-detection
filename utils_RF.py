
#import libraries
import numpy as np
from joblib import load
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot  as plt

#_____________________________________________________________________________________

def randomforest(img1,img2):
    # Assuming 'img1' and 'img2' are numpy arrays with shape (height, width, band)
    imgvis1=img1[:,:,10:13].astype(np.uint8)
    imgvis2=img2[:,:,10:13].astype(np.uint8)
    img1=img1.astype(np.float64)
    img2=img2.astype(np.float64)

    # reshape array to vector (row, col,band)--->  (row*col,band)
    image1 =img1.reshape(-1,13)
    image2 =img2.reshape(-1,13)
    #_____________________________________________________________________________________

    # transform 
    scaler = StandardScaler()
    
 
    image1=scaler.fit_transform(image1)
    image2=scaler.fit_transform(image2)
    #_____________________________________________________________________________________

    # load model
    rf_classifier=load('/home/zahra/Desktop/codes/change_detection/RFclassifier.joblib')
    #_____________________________________________________________________________________
    #predict labels1 , labels2
    prediction1=rf_classifier.predict(image1)
    prediction2=rf_classifier.predict(image2)

    # Replace classes 1 with 0 in the predictions
    prediction1[prediction1 == 1] = 0
    prediction2[prediction2 == 1] = 0

    # Reshape predictions back to original image size
    prediction1 = prediction1.reshape(img1.shape[0], img1.shape[1])
    prediction2 = prediction2.reshape(img2.shape[0], img2.shape[1])

    # Create the change mask
    change_mask = np.where(
        ((prediction1 == 0) & (prediction2 == 3)) |
        ((prediction1 == 0) & (prediction2 == 4)) |
        ((prediction1 == 2) & (prediction2 == 3)) |
        ((prediction1 == 2) & (prediction2 == 4)) |
        ((prediction1 == 4) & (prediction2 == 3)), 1, 0)
     #____________________________________________________________________________________
    plt.figure(figsize=(10, 10))
    ax=plt.subplot(221)
    plt.imshow(prediction1   , cmap='jet')

    plt.title('classified image1')
    plt.subplot(223, sharex=ax, sharey=ax)

    plt.imshow(prediction2   , cmap='jet')

    plt.title('classified image2')

    plt.subplot(222, sharex=ax, sharey=ax)

    plt.imshow(imgvis1   , cmap='jet')

    plt.title(' image1')
    plt.subplot(224, sharex=ax, sharey=ax)

    plt.imshow(imgvis2   , cmap='jet')

    plt.title(' image2')

            # Add spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()
    return change_mask





