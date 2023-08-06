
#import libraries
import numpy as np
from joblib import load
import os
from sklearn.preprocessing import StandardScaler


#_____________________________________________________________________________________

def randomforest(img1,img2):
    # Assuming 'img1' and 'img2' are numpy arrays with shape (height, width, band)


    # reshape array to vector (row, col,band)--->  (row*col,band)
    image1 =img1.reshape(-1,13)
    image2 =img2.reshape(-1,13)
    #_____________________________________________________________________________________

    # transform 
    scaler = StandardScaler()
    scaler.fit(image1)
    image1=scaler.transform(image1)
    image2=scaler.transform(image2)
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

    return change_mask





