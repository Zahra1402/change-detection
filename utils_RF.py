
#import libraries
import numpy as np
from joblib import load
import os
from sklearn.preprocessing import StandardScaler


#_____________________________________________________________________________________

def randomforest(image1,image2):
    # Assuming 'image1' and 'image2' are numpy arrays with shape (height, width, band)

    # Slice the desired bands (10 to 12, inclusive) from each image
    img1 = image1[:, :, 10:13]
    img2 = image2[:, :, 10:13]
    # reshape array to vector (row, col,band)--->  (row*col,band)
    image1 =img1.reshape(-1,3)
    image2 =img2.reshape(-1,3)
    #_____________________________________________________________________________________

    # transform 
    scaler = StandardScaler()
    image1=scaler.transform(image1)
    image2=scaler.transform(image2)
    #_____________________________________________________________________________________
    root=os.getcwd()
    path_model=os.path.join(root,'RFclassifier.joblib')

    # load model
    rf_classifier=load(path_model)
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





