
#import libraries
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import load , dump
import os
import rasterio as rio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
#_____________________________________________________________________________________

root=os.getcwd()
path_image1=os.path.join(root,'data','SVM','images','After_R56C31_13_F.tif')#after img
#path_file_image2=os.path.join(root,'data','SVM','images','image1.tif')#before img
path_roi   =os.path.join(root,'data','SVM','labels','label.tif')
# read image and labels
X=rio.open(path_image1)
Xr=X.read()
y=rio.open(path_roi)
yr=y.read()
# transpose image and label (band,col,row)---> (row, col,band)
Xt=Xr.T
yt=yr.T
# reshape array to vector (row, col,band)--->  (row*col,band)
X1=Xt.reshape(-1,13)
y1=yt.reshape(-1,1)
# remove Nodata in labels
X=X1[np.where(y1>0)[0]]
y=y1[y1>0]
# remove water class
X=X[np.where(y!=5)[0]]
y=y[y!=5]


# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#transform 
scaler = StandardScaler()
# fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
prd=scaler.transform(X1)
X_test = scaler.transform(X_test)
#_____________________________________________________________________________

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#Train the classifier on the training data
rf_classifier.fit(X_train, y_train)
#_____________________________________________________________________________

path_model=os.path.join(root,'RFclassifier.joblib')
# save model
#joblib.dump(rf_classifier,path_model)

# load model

rf_classifier=joblib.load(path_model)

#_____________________________________________________________________________
# predict test data for validation
predictions = rf_classifier.predict(X_test)

# print classification report
print(classification_report(y_test, predictions))

#_____________________________________________________________________________
# predict original image
predicted_labels=rf_classifier.predict(prd)
# reshape of predicted_labels to dimension of original image
predicted_labels=predicted_labels.reshape(Xt.shape[0],Xt.shape[1])
predicted_labels=predicted_labels.T

#_____________________________________________________________________________



output_raster_path = os.path.join(root,'predicted_labels.tif')

# Read the source raster to get necessary information
with rio.open(path_image1) as src:
    profile = src.profile
    data = src.read(1)  # Assuming there is only one band

# Update the profile to write the predicted labels
profile['dtype'] = 'uint8'  # Change the data type if needed
profile['count'] = 1  # Set the number of bands for the predicted labels (1 in this case)

# Write the predicted labels to the GeoTIFF file
with rio.open(output_raster_path, 'w', **profile) as dst:
    dst.write(predicted_labels, 1)

print("Predicted labels saved as a GeoTIFF file.")

