# Airbus_Ship_Detection
CS2470 Final Project

## This branch has a simple Unet Model. 
In testing, dummy data is used and the model works as expected. 
But we don't know its performance as the data is not real data. 

## Todo: It needs to integrate with the preprocess(), as the dummy data may not conform to the actual data format. 
Here I assume the image format is (num_img, 256, 256, 3) and label/mask format is (num_label, 256, 256, 1). 
(256, 256, 3) is not the original image size. 
If we use the original image size (768, 768, 3), the model will not work as the concat layer and upsampling layer will not match exactly when concatenate. This may be why other people also choose to divide the original images into (256,256,3) size smaller images. 
The loss function and the accuracy function works, but may need to be improved in the future. 

