# Airbus_Ship_Detection
CS2470 Final Project

## This branch has a simple Unet Model. 
In testing, dummy data is used and the model works as expected. 

## However, it needs to integrate with the process(), as the dummy data may not conform to the actual data format. 
Here I assume the image format is (num_img, 256, 256, 3) and label/mask format is (num_label, 256, 256, 1).
If we use the original image size, the model will not work as the concat layer and upsampling layer will not match exactly when concatenate. 
The loss function and the accuracy function works, but may need to be improved in the future. 
