# Airbus_Ship_Detection

Changed the U-Net model:
(1. have changed UpSampling to Conv2DTranspose)
(2. rename sample_train to data, to fit with the format on gcp)
(3. tried dice_loss instead of binary-cross-entropy loss, but still reaching accuracy of 1 too fast, and then iou stay at 0: this post (https://www.kaggle.com/iafoss/unet34-dice-0-87/notebook) mentions that we might need a mixed loss function. binary-cross-entropy loss may not work well for images where the mask is sparse.)
(4. tried to get the img encodings where the images have at least one ship in it, rational: want to see if we delete all empty images, can we get a better performance.)


TODO
- Train on the dataset
- experiment with different loss
- visualize the final output to see why the accuracy fastly approaches 1
- Model: 
  - U-Net 
  - 2U-Net
- Data preprocess / augment
