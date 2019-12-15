# Ship Detection in Satellite Images Using U-Net-Based Architectures

## Progress
- finished data decoding, pre-processing 
- finished model building 
- finished model training and test
- finished visualization
-------------------------------------
- have experimented with separating each image into smaller sub-images
- have experimented with U-Net model
- have experimented with different loss functions
- have experimented with 2U-Net model
- have experimented with U-Net model with CNN model
- have experimented with different hyper-parameters

## Repository Structure
This master branch contains the baseline U-Net model and the CNN classifier. For the 2U-Net model, check the branch 2U-Net.

## How to run
As the requirement for focal loss, please make sure installing `tensorflow_addons` using the following command:

```bash
pip install tensorflow-addons
```

You can use `assignment.py` to train or test a U-Net variant with optional argument.

1. Train

```bash
python assignment.py --mode train --encoding-file [PATH TO MASK_ENCODING_FILE] --img-dir [PATH TO IMAGE DIR] --out-dir [PATH TO VISUAL RESULT] --batch-size 4 --num-epochs 1 --learn-rate 1e-4
```

2. Test

```bash
python assignment.py --mode test --encoding-file [PATH TO MASK_ENCODING FILE] --img-dir [PATH TO IMAGE DIR] --out-dir [PATH TO VISUAL RESULT] --batch-size 4
```
*Note: there is a small dataset to test the functionalities. To use it, just leave* `--encoding-file` *and* `--img-dir` *unchanged*.
