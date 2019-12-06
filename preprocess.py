import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import os.path


def read_encodings(file_name):
    img_to_encodings = {} # jpg name -> run length encoding strings
    with open(file_name) as fd:
        fd.readline()
        for line in fd:
            jpg, encoding = line.strip().split(',')
            if jpg not in img_to_encodings:
                img_to_encodings[jpg] = []
            if encoding:
                img_to_encodings[jpg].append(encoding)

    # print("=====> read the number of encodings = ", len(img_to_encodings))
    return img_to_encodings

# masks will be memory intensive so use small batches
def encodings_to_masks(encodings): # input an array of encoding string
    out=[]
    for encoding in encodings:
        img = np.zeros(768*768, dtype=np.uint8)
        arr = encoding.split()
        arr = [int(i) for i in arr]
        for i in range(0, len(arr)-1, 2):
            img[arr[i]:arr[i]+arr[i+1]] = 1
        out.append(img.reshape((768, 768)).T)
    out = np.array(out) # [batch size, 768, 768]
    return out

def jpg_img_to_array(img_file_name):
    img = mpimg.imread(img_file_name)
    return img # np array of shape [768,768,3]

def get_data(img_dir, img_names, img_to_encodings):
    '''
    Takes in an image directory path, an array of image names, image-to-encodings map,
    and returns (NumPy array of images, NumPy array of labels).
	:param images_dir: directory path for images, something like 'sample_jpgs'
    :param img_names: the list of names of images to be read
    :param img_to_encodings: map image name to run-length-encodings
    :return: a tuple (images, labels) - images is a matrix of shape (9 * batch size, 256, 256, 3),
    labels is a matrix of shape (9*batch size, 256, 256, 1)
    '''

    # print("======>read the number of images = ", len(img_names))

    images = []
    masks  = []
    #step1:  get all images -- images shape is (num_examples, 768, 768, 3)
    for img_name in img_names:
        image = jpg_img_to_array(os.path.join(img_dir,img_name)) ## np array of shape [768,768,3]
        images.append(image)
    images = np.reshape(images, [-1, 768, 768, 3])
    images.astype(np.float32) #then need to cast the data to float32
    images = images/255 #Normalize images

    #step2: read in the encodings of each image, transform it to mask and stores in an array
    for img_name in img_names:
        mask = encodings_to_masks(img_to_encodings[img_name])
        mask = np.sum(mask, axis=0) # sum up masks (each mask is for one ship, to get the mask for the whole image, we need to sum up)
        # mask = np.reshape(mask, [768,768, 1]) #todo: do we need this?
        masks.append(mask)
    masks = np.reshape(masks, (-1, 768, 768, 1))


    return (images, masks)


    



if __name__ == '__main__':
    # img_to_encodings = read_encodings('data/train_ship_segmentations_v2.csv')
    img_to_encodings = read_encodings('sample_train.csv')
    img_name = '000194a2d.jpg'
    mask = encodings_to_masks(img_to_encodings[img_name])

    import matplotlib.pyplot as plt
    # jpg = jpg_img_to_array('data/'+img_name)
    jpg = jpg_img_to_array('sample_jpgs/'+img_name)

    mask = np.sum(mask, axis=0) # sum up masks for each ship
    _, axarr = plt.subplots(1, 3)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')

    axarr[0].imshow(jpg)
    axarr[1].imshow(mask)
    axarr[2].imshow(jpg)
    axarr[2].imshow(mask, alpha=0.4)

    # plt.imshow(jpg)
    # plt.imshow(jpg)
    # plt.imshow(np.sum(mask, axis=0))
    plt.show()
    get_data('sample_jpgs/', 'sample_train.csv', 4)

