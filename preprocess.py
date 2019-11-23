import numpy as np
import matplotlib.image as mpimg


def read_encodings(file_name):
    img_to_encodings = {} # jpg name -> run length encoding strings
    with open(file_name) as fd:
        fd.readline()
        for line in fd:
            jpg, encoding = line.split(',')
            if jpg not in img_to_encodings:
                img_to_encodings[jpg] = []
            img_to_encodings[jpg].append(encoding)

    print(len(img_to_encodings))
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

if __name__ == '__main__':
    img_to_encodings = read_encodings('data/train_ship_segmentations_v2.csv')
    img_name = '000194a2d.jpg'
    mask = encodings_to_masks(img_to_encodings[img_name])

    import matplotlib.pyplot as plt
    jpg = jpg_img_to_array('data/'+img_name)
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
