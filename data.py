import cv2
import glob
import numpy as np
import imageio



# # Get all jpg image filenames from the specified directory
image_filenames = glob.glob('E:\FiveKLegacyUpdate\*.jpg')

counter = 0
# Loop to preprocess images for ground truth
for filenname in image_filenames:
# Read the image and convert from BGR to RGB
    img = cv2.imread(filenname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize_img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
# Resize the image
    norm_resize_img = np.float32(resize_img)
# Normalize the resized image to float
    norm_resize_img = cv2.normalize(norm_resize_img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    norm_resize_img = np.uint8(norm_resize_img * 255)
# Adjust the brightness of the image
    brightness_factor = 0.5
    adjusted_img = cv2.convertScaleAbs(norm_resize_img, alpha=brightness_factor, beta=0)
    #adjusted_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB)
# Save the adjusted image
    save_path = 'data/ground_truth/class1/image{}.jpg'.format(counter)
    imageio.imwrite(save_path, adjusted_img)
    #cv2.imwrite(save_path, adjusted_img)
    counter +=1

# Loop to preprocess images for label
for filenname in image_filenames:
        # Read the image and convert from BGR to RGB
    img = cv2.imread(filenname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    resize_img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)

    norm_resize_img = np.float32(resize_img)

    norm_resize_img = cv2.normalize(norm_resize_img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    norm_resize_img = np.uint8(norm_resize_img * 255)

    save_path = 'data/label/class1/image{}.jpg'.format(counter)
    #cv2.imwrite(save_path, norm_resize_img)
    imageio.imwrite(save_path, norm_resize_img)
    counter +=1
