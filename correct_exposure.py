from model import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import imageio 



#image to enhance
img = cv2.imread('degradedImage.jpg')
#converting colour space for correct functionality and prevent type error being thrown
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = imageio.imread('image534.jpg')
#img = np.expand_dims(img, axis=0)

#normalising image for correct functionaltiy and efficent and high performing predications
img = img.astype('float32')/255.0

#defining input shape 
input_shape = (256, 256, 3)
#building the model
model = build_autoencoder(input_shape=input_shape)
#summary facts such as trainable paramaters
model.summary()
#loading trained weights :) 
model.load_weights("model_weights.h5")
#predicting/enhancing image 
pred = model.predict(np.expand_dims(img, axis= 0))
predimg = pred[0]

#predimg = np.uint8(predimg)
#formatting and saving image
predimg = (predimg*255).astype('uint8')
imageio.imsave(f'EnhancedImage.jpg', predimg)


# img = cv2.imread('image809.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   
# resize_img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
# norm_resize_img = np.float32(resize_img)
# norm_resize_img = cv2.normalize(norm_resize_img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
# norm_resize_img = np.uint8(norm_resize_img * 255)
# save_path = 'imageb.jpg'
#     #cv2.imwrite(save_path, norm_resize_img)
# imageio.imwrite(save_path, norm_resize_img)