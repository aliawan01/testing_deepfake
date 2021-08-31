import cv2
import numpy as np
from image import Video, Training_Utilities
from model import enc, decoderA, decoderB, aeA, aeB
import os
from keras import Input

# For colored error messages
from colorama import init, Fore
from PIL import Image
import matplotlib.pyplot as plt


enc.load_weights('/home/amer/deepfake/new_models/encoder.h5')
decoderA.load_weights('/home/amer/deepfake/new_models/decoder_A.h5')
decoderB.load_weights('/home/amer/deepfake/new_models/decoder_B.h5')

init(autoreset=True)


def convert_one_image( autoencoder, image ):
    crop = slice(48,208)
    face = image[crop,crop]
    face = cv2.resize( face, (64,64) )
    face = np.expand_dims( face, 0 )
    new_face = autoencoder.predict( face / 255.0 )[0]
    new_face = np.clip( new_face * 255, 0, 255 ).astype( image.dtype )
    new_face = cv2.resize( new_face, (160,160) )
    new_image = image.copy()
    new_image[crop,crop] = new_face
    return new_image

original_image = '/home/amer/deepfake/trump\\3.jpg'
read_image = cv2.imread(original_image)
new = convert_one_image(aeB, read_image)
plt.imshow(new)
plt.title("image", fontweight="bold")
plt.show()

'''
original_image = '/home/amer/deepfake/musk\\3.jpg'
read_image = cv2.imread(original_image)
expanded_image = cv2.resize(read_image, (64, 64))
expanded_image = np.expand_dims(expanded_image, axis=0)
img = aeA.predict(expanded_image/255.0)[0]
img = np.clip(img * 255, 0, 255).astype(read_image.dtype)
print(f"\nValue of predicted image: {img.shape}\n")
#new_img = np.squeeze(img)
plt.imshow(img)
plt.title("new_img", fontweight="bold")
plt.show()
#cv2.imshow('image', new_img)
print(f"\nShape of new_img: {new_img.shape}\n")
'''

'''
Code for cropping images:

crop = slice(48, 208)
cropped_face = read_image[crop, crop]
cropped_face = read_image
'''



#img = cv2.imwrite("/home/amer/deepfake/new_thing.jpg", np.array(img))
plt.imsave('/home/amer/deepfake/new_img.jpg', new_img)

#cv2.imwrite("/home/amer/deepfake/new_image.jpg", new_img)
#if img == False:
#    print(Fore.RED + "Could not save image")
#elif img == True:
#    print(Fore.GREEN + "SAVED IMAGE SUCCESSFULLY!!!")
# cv2.imshow('image', img)

# cv2.imshow("image", expanded_image)
# xp_img = np.expand_dims(expanded_image, axis=1)
# print(xp_img.shape)
# # if xp_img.shape == (None, 64, 64, 3):
# predicted_image = aeA.predict(xp_img)
# if predicted_image in globals():
#     cv2.imshow("image", predicted_image)
# # else:
# # else:
#     # print('Did not work')

cv2.waitKey(0) 
cv2.destroyAllWindows() 
