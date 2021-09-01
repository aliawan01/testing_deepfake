import os
import glob
from image import Video, Training_Utilities
from model import enc, decoderA, decoderB, aeA, aeB
import platform
import numpy as np 
import cv2

enc .load_weights('/root/deepfake/models/encoder.h5')
decoderA.load_weights('/root/deepfake/models/decoder_A.h5')
decoderB.load_weights('/root/deepfake/models/decoder_B.h5')

vid = Video()
training_util = Training_Utilities()

if platform.system() == 'Windows':
    file_dir = 'C:\\Users\\tehni\\Dev\\deepfake'
    trump = 'trump.mp4'
    file = 'faceswap'
    video_path = vid.get_file_path(trump)


elif platform.system() == 'Linux':
    file_dir = '/home/ali/Dev/deepfake'
    trump = 'trump.mp4'
    file = 'faceswap'
    video_path = vid.get_file_path(trump)

else:
    pass

vid.video_to_images('/root/deepfake/musk.mp4', 'faceswap')
# turn into (256, 256, 3)
vid.resize_images('faceswap', 'faceswap_resized')

file_path = vid.get_file_path('faceswap_resized')
print(f"First file path is: {file_path}")

ori = os.getcwd()
tar = os.getcwd() + 'faceswap_resized'
vid.move_file('image', ori, tar, ori)

if int(len(os.listdir('/root/deepfake/faceswap_resized'))) > 0:
    if platform.system() == 'Windows':
        file_direc = '/root/deepfake/finished_vid\\'

    elif platform.system() == 'Linux':
        file_direc = '/home/ali/Dev/deepfake/finished_vid/'

    else:
        pass

vid.make_dir('/root/deepfake/finished_vid')

trump = 'trump.mp4'
unsorted_img_paths = []
img_paths = []
#Image frames path in array 
print(f"File path is: {file_path}")

if platform.system() == 'Windows':
    file_path = file_path

elif platform.system() == 'Linux':
    file_path = file_path + '/'

for fn in glob.glob(file_path + '*.jpg'):
    unsorted_img_paths.append(fn)

print(f"Unsorted image paths are: {unsorted_img_paths}")

# Bubble sort 
numbered_images = []
for images in unsorted_img_paths:
    image = str(images.rstrip('.jpg'))
    numbered_images.append(int(image.lstrip('/root/deepfake/faceswap_resized\\')))

for i in range(len(numbered_images)):
    for y in range(i + 1, len(numbered_images)):
        if numbered_images[i] > numbered_images[y]:
            z = numbered_images[i]
            numbered_images[i] = numbered_images[y]
            numbered_images[y] = z

for x in numbered_images:
    img_paths.append('/root/deepfake/faceswap_resized\\' + str(x) + '.jpg')
    
print(f"Img paths is: {img_paths}")

def convert_one_image(autoencoder, image):
    assert image.shape == (256,256,3)
    crop = slice(48,208)
    face = image[crop,crop]
    face = cv2.resize( face, (64,64) )
    face = np.expand_dims( face, axis=0 )
    new_face = autoencoder.predict( face / 255.0 )[0]
    new_face = np.clip( new_face * 255, 0, 255 ).astype( image.dtype )
    new_face = cv2.resize( new_face, (160,160) )
    new_image = image.copy()
    new_image[crop,crop] = new_face
    return new_image

#Convert the images to elon musk
if trump == 'trump.mp4':
    i = 0 
    for fn in img_paths:
        image = cv2.imread(fn)
        new_image = convert_one_image( aeB, image )
        cv2.imwrite( str(i) + '.jpg', new_image )
        i += 1 

else:
    i = 0 
    for fn in img_paths:
        image = cv2.imread(fn)
        new_image = convert_one_image( aeA, image )
        cv2.imwrite( str(i) + '.jpg', new_image )
        i += 1 

#Checks the length if of the finished vid directory 
if int(len(os.listdir('/root/deepfake/finished_vid'))) > 0:
    vid.convert_frame_to_video('finished_vid')

else:
    ori = os.getcwd()
    vid.move_file('image', ori, '/root/deepfake/finished_vid', ori)
    vid.convert_frame_to_video('finished_vid')



