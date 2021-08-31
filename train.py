from json import load
import image
from model import aeB, aeA, enc, decoderA, decoderB
import tensorflow as tf
#from tensorflow.keras.Model import load_weights, save_weights
import numpy as np
import platform
import os
import cv2
#import h5py


#Computer Spec Info
if platform.system() == 'Windows':
    os.system('systeminfo')
elif platform.system() == 'Linux':
    os.system('neofetch')
else:
    pass

#Library Versions
print(tf.__version__)
print(np.__version__)
print(cv2.__version__)

''''if not os.path.exists('/home/ali/Dev/deepfake/models/encoder.hdf5'):
    encoder_file = h5py.File('/home/ali/Dev/deepfake/models/encoder.hdf5','a')
    
if not os.path.exists('/home/ali/Dev/deepfake/models/decoder_A.hdf5'):
    decoder_A = h5py.File('/home/ali/Dev/deepfake/models/decoder_A.hdf5','a')
    
if not os.path.exists('/home/ali/Dev/deepfake/models/decoder_B.hdf5'):
    decoder_B = h5py.File('/home/ali/Dev/deepfake/models/decoder_B.hdf5','a')'''

image_man = image.Image_manipulation()
train_util = image.Training_Utilities()
video = image.Video()

#load_weights(encoder_file, decoder_A, decoder_B)
#load_weights()

enc .load_weights('models/encoder.h5')
decoderA.load_weights('models/decoder_A.h5')
decoderB.load_weights('models/decoder_B.h5')

def save_model_weights():
    enc .save_weights('models/encoder.h5')
    decoderA.save_weights('models/decoder_A.h5')
    decoderB.save_weights('models/decoder_B.h5')

if platform.system() == 'Windows':
    file_dir = 'C:\\Devs\\deepfake'
    trump = 'trump'
    musk = 'musk'
    setA_path = video.get_file_path(trump)
    setB_path = video.get_file_path(musk)


elif platform.system() == 'Linux':
    file_dir = '/home/ali'
    trump = 'trump'
    musk = 'musk'
    setA_path = video.get_file_path(trump)
    print(setA_path)
    setB_path = video.get_file_path(musk)
else:
    pass

train_setA = video.loading_images(setA_path)/255.0
train_setB = video.loading_images(setB_path)/255.0


train_setA += train_setB.mean( axis=(0,1,2) ) - train_setA.mean( axis=(0,1,2) )

batch_size = int(len(os.listdir(setA_path))/20)

print( "press 'q' to stop training and save model" )

for epoch in range(1000000):
    batch_size = 64
    warped_A, target_A = train_util.training_data( train_setA, batch_size )
    warped_B, target_B = train_util.training_data( train_setB, batch_size )

    loss_A = aeA.train_on_batch( warped_A, target_A )
    loss_B = aeB.train_on_batch( warped_B, target_B )
    print( loss_A, loss_B )
    print('Current epoch no... ' + str(epoch))

    if epoch % 100 == 0:
        save_model_weights()
        print('Model weights saved')
        test_A = target_A[0:14]
        test_B = target_B[0:14]

    figure_A = np.stack([
        test_A,
        aeA.predict( test_A ),
        aeB.predict( test_A ),
        ], axis=1 )
    figure_B = np.stack([
        test_B,
        aeB.predict( test_B ),
        aeA.predict( test_B ),
        ], axis=1 )

    figure = np.concatenate( [ figure_A, figure_B ], axis=0 )
    figure = figure.reshape( (4,7) + figure.shape[1:] )
    figure = train_util.stack_images( figure )

    figure = np.clip( figure * 255, 0, 255 ).astype('uint8')

    cv2.imshow( "", figure )
    key = cv2.waitKey(1)
    if key == ord('q'):
        save_model_weights()
        exit()


'''for epochs in range(180000):
    warp_A, tar_A = train_util.training_data(train_setA, batch_size)
    warp_B, tar_B = train_util.training_data(train_setB, batch_size)

    print('warped warp_A')
    print('warped warp_B')
    print(aeA.summary())

    #Calculate the loss values
    loss_A = aeA.train_on_batch(warp_A, tar_A)
    print('trained loss_A')
    loss_B = aeB.train_on_batch(warp_B, tar_B)
    print('trained loss_B')

    #How many times the adjusted weights should be saved

    if epochs % 20 == 0:
        save_weights()
        print('saved weights')
        test_A = tar_A[0:14]
        test_B = tar_B[0:14]

    figure_A = np.stack([test_A, aeA.predict(test_A), aeB.predict(test_A)], axis=1)
    print("Done figure_A")
    figure_B = np.stack([test_B, aeB.predict(test_B), aeB.predict(test_B)], axis=1)
    print("Done figure_B")

    fig = np.concatenate([figure_A, figure_B], axis=0)
    print("fig concatenated")
    fig = train_util.stack_images(fig)
    print("images stacked")

    figure = np.clip(fig*255,0,255).astype('uint8')
    cv2.imshow('', fig)'''









