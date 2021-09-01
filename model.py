from json import encoder
from keras.losses import binary_crossentropy
from keras.layers import Dense, Flatten, Reshape, Conv2D, Input, LeakyReLU
from keras.models import model_from_yaml
#from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from pixel_shuffler import PixelShuffler
from keras import Model
from tensorflow.keras.optimizers import Adam

import h5py
import os
import pandas as pd

#Constants
LOSS = binary_crossentropy
OPTIMIZER = Adam(learning_rate=0.0005, beta_1=0.5, beta_2=0.999)
IMG_DIM = (64, 64, 3)
x = Input(shape=IMG_DIM)
DECODER_INPUT = (8, 8, 512)
ENCODER_DIM = (4, 4, 1024)
DEPTH = 32
K_SIZED = 3
K_SIZEE = 5


class Nets(PixelShuffler):
    def __init__(self):
        super().__init__()

    def encoder(self):
        input_img = Input(shape=(IMG_DIM))
        x = Conv2D(DEPTH * 4, kernel_size=K_SIZEE, strides=2, padding="same")(input_img)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(DEPTH * 8, kernel_size=K_SIZEE, strides=2, padding="same")(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(DEPTH * 16, kernel_size=K_SIZEE, strides=2, padding="same")(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(DEPTH * 32, kernel_size=K_SIZEE, strides=2, padding="same")(x)
        x = LeakyReLU(0.1)(x)
        x = Dense(1024)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape(ENCODER_DIM)(x)
        x = Conv2D(DEPTH * 16, kernel_size=3, padding="same")(x)
        x = LeakyReLU()(x)
        x = PixelShuffler()(x)
        return Model(input_img, x)


    def decoder(self):
        input_val = Input(shape=(8, 8, 128))
        x = Conv2D(DEPTH * 8, kernel_size=K_SIZED, padding="same")(input_val)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        x = Conv2D(DEPTH * 4, kernel_size=K_SIZED, padding="same")(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        x = Conv2D(DEPTH * 2, kernel_size=K_SIZED, padding="same")(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        x = Conv2D(3, kernel_size=5, padding="same", activation='sigmoid')(x)
        return Model(input_val, x)

    def discriminator(self):
        input_val = Input(shape=(64, 64, 6))
        x = Conv2D(DEPTH * 2, kernel_size=K_SIZEE, strides=2)(input_val)
        x = LeakyReLU()(x)
        x = Conv2D(DEPTH * 4, kernel_size=K_SIZEE, strides=2)(x)
        x = LeakyReLU()(x)
        x = Conv2D(DEPTH * 8, kernel_size=K_SIZEE, strides=2)(x)
        x = LeakyReLU()(x)
        x = Conv2D(1, kernel_size=5, )(x)
        return Model(x)


n = Nets()
enc = n.encoder()
decoderA = n.decoder()
decoderB = n.decoder()

'''
enc_yaml = enc.to_yaml()
with open("enc.yaml", "w") as yaml_file:
    yaml_file.write(enc_yaml)

enc.save_weights("/root/deepfake/models/encoder.h5")
print("Saved model to disk")

da_yaml = decoderA.to_yaml()
with open("decoderA.yaml", "w") as yaml_file:
    yaml_file.write(da_yaml)

decoderA.save_weights("/root/deepfake/models/decoder_A.h5")
print("Saved model to disk")

db_yaml = decoderB.to_yaml()
with open("decoderB.yaml", "w") as yaml_file:
    yaml_file.write(db_yaml)

decoderB.save_weights("/root/deepfake/models/decoder_B.h5")
print("Saved model to disk")
'''

discriminator_A = n.discriminator()
discriminator_B = n.discriminator()

aeA = Model(x, decoderA(enc(x)))
aeB = Model(x, decoderB(enc(x)))

aeA.compile(optimizer=OPTIMIZER, loss=LOSS)
#print(type(ae_A), type(ae_B))
aeB.compile(optimizer=OPTIMIZER, loss=LOSS)


def load_weights():
    enc .load_weights('/root/deepfake/models/encoder.h5')
    decoderA.load_weights('/root/deepfake/models/decoder_A.h5')
    decoderB.load_weights('/root/deepfake/models/decoder_B.h5')

def save_weights():
    enc .save_weights('/root/deepfake/models/encoder.hdf5')
    decoderA.save_weights('/root/deepfake/models/decoder_A.h5')
    decoderB.save_weights('/root/deepfake/models/decoder_B.h5')

