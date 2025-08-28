from tensorflow.keras.models import Sequential   #sequential api for the generator and discriminator
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
import numpy as np

def build_generator():
    model = Sequential()
    #takes in random values and shapes it to 7x7x128, beginning of generated image
    model.add(Dense(7*7*128, input_dim=128))  
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())  #upsample to 14x14x128
    model.add(Conv2D(128, 5, padding='same')) 
    model.add(LeakyReLU(0.2))
    
    model.add(UpSampling2D())  #upsample to 28x28x128
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(1, 4, padding='same', activation='sigmoid')) 
    model.add(LeakyReLU(0.2))

    return model

generator = build_generator()
generator.summary()

# img = generator.predict(np.random.randn(4,128,1))  #generate 4 random images
# print(img)

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))   #1: fake, 0: real

    return model

discriminator = build_discriminator()
discriminator.summary()


# value=discriminator.predict(img)  
# print(value)  #values close to 0 are real, values close to 1 are fake

