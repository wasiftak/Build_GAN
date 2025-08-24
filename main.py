import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import numpy as np

# Set memory growth for GPUs to prevent TensorFlow from allocating all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

ds = tfds.load('fashion_mnist', split='train')

dataiterator = ds.as_numpy_iterator() #setup connections aka iterator
__all__ = ['dataiterator']   #expose iterator for external use

def scale_images(data):
    image = data['image']
    return image / 255

#data pipeline for tf: map-cache-shuffle-batch-repeat-prefetch
ds = tfds.load('fashion_mnist', split='train')
ds = ds.map(scale_images)     #scale images to 0-1
ds = ds.cache()               #cache data after scaling(to avoid re-scaling every epoch)
ds = ds.shuffle(60000)        #shuffle data to randomize order
ds = ds.batch(128)            #batch data to process multiple images at once
ds = ds.repeat()              #repeat data for multiple epochs
ds = ds.prefetch(64)          #prefetch data to avoid idle time

