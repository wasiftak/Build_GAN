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
