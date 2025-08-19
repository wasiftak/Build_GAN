import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from main import dataiterator
fig, ax = plt.subplots(ncols=4, figsize=(20,20)) #ax: create a grid of subplots
for idx in range(4):
    sample = dataiterator.next() 
    ax[idx].imshow(np.squeeze(sample['image'])) #np.squeeze: condense it down to 2D
    ax[idx].set_title(sample['label'])

plt.show() 
