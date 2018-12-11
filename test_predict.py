
import sys
sys.path=['', '/home/ubuntu/anaconda3/envs/tensorflow_p36/bin', '/home/ubuntu/src/cntk/bindings/python', '/home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python36.zip', '/home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6', '/home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/lib-dynload', '/home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages', '/home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/IPython/extensions', '/home/ubuntu/.ipython']


from unet import *
from data import *
import numpy as np

import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

mydata = dataProcess(512,512)

imgs_test = mydata.load_test_data()

myunet = myUnet()

model = myunet.get_unet()

model.load_weights('unet.hdf5')

imgs_mask_test = model.predict(imgs_test, verbose=1)

np.save('imgs_mask_test.npy', imgs_mask_test)