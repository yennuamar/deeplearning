
import sys
sys.path=['', '/home/ubuntu/anaconda3/envs/tensorflow_p36/bin', '/home/ubuntu/src/cntk/bindings/python', '/home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python36.zip', '/home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6', '/home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/lib-dynload', '/home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages', '/home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/IPython/extensions', '/home/ubuntu/.ipython']

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Conv3D, MaxPooling3D, UpSampling3D
from keras.layers.merge import concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data_3d import *

def dice_coef(y_true, y_pred, smooth=0.00001):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def weighted_dice_coefficient(y_true, y_pred, axis=(-2, -1, 0), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return keras.mean(2. * (keras.sum(y_true * y_pred, axis=axis) + smooth/2)/(keras.sum(y_true, axis=axis) + keras.sum(y_pred,axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return 1-weighted_dice_coefficient(y_true, y_pred)

class myUnet(object):

    def __init__(self, slices = 32,  img_rows = 192, img_cols = 192):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.slices = slices

    def load_data(self):
        mydata = dataProcess(self.slices, self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        imgs_train2 = imgs_train[:,3:35,:,:,:]
        imgs_mask_train2 = imgs_mask_train[:,3:35,:,:,:]
        imgs_test2 = imgs_test[:,3:35,:,:,:]

        return imgs_train2, imgs_mask_train2, imgs_test2



    def get_unet(self):

        inputs = Input((self.slices, self.img_rows, self.img_cols,1))
        print ("inputs shape:",inputs.shape)
        conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        print ("conv1 shape:",conv1.shape)
        conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        print ("conv1 shape:",conv1.shape)
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
        print ("pool1 shape:",pool1.shape)

        conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        print ("conv2 shape:",conv2.shape)
        conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        print ("conv2 shape:",conv2.shape)
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
        print ("pool2 shape:",pool2.shape)

        conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        print ("conv3 shape:",conv3.shape)
        conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        print ("conv3 shape:",conv3.shape)
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        print ("pool3 shape:",pool3.shape)

        conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        print ("conv4 shape:",conv4.shape)
        conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        print ("conv4 shape:",conv4.shape)
        drop4 = Dropout(0.1)(conv4)
        print ("drop4 shape:",drop4.shape)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)
        print ("pool4 shape:",pool4.shape)

        conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        print ("conv5 shape:",conv5.shape)
        conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        print ("conv5 shape:",conv5.shape)
        drop5 = Dropout(0.1)(conv5)
        print ("drop5 shape:",drop5.shape)

        up6 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(drop5))
#        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 4)
        print ("up6 shape:",up6.shape)
        merge6 = concatenate([drop4,up6], axis = 4)
        print ("merge6 shape:",merge6.shape)
        conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        print ("conv6 shape:",conv6.shape)
        conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        print ("conv6 shape:",conv6.shape)

        up7 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv6))
#        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 4)
        print ("up7 shape:",up7.shape)
        merge7 = concatenate([conv3,up7], axis = 4)
        print ("merge7 shape:",merge7.shape)
        conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        print ("conv7 shape:",conv7.shape)
        conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        print ("conv7 shape:",conv7.shape)

        up8 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv7))
#        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 4)
        print ("up8 shape:",up8.shape)
        merge8 = concatenate([conv2,up8], axis = 4)
        print ("merge8 shape:",merge8.shape)
        conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        print ("conv8 shape:",conv8.shape)
        conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        print ("conv8 shape:",conv8.shape)

        up9 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,2))(conv8))
#        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 4)
        print ("up9 shape:",up9.shape)
        merge9 = concatenate([conv1,up9], axis = 4)
        print ("merge9 shape:",merge9.shape)
        conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        print ("conv9 shape:",conv9.shape)
        conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        print ("conv9 shape:",conv9.shape)
        conv9 = Conv3D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        print ("conv9 shape:",conv9.shape)
        conv10 = Conv3D(1, 1, activation = 'sigmoid')(conv9)
        print ("conv10 shape:",conv10.shape)
        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics = [dice_coef])

        return model

    def train(self):

        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
#        model.load_weights('/home/ubuntu/unet-master-DWI/weights2.h5')
        model_checkpoint = ModelCheckpoint('/home/ubuntu/unet-master-DWI/weights2.h5', monitor='loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

        print('predict test data')
        model.load_weights('/home/ubuntu/unet-master-DWI/weights2.h5')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('/home/ubuntu/unet-master-DWI/results2/imgs_mask_test.npy', imgs_mask_test)



    def save_img(self):

        print("array to image")
        imgs = np.load('/home/ubuntu/unet-master-DWI/results2/imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("/home/ubuntu/unet-master-DWI/results2/"+str(i+80)+".jpg")




if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
#    myunet.save_img()








