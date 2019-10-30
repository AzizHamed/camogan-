#!/usr/bin/env python3

'''
Evolving optimum camouflage using Generative Adverserial Networks
Author: Laszlo Talas
Project: https://github.com/****
Dependencies: tensorflow 1.* and keras 2.0.*
Usage: python3 camogan.py
'''

# The basic architecture of the code is inspired by: 
# https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py

import matplotlib
matplotlib.use('TkAgg')
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os
from PIL import Image
from subprocess import call
from keras.models import Sequential
from keras.models import Model, Input
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D, MaxPooling2D
from keras.layers import add, multiply
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras import backend as K
from keras.layers.merge import Concatenate
import tensorflow as tf

from options import *
from utils import make_parallel

class DCGAN(object):
    def __init__(self, img_rows=imsize, img_cols=imsize, channel=3, target_rows=targetsize[0], target_cols=targetsize[1]):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator(self):
        if self.D:
            return self.D
        input_d = Input(shape=(self.img_rows,self.img_cols,self.channel), name = 'input_d')

        d = Conv2D(depth_DM*1, 3, strides = 1, padding = 'same')(input_d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Dropout(drop_DM)(d)
        
        d = MaxPooling2D((2,2), padding= 'same')(d)

        d = Conv2D(depth_DM*2, 3, strides = 1, padding = 'same')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Dropout(drop_DM)(d)

        d = MaxPooling2D((2,2), padding= 'same')(d)

        d = Conv2D(depth_DM*4, 3, strides = 1, padding = 'same')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Dropout(drop_DM)(d)

        d = Conv2D(depth_DM*8, 3, strides = 1, padding = 'same')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Dropout(drop_DM)(d)

        d = Flatten()(d)
        d = Dense(1)(d)
        predictions_d = Activation('sigmoid')(d)

        self.D = Model(inputs = input_d, outputs = predictions_d)

        self.D.summary()
        plot_model(self.D, to_file = 'discriminator.png')
        return self.D

    def generator(self):
        if self.G:
            return self.G
        dim1 = int(targetsize[0])
        dim2 = int(targetsize[1])

        input_g_noise = Input(shape=(rand_input,), name='input_g_noise')

        g = Dense(dim1*dim2*depth_AM)(input_g_noise)
        g = BatchNormalization(momentum=0.9)(g)
        g = Activation('relu')(g)
        g = Dropout(drop_AM)(g)

        g = Dense(dim1*dim2*int(depth_AM/2))(g)
        g = BatchNormalization(momentum=0.9)(g)
        g = Activation('relu')(g)

        g = Reshape((dim1, dim2, int(depth_AM/2)))(g)

        g = Conv2DTranspose(int(depth_AM), 3, padding = 'same')(g)
        g = BatchNormalization(momentum=0.9)(g)
        g = Activation('relu')(g)

        g = Conv2DTranspose(3, 3, padding = 'same')(g)
        
        g_target = Activation('sigmoid')(g)

        # apply mask
        input_g_mask = Input(shape=(self.target_rows,self.target_cols,self.channel), name = 'input_g_mask')
        g = multiply([g_target, input_g_mask])
        g = ZeroPadding2D((int((self.img_rows-self.target_rows)/2),int((self.img_cols-self.target_cols)/2)))(g)

        # input: background images
        input_g_background = Input(shape=(self.img_rows,self.img_cols,self.channel), name = 'input_g_background')

        # merge targets and backgrounds
        predictions_g = add([input_g_background, g])
        
        self.G = Model(inputs = [input_g_noise, input_g_background, input_g_mask], outputs = predictions_g)
        
        self.G.summary()
        plot_model(self.G, to_file = 'generator.png')
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=lr_DM, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        if n_gpu > 1:
            self.DM = make_parallel(self.DM , n_gpu)
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=lr_AM, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        if n_gpu > 1:
            self.AM = make_parallel(self.AM , n_gpu)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        plot_model(self.AM, to_file = 'adversarial.png')
        return self.AM

class CAMO_DCGAN(object):
    def __init__(self):
        self.img_rows = imsize
        self.img_cols = imsize
        self.channel = 3
        self.x_train = samples_empty
        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(0, 1.0, size=[n_plot_samples, rand_input])
        f_d = open(dir_output+'/d_loss.txt', 'w')
        f_a = open(dir_output+'/a_loss.txt', 'w')
        for i in range(train_steps):

            # true (empty scenes)
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]

            # false (images with targets)
            images_fake = samples_empty[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            for j in range(0,images_fake.shape[0]):
                tlX = int(imsize/2-targetsize[0]/2)
                tlY = int(imsize/2-targetsize[1]/2)
                images_fake[j,tlX:tlX+targetsize[0],tlY:tlY+targetsize[1], :] = images_fake[j,tlX:tlX+targetsize[0],tlY:tlY+targetsize[1], :] * mask_inv

            # Calculate D loss
            masks = mask_holder[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(0, 1.0, size=[batch_size, rand_input])
            patterns_fake = self.generator.predict({'input_g_noise': noise, 'input_g_background': images_fake, 'input_g_mask': masks}) # add background images here
            x = np.concatenate((images_train, patterns_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            # Calculate A loss
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(0, 1.0, size=[batch_size, rand_input])
            masks = mask_holder[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            a_loss = self.adversarial.train_on_batch({'input_g_noise': noise,\
                'input_g_background': images_fake,\
                'input_g_mask': masks},\
                y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            f_d.write("%f - %f\n" % (d_loss[0], d_loss[1]))
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            f_a.write("%f - %f\n" % (a_loss[0], a_loss[1]))
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))
        f_d.close()
        f_a.close()

    def plot_images(self, save2file=False, samples=64, noise = None, step=0):
        filename = 'camo1.png'
        noise = np.random.uniform(0, 1.0, size=[samples, rand_input])

        images_fake = samples_empty[np.random.randint(0,
            self.x_train.shape[0], size=samples), :, :, :]
        for j in range(0,images_fake.shape[0]):
            tlX = int(imsize/2-targetsize[0]/2)
            tlY = int(imsize/2-targetsize[1]/2)
            images_fake[j,tlX:tlX+targetsize[0],tlY:tlY+targetsize[1], :] = images_fake[j,tlX:tlX+targetsize[0],tlY:tlY+targetsize[1], :] * mask_inv
        masks = mask_holder[np.random.randint(0,
            self.x_train.shape[0], size=samples), :, :, :]
        images = self.generator.predict({'input_g_noise': noise, 'input_g_background': images_fake, 'input_g_mask': masks})

        if not os.path.exists(dir_output+'/images'):
            os.makedirs(dir_output+'/images')
        if not os.path.exists(dir_output+'/targets'):
            os.makedirs(dir_output+'/targets')
        for i in range(images.shape[0]):
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols, self.channel])
            target_only = image
            target_mask = mask_holder[0,:,:,:]
            target_mask = np.pad(target_mask, ((int((image.shape[0]-target_mask.shape[0])/2),int((image.shape[0]-target_mask.shape[0])/2)),
                (int((image.shape[1]-target_mask.shape[1])/2),int((image.shape[1]-target_mask.shape[1])/2)),(0,0)), 'constant')
            target_only = np.float32(target_only) * np.float32(target_mask)
            target_only = target_only[(int((image.shape[0]-mask_holder.shape[1])/2)):(int((image.shape[0]-mask_holder.shape[1])/2)+mask_holder.shape[1]),
            (int((image.shape[1]-mask_holder.shape[2])/2)):(int((image.shape[1]-mask_holder.shape[2])/2)+mask_holder.shape[2]),:]

            filename_image = save_name_image + "%d_%d.png" % (step,i)
            filename_target = save_name_target + "%d_%d.png" % (step,i)
            if save2file:
                cv2.imwrite(dir_output+'/images/'+filename_image,cv2.cvtColor(image*255, cv2.COLOR_BGR2RGB))
                cv2.imwrite(dir_output+'/targets/'+filename_target,cv2.cvtColor(target_only*255, cv2.COLOR_BGR2RGB))
        
if __name__ == '__main__':

    # get path to images
    im_backgrounds = os.listdir(dir_backgrounds)

    # import mask and images
    mask = cv2.imread(mask_name)
    mask = cv2.resize(mask, (targetsize[1], targetsize[0]), interpolation = cv2.INTER_NEAREST)
    mask = np.float32(mask[:,:,:]/255)
    mask_inv = 1 - mask
    mask_holder = np.zeros((len(im_backgrounds*sample_per_im),mask.shape[0],mask.shape[1],3))
    samples_empty = np.zeros((len(im_backgrounds*sample_per_im),imsize,imsize,3))
    counter = 0
    for ii in range(0,len(im_backgrounds)):
        im = cv2.imread(dir_backgrounds + im_backgrounds[ii])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.float32(im[:,:,:]/255)
        if resize_factor != 1:
            im = cv2.resize(im, (0,0), fx = resize_factor, fy = resize_factor)
        for jj in range(0,sample_per_im):
            sample_x = random.randint(0, im.shape[0]-imsize-1)
            sample_y = random.randint(0, int((im.shape[1]-imsize-1)/2))
            samples_empty[counter,:,:,:] = im[sample_x:sample_x+imsize,sample_y:sample_y+imsize,:]
            mask_holder[counter,:,:,:] = mask
            counter = counter + 1

    # train network and print out images
    camo_dcgan = CAMO_DCGAN()
    camo_dcgan.train(train_steps=ts, batch_size=bs, save_interval=si)
