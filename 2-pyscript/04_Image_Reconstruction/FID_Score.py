#!/usr/bin/env python
# coding: utf-8

# Import

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
import glob
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys


import matplotlib.pyplot as plt
import numpy as np

# Define Variables

par = sys.argv[1]
task = sys.argv[2]
test_type = sys.argv[3]


# Define FID Function

def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Load Generated Images

imgs = []
for i in range(54):
    num = i
    img = plt.imread('../data/participants/{par}/04_Image_Reconstruction/{task}/FID/{test_type}/image_{num}.png'.format(par=par,task=task,test_type=test_type,num=num))
    imgs.append(img)

# Load Inception V3 Model

model = InceptionV3(include_top=False, pooling='avg', input_shape=(224, 224, 3))


# Load Real Images

label_imgs = ['R','G','B']
imgs_label = []

composed = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor()
                              ])
imgs_label = []

for idx, val in enumerate(label_imgs):
    img_label = Image.open('../1-acquisition/stimulus/cylinder/{}/{}.png'.format(val,idx+1))
    #print('2-HCI_stimuli/stimulus/cylinder/{}/{}'.format(val,idx))
    #print(img_label.shape)
    img_label = composed(img_label.convert('RGB'))
    img_label = np.transpose(img_label,(1,2,0)).numpy()
    imgs_label.append(img_label)   


# Load Generated Image Labels
    
index = 3
# ### Concate generate images with same class 
# - 0 Red
# - 1 Green
# - 2 Blue 

label = np.load('../data/participants/{par}/04_Image_Reconstruction/{task}/FID/{test_type}/generated_labels.npy'.format(par=par,task=task,test_type=test_type))


# Data Preparation

generate_red = []
generate_green = []
generate_blue = []

for idx, label_val in enumerate(label):
    
    if label_val == 0:
        generate_red.append(imgs[idx])
    elif label_val == 1:
        generate_green.append(imgs[idx])
    elif label_val == 2:
        generate_blue.append(imgs[idx])
   #print(idx,label_val)


generate_red = np.stack(generate_red, axis=0)
generate_green = np.stack(generate_green, axis=0)
generate_blue = np.stack(generate_blue, axis=0)


# Concate label with same class 

label_red =  imgs_label[0]
label_red = np.repeat(label_red[np.newaxis,:,:,:], generate_red.shape[0], axis=0)
label_green =  imgs_label[1]
label_green = np.repeat(label_green[np.newaxis,:,:,:], generate_green.shape[0], axis=0)
label_blue =  imgs_label[2]
label_blue= np.repeat(label_blue[np.newaxis,:,:,:],  generate_blue.shape[0], axis=0)


real_imgs = [label_red,label_green,label_blue]
gen_imgs = [generate_red,generate_green,generate_blue]
color = ['red','green','blue']

# Calculate FID and Write Results to .txt and .csv

fids = []
for i in range(len(color)):
# pre-process images
    images1 = preprocess_input(real_imgs[i])
    images2 = preprocess_input(gen_imgs[i])
    fid  = calculate_fid(model,images1,images2)
    fids.append(fid)

with open("../results/FID_{task}_{test_type}.txt".format(task=task,test_type=test_type), "a") as myfile:
    myfile.write(f'================= {sys.argv[1]} ================\n')
    for i in range(len(color)):
        myfile.write('FID SCORE({}): %.3f \n'.format(color[i]) % fids[i])

with open("../results/FID_{task}_{test_type}.csv".format(task=task,test_type=test_type), "a") as myfile:
    for i in range(len(color)):
        myfile.write(f'{fids[i]},')
    myfile.write("\n")



