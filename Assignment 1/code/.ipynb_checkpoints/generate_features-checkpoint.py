import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import torchvision.models as models


# loading the pretrained model of Vgg_16 which uses batch_normalization
net = models.vgg16_bn(pretrained=True)

# set the load_path for all image file
load_path = '.././data/image_data/train/'

# set the save_path for the extracted features file for all the classes
save_path = '.././data/image_data/Feature_extraction_2D/'
os.mkdir(save_path)


# will get the names of the files present in the load path
# The training data
get_class_names = os.listdir(load_path)

# for each class file
for i in get_class_names:
    
    # To save the file with the same name for the extracted features
    new_save_path = save_path + i
    
    # To load the class file
    class_path = load_path + i

    # to load the numpy file
    img = np.load(class_path)

    # To append the extracted features
    arr = []

    # for each image in the class file
    for j in img:   

        # converting the numpy array to tensor
        j = torch.tensor(j)
        
        # reshaping the image to [batch_size,number_of_channel,height,width]
        j = j.view([-1,3,32,32])
        
        # rescaling the image to [1,3,224,224]
        # vgg_net the required input is of size 224*224 and single image so batch size 1 
        j = F.interpolate(j,(224,224))
        
        # Extracting the features from the middle layer of the network
        z = net.features(j)
        
        # Features extracted are of size [1,512,7,7]
        # Taking the average pooling for each channel
        m = F.avg_pool2d(z,(7,7),1,0)
        
        # Now the features are of size [1,512,1,1]
        #reshaping the features to [512] 
        m = m.view([-1]).detach()
        
        # converting it back to numpy array
        m = np.asarray(m)

        # appending to the arr
        arr.append(m)

    arr = np.asarray(arr)
    print(arr.shape)

    # To save the numpy array  
    np.save(new_save_path,arr)