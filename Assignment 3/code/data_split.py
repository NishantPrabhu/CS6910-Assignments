
# =================================================================
# Code to split image data into train, validation and test
# =================================================================

# Dependencies
import os
from glob import glob
from shutil import copyfile
from tqdm import tqdm
from ntpath import basename


# Procedure

# Read a folder in images
# Split into 60-20-20
# Create folder in train and send 60 there
# Create folder in test and send 20 there
# Create folder in validation and send 20 there


root_path = '../images'
train_path = '../data/train/'
test_path = '../data/test/'
valid_path = '../data/validation/'


file_bar = tqdm(total=len(os.listdir(root_path)), desc='Progress', position=0)
file_status = tqdm(total=0, position=1, bar_format='{desc}')


for bird_name in os.listdir(root_path):

    file_status.set_description_str(f'Processing {bird_name} ...')

    os.mkdir(train_path + bird_name)
    os.mkdir(test_path + bird_name)
    os.mkdir(valid_path + bird_name)

    image_paths = glob(root_path + '/' + bird_name + '/*.jpg')
    train_files = image_paths[:int(0.7*len(image_paths))]
    valid_files = image_paths[int(0.7*len(image_paths)) : int(0.8*len(image_paths))]
    test_files = image_paths[int(0.8*len(image_paths)):]

    for f in train_files:
        copyfile(f, train_path + bird_name + '/' + basename(f))

    for f in valid_files:
        copyfile(f, valid_path + bird_name + '/' + basename(f))

    for f in test_files:
        copyfile(f, test_path + bird_name + '/' + basename(f))

    file_bar.update(1)
