For Loading and assigning labels to the provided classes.

1. Copy the code from "load_extracted_features.py" for loading the class files and assigning labels.

    a. Set the path for the saved extracted files.

2. The final output is randomlly shuffled datapoints with corresponding class labels. 




Defined is the method we have used for extracting the 2D features from the image which is provided.
Can use the 2D features provided.  

For extracting 2D features from the image(self).
Instructions for running the program

1. Install pytorch
    
    a. https://pytorch.org/

        Select the appropriate version according to the specification.

2. Run the program "generate_features_from_image.py"
    
    a. Define the load path as the folder containing the images "path/Team_Number".


How are we extracting features.

1. Using VGG_16 model pretrained on ImageNet dataset.

2. The images are of shape 32*32*3 [height,width,channels], we reshape the images to [1,3,224,244]
    
    a. VGG net takes input of shape [224,224]
    b. [1,3,224,224] ==> [batch_size,channels,height,width] is the standard we provide input in pytorch convolutional models.

3. We are extracting the features from 13 layer of the VGG_16 network

    a. Which has output shape of [1,512,7,7]
    b. We apply a global average pooling on the output to get [1,512,1,1] shape output
    c. Reshape the tensor to [512]
    d. For each image in the class the steps a,b,c are performed and each extracted features are appended to get the feature for entire class [300,512] ==> [number of data_points,features_for_each_data_points]
    e. Each feature extracted class_file is stored in the folder "./Feature_extraction_2D".


