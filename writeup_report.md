# **Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project** 

### *My solution to the Udacity Self-Driving Car Engineer Nanodegree Behavioral Cloning project.*

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image8]: ./examples/model.png "Final Model"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Initially I tried LeNet, but it was not preforming well and then I decided to go with the [NVIDIA model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) as suggested by Udacity. 

The model architecture is as follows:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
activation_1 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_2 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________
```

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with dropout rate 0.25 in order to reduce overfitting (model.py lines 142).

In addition to that, I split my sample data into training and validation data. Using 80% as training and 20% as validation.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 42). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 160).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the data provided by Udacity a combination of center lane driving, recovering from the left and right sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use LeNet model and the training data provided by Udacity. The model didn't perform well driving all aroud the track.

Then I tried the Nvidia model with few modification by adding a new layer at the end to output single value steering angle.

For pre-processing a new Lambda layer was introduced to normalize the input images to zero means. This step allows the car to move a bit further, but it didn't get to the first turn. Another Cropping layer was introduced, by removing the top 50 (trees and sky) and bottom 20 (car). This improved the driving a little bit but not perfect.

Then I Augmented the data by adding the same image flipped with a negative angle(lines 83 - 89).

To combat the overfitting, I modified the model by adding dropout layer in the fully connected layer with dropout rate of 0.25, so that the model generalizes on a track that it has not seen.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 111-154) consisted of the following layers.

Here is a visualization of the architecture

![alt text][image8]

#### 3. Creation of the Training Set & Training Process

* I used the dataset privided by udacity and also I created my own dataset on my local machine by driving a lap around the first track. I tried to maintain the car at the center of the track and also created recovery data by driving to the center from the left and right sides.

* I split the dataset into training and validation set using sklearn preprocessing library with 80% as Training Set and 20% as Validation Set.

* Correction factor of 0.2 is done for the steering angle for the left and right images as suggested by udacity. or the left images I increase the steering angle by 0.2 and for the right images I decrease the steering angle by 0.2

* I used generator to generate the data to avoid loading all the images in the memory and instead generate it at the run time in batches of 32. The image augmentation and correction factor are introduced inside the generator function.

* To augment the dataset, I also flipped images horizontally and adjust the steering angle accordingly using cv2 flip function.

* Model Parameters
    - No. of Epochs = 5
    - Optimizer Used = Adam
    - Validation Data split = 0.20
    - Generator batch size = 32
    - Correction factor = 0.2
    - Loss Function Used- MSE(Mean Squared Error as it is efficient for regression problem).

Finally after training using the above method, the model is saved to model.hf file and used to drive the car simulator.It perfomed really well in maintaining the car on the track.
