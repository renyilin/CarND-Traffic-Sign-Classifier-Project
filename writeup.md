# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writerup_img/visualization.png "Visualization"
[image2]: ./writerup_img/example_signs.png "Example_signs"
[image3]: ./writerup_img/grayscale.png "Grayscaling"
[image4]: ./writerup_img/my_found_signs.png "my Traffic Sign"
[image5]: ./writerup_img/new_img_prediction.png "New img prediction"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

This is my writeup and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The data set used in this project is German Traffic Sign Dataset. It can be downloaded from this [link](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

I used the Numpy to calculate summary statistics of the traffic signs data set:

| Data set                          | Size        |
| :-------------------------------: | :---------: |
| The size of training set          | 34799       |
| The size of the validation set    | 4410        |
| The size of test set              | 12630       |
| The shape of a traffic sign image | (32, 32, 3) |
| The number of unique classes      | 43          |

#### 2. The exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the count of each sign in training data set.

![visualization][image1]

The random selected 10 traffic signs images are illustrated as below.

![Example_signs][image2]


### Design and Test a Model Architecture

#### 1. The pipeline of the image data pre-processing.

* Convert the images to grayscale by averaging the RGB.
 This was done by using (R+G+B)/3. This worked well probably because the shape of traffic sign, rather than the color, is the major determinant to recognize a traffic sign. This method can help to reduce training time.

* Normalize the data to the range (-1,1)
This was done by using (X_train - 128)/128. The normalized dataset mean reduced to roughly -0.35. We always want our data have zero mean and equal variance whenever possible, in order to make it easier for optimizer to do its job.

Here is an example of traffic sign images after pre-processing.

![grayscale][image3]


#### 2. The final model architecture.

My final model basically adpots LeNet-5, which is consisted of the following layers:

| Layer           | Description                              |
| :-------------: | :--------------------------------------: |
| Input           | 32x32x1 Grayscale image                  |
| Convolution 5x5 | 1x1 stride, no padding, outputs 28x28x6  |
| ReLu            |                                          |
| Max pooling     | 2x2 stride, no padding, outputs 14x14x6  |
| Convolution 5x5 | 1x1 stride, no padding, outputs 10x10x16 |
| ReLu            |                                          |
| Max pooling     | 2x2 stride, no padding, outputs 5x5x16   |
| Fully connected | Dropout, outputs 120                     |
| Fully connected | Dropout, outputs 84                      |
| Fully connected | outputs 43                               |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the Adam optimizer. The final hyperparameters used were as following:
* learning rate: 0.0009
* batch size: 128
* epochs: 60
* mu: 0
* sigma: 0.1
* dropout keep probability: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated.

At first, I chose the LeNet-5 architecture without dropout. Then I tried to add dropout in fully connected layers to reduce overfitting.

My final model results were:
* training set accuracy: 0.999
* validation set accuracy: 0.956
* test set accuracy: 0.945

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![my Traffic Sign][image4]

The traffic signs I found on internet are in different sizes. The quality of each is similar with the training data set. 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image         | Prediction    |
| :-----------: | :-----------: |
| Keep left     | Keep left     |
| Yield         | Yield         |
| Priority Road | Priority Road |
| Stop          | Stop          |
| 30 km/h       | 30 km/h       |

The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the 1st, 2nd, 3rd, 5th new images, the model is quite sure about the prediction. All of these model's softmax probabilities are above 0.92.

For the 4th new images, although the prediction is right, the model is not quite sure. There are two major predictions with similar softmax probabilities(Stop: 0.53, No entry: 0.45). 

The top 5 softmax probabilities for each image are summarized as below:  

![New Images Prediction][image5]
