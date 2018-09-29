# DIP-Project
DIP PROPOSAL

## Project Id -> 20
## Title -> Given pictures of large Indian meal spreads, identify items (label and segments

## TEAM NAME: 2 Fast 2 Fourier

## TEAM MEMBERS:
Himansh Sheoran
Lovish Narang
Ayush Anand

## Github Link : https://github.com/sheoranhimansh/DIP-Project 

## Main Goals:
Need to successfully segment out regions in an image consisting of Indian Foods using some sophisticated image segmentation techniques.  Then train our model to so that it can recognise the label of any given image of an Indian meal spread.
Steps to be followed->
Create a Database of different types of Indian Meal spreads. Each type will be treated as a separate class containing large amount of instances in it.
Implement an Object Detector/Segmenter which given an image, can easily segment out a bounding box containing the food
Train a multi-class classifier  (eg. svm, linear etc) labelling the bounding box
Test on new images


## Problem Definition:
The four main problem that will arise in this project is to :

->  find an algorithm which can successfully segment the required part which we require, out of the image so that our training can be made more efficient. 

-> Training our model to fit the given data. There are many multi-class classifier available which again has its own advantages and disadvantages. 

->Many of the good classifiers for ex SVM are basically binary class classifier. So we also need to present a model in which we can implement a binary class classifier to work for multi class classifiers also.

-> Prevent overfitting.

## Results Of the Project:

Solution to above Problems

-> Image segmentation is the techniques are used to partition an image into meaningful parts have similar features and properties.The aim of segmentation is simplification i.e. representing an image into meaningful and easily analyzable way. Image segmentation is the first step in image analysis.

The main techniques  to Image Segmentation are:
Structural Segmentation Techniques
Stochastic Segmentation Techniques
Hybrid Techniques

Different Methods which follow the given above techniques are
Thresholding Method
Edge Based Segmentation Method
Region Based Segmentation Method
Clustering Based Segmentation Method
Watershed Based Methods
Partial Differential Equation Based Segmentation Method



Each method has its own pros and cons. Thus we need to figure out which method will work out best for us.

-> Out of various machine learning classifiers SVM gives out pretty good results. Therefore we’ll probably use it.

-> SVM being a binary class classifier can only classify among two classes at once.
 
Two of the common methods to enable this adaptation we can include the 1A1 and 1AA techniques:

The 1AA approach represents the earliest and most common SVM multiclass approach  and involves the division of an N class dataset into N two-class cases.

 The 1A1 approach on the other hand involves constructing a machine for each pair of classes resulting in N(N-1)/2 machines. When applied to a test point, each classification gives one vote to the winning class and the point is labeled with the class having most votes. This approach can be further modified to give weighting to the voting process.

## Final Result:
The final result will be a model which when given any image containing an Indian meal spread it will classify into a particular class thus labelling it among others.

## Tasks For Each Member:

>LITERATURE REVIEW - ALL MEMBERS
>Collecting Images - Ayush  Anand
>Generating Dataset(Training,Testing) - Himansh Sheoran
>Labelling Images - Lovish Narang
>Segmentation Implementation - Himansh Sheoran , Ayush Anand
>Multi-class Classifier - Lovish Narang Himansh Sheoran
>Testing - Lovish Narang , Ayush Anand


## Timeline:
1st Week:
>Getting to know the problem thoroughly through literature review and figuring out the ways to solve the problem
2nd Week: 
>Collecting images, Generating Dataset , labelling dataset and start working on Project
3rd Week:   
Selecting a segmentation procedure ,implementing it and figuring
Of Classification techniques
4th Week:
Implement the multi-class classifier , computing accuracy and testing it.
5th Week:
Testing of complete code and improving accuracy 

