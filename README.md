# Landscapes-Prediction

I have developed a classification model designed to predict and categorize landscapes based on image data. The model is capable of distinguishing between six distinct categories of landscapes. The process involved training the model on a diverse dataset of extracted features of images, ensuring that it can accurately identify and classify each type. The six categories encompass a broad range of natural and possibly man-made environments, allowing the model to be highly versatile and applicable to various real-world scenarios.

# Approach

## Pre-Processing

This step majorly required to enhance sharpness of images, in case if any of them are blurred. Additionally, it involves resizing all images to ensure they have consistent dimensions.

## Feature Selection

To train the model, we need to extract features on behalf of which they can be learned to be differentiated. Using those extracted features, a trained model can uniquely categorize them.

In this project, we used **Haralick**, **HuMoments**, and **Colour** as features of images.

Haralick Texture is used to quantify an image based on texture. The fundamental concept involved in computing Haralick Texture features is the Gray Level Co-occurrence Matrix or GLCM. The basic idea is that it looks for pairs of adjacent pixel values that occur in an image and keeps recording it over the entire image. From the GLCM matrices, 14 textural features are computed that are based on statistical calculation. These 14 features can be used as an extracted feature of image.

Image Moment is a particular weighted average of image pixel intensities, with the help of which we can find some specific properties of an image, like radius, area, centroid etc. Hu moments are a set of 7 values calculated using central moments that are invariant to image transformations such as translation, scale, and rotation. These moments are useful for shape recognition tasks. Once we have extracted the Hu moments for all images, they can be used as features for training the model.

Colour can also be used as a training feature of model since they can be useful in categorizing images uniquely. 

All three features mentioned above are used in this project for training the classifier.

## Training model

For this specific case, we used Random Forest Classifier model since it was best suited. Its ability to handle high-dimensional data, provide feature importance, and generalize well makes it a strong candidate for many machine learning problems.


## Installation and Flask App setup

Make sure to have Python v3.9 or higher (v3.9.12 recommended)

After cloning the repository, create a new environment by executing the command inside the cloned directory:
```
python -m venv env
```
Now activate the environment and install the required packages by running the command:
```
env\Scripts\activate
pip install -r requirement.txt
```
Finally, run the command to deploy the flask app over localhost:5000
```
python app.py
```

## Other Information

You can find all the guided steps in [Model_Research](Model_Research.ipynb) file where all steps are properly documented.

Also, if you want to train the model, you can run [model_build](model_build.py) file. Make sure to uncomment last line to save the model in pkl file format.

## Further Enhancement

- The performance of the model can be further improved if imbalance of dataset is reduced. Also if size of total dataset can be increased, we can see further increase in accuracy of model.
- Also, K-cross validation and Data Augmentation can also be implemented to improve accuracy of the model.
