# ImageColorization-Deep-CNN
# Introduction:

In image colorization, our goal is to produce a colored image given a greyscale input image. The problem in this task is that a single grayscale image may correspond to many colored images. As a result, traditional models often relied on significant user input alongside a grayscale image. 
Recently, deep neural networks have shown remarkable success in automatic image colorization going from grayscale to color with no additional human input. This success may in part be due to their ability to capture and use semantic information in colorization.

This project implements a deep convolutional neural network for automatic colorization, the problem of converting grayscale input images into colored images. The model is based on the ResNet-18 classifier and trained on the MIT Places database of landscapes and scenes.

# The Problem:
We aim to infer a full-coloured image, which has 3 values per pixel (lightness, saturation, and hue), from a grayscale image, which has only 1 value per pixel (lightness only). For simplicity, we will only work with images of size 256×256, so our inputs are of size 256×256×1 (the lightness channel) and our outputs are of size 256×256×2 (the other two channels). 
Rather than work with images in the RGB format, as people usually do, we will work with them in the LAB colour space (Lightness, A, and B). This colour space contains exactly the same information as RGB, but it will make it easier for us to separate out the lightness channel from the other two (which we call A and B). We'll make a helper function to do this conversion.
We'll try to predict the colour values of the input image directly (we call this regression).

# Papers used:
Colorful Image Colorization - https://arxiv.org/abs/1603.08511
Let there be color! - https://www.semanticscholar.org/paper/Let-there-be-color!-Iizuka-Simo-Serra/ec3453b0892ed8bf0531ffb5370c0159597eec11
Also, I found this paper very useful to take its work into consideration: Combining Deep Convolutional Neural Networks with Markov Random Fields for Image Colorization - https://drive.google.com/file/d/12GfrYOoHUIyUd7gQlLRoe-DdkjhvMOId/view?usp=sharing

# Dataset:
We are going to use subset of MIT Places dataset (containing places, landscapes, and buildings). You can download this dataset from this link: http://data.csail.mit.edu/places/places205/testSetPlaces205_resize.tar.gz 

Dataset is split into 90% training, and 10% test (validation) ... 40k images for training and 1k for validation.

# Tools:
We'll build and train our model with PyTorch. We'll also use torchvision, a helpful set of tools for working with images and videos in PyTorch, and scikit-learn for converting between RGB and LAB colorspces.

# Model:
More info about our model can be found in the PDF documentation. 
Here is a figure showing the architecture of the model:
![model](https://user-images.githubusercontent.com/33100615/167316005-35e05795-fca3-497d-ba87-7f62d3be8d3a.jpg)

# Training:
We trained our model on Google Co-lab session for 100 epochs.

# Results:
![image](https://user-images.githubusercontent.com/33100615/167316119-e89343a2-f2f8-4c80-bf4b-06bc0d815f57.png)



