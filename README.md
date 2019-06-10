# ECE285-MLIP-Artistic-Style-Transfer
2019 spring ECE285 project B Style Transfer, Gatys and CycleGAN

Group: GAN Styles.

Members: Renjie Zhao, Jared Pham, Joseph Walker


Description
===========
This repository is for the class project. The project contains two part, one is the Gatys' style transfer model and the other is the CycleGAN based style transfer model.


Requirements
============
The cyclegan demo needs to be run in the same directory as the models and images folder from below:
https://drive.google.com/open?id=12tmeCocKZws0O1npbKvuONOcu9ohKA0q

These two images are used for neuralstyle_preprocess and neuralstyle that are not from hosted dataset:
https://drive.google.com/open?id=1RMxAh4g-IR51xWz9Dngg79l1LlUtUJeg

If you would like to run the training code as is you need our landscape dataset at 
https://drive.google.com/open?id=1f-JyOhuPhWYHfM96IDTKJgkbq5QzWRbZ

Code organization
=================
cyclegandemo.ipynb -- Run a demo of our code 

training_base_model.ipynb -- Run the training of the Monet base model for CycleGANs

test_models.ipynb -- Notebook used to view results on random images on the models. 

neuralstyle.ipynb - Neural Style Transfer that can be run on DSMLP

neuralstyle_preprocess.ipynb - Neural Style transfer that was run on google colab. Normalized input and equations written out.

