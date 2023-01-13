# scoliosis_detector
As part of our research project, we implemented a program to diagnose scoliosis.

First we detect the vertebrae in an X-ray image of the spine, 
then we detect the four corners of each vertebra, all using deep neural networks, the last step is to calculate the Cobb angle using the detected corners 
to calculate the slope of each vertebra.
In addition, we have implemented a means of correcting the results
obtained in order to improve the performance and accuracy of the detection and
therefore of the Cobb angle.

![Demonstration](https://i.imgur.com/oelmsaw.gif)

All the outputs (images and landmark positions) are generated and stored inside the "data" directory.

## Requirements
#### Python (3.7.9)
#### Detectron2 (0.4) 
#### TensorFlow (2.4.1) 
#### OpenCV (4.5.2.52) 
#### NumPy (1.20.2)
#### Albumentations (0.5.2)
