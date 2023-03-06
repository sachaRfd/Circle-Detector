# Circle-Detector
The Circle-Detector is an object detection model created to detect circles in noisy images. The implementation is done using PyTorch and is separated into a training notebook and separate Python files.

The required code can be found in the following files:

model_dataset.py: Contains the model and the dataset.
model_functions.py: Contains the training and testing functions.
functions.py: Contains the noisy circle and other functions.
Dataset
To generate noisy circle images, a PyTorch dataset class was created. The dataset generates images with their centre coordinates and radius. The images are given a noise level along with other parameters. The outputted images are then transformed into tensors for PyTorch implementation. The pixel values are normalised so that all images contain relative features.

Model
Initially, a model was created with mostly convolutional neural networks (CNNs) and two linear layers that end with three outputs: the x-centre coordinate, the y-centre coordinate, and the radius of the circle. The aim was to extract information from the potentially noisy input and to then feed this information into linear layers which could approximate the centre location point and the radius of the circle. The initial model contained around 10 million parameters and showed promising results on images with a 0.1 noise level but significantly worse results for noisier levels.

To optimise the model's performance, the following steps were taken:

Increasing the length of the dataset from 100 to 10,000 showed an increase in model performance. A larger dataset helps the model to learn better.
The learning rate for the Adam optimizer was optimised to 0.01, and the number of epochs was increased from 10 to a higher number to obtain better convergence.
Multiple other hyperparameters were tuned, including adding weight-decay, changing batch size, and the number of feature maps in the convolutional layers.
Adding skip connections from the convolutional layers to the linear ones was found to be the most effective model enhancer. Adding these connections reduced training times, improved performance and led to faster convergence.
Evaluation
The model was evaluated visually by comparing sample outputs from the model to the ground truth input. For a more in-depth performance metric, IoU (Intersection over Union) calculations were performed on sample images with varying noise levels. The IoU50, IoU75 and IoU90 thresholds were calculated.

The final model had around 2.6 million parameters, which is relatively small compared to the initial 10 million parameters. This decrease in parameters did not show a decrease in model performance. The final model was trained with a noise level of 0.6, as it showed the most balance between less noisy...
