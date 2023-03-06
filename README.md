# Circle-Detector
Object detection Model to detect circle in a noisy image


Initial Description: 

I have formatted my code into a training notebook and separate python files where you can find the required code. 
-	The model and the dataset can be found in the model_dataset.py file
-	The training and testing functions can be found in the model_functions.py file
-	The noisy circle and other functions you have given me can be found in the functions.py file
-	The training notebook can be accessed via Google Colab and includes some plotted examples, evaluation functions and plots for the IoU50 of my final model.

-	I created a PyTorch dataset class that generates noisy circle images along with its centre coordinates and radius. The dataset is given a noise level along with multiple other parameters. The outputted images are then transformed into tensors for PyTorch implementation, and the pixel values are normalised so they all contain relative features.
-	I started off very simply with a noise level of 0.1 but found that the model did not generalise well with noisier circle images. 
-	I also started off with 100 images in the dataset – just so I could have a running model going. After this, I quickly changed the length of the dataset so that I could get more accurate results and so that the model can be trained on a larger dataset.
-	I initially started off with mostly CNNs and 2 linear layers that end with 3 outputs, the x-centre coordinate, the y, and the radius of the circle. I went for this because when the images are inputted into the model, I wanted it to learn to extract information from the potentially noisy input and to then feed this information into linear layers which could approximate the centre location point and the radius of the circle. My Initial model contained around 10 Million parameters and showed promising results on images with 0.1 noise level, but significantly worse results for the noisier levels.
-	I decided to go with the classic MSE_loss for my criterion as it measures the mean-squared error between my model’s predicted values and the ground truth. This is an effective criterion in deep learning and object detection, even though other implementations such as YOLOV5 implement multiple loss functions including integrated IoU50 functions, which can be very effective. 

To optimise the model’s performance:
-	As expressed previously, increasing the length of the dataset from 100 to 1000 showed an increase in model performance. This makes sense as the model learns on a larger dataset. I further increased the size of the dataset to 10_000, which took longer to train but led to much better results. I had to increase my batch size so I could run my training faster and I did some tuning on this number to get the most effective results. 
-	I was initially only running 10 epochs and the model was showing poor convergence. After optimising the learning rate for my adam optimiser to 0.01, I saw much better convergence.
-	I tried multiple other hyper-parameters including adding weight-decay, changing batch size, number of feature maps in my convolutional layers and many more. I also tried adding patience to my model using PyTorch lightning, but as my model converged quite fast, it showed not much increase in model performance. This makes sense as our model converges quite fast, and we do not have a dimensionality problem. 
-	I tried implementing my own IoU50 criterion function but saw poor results and the model was overfitting with a small circle which could fit well into the larger ground truth circle to reach an IoU of 1. Plus it had quite a lot of bugs and I wanted to keep the project quite simple and time-effective.
-	In the end, the most effective model enhancer I saw was adding skip connections from the convolutional layers to the linear ones. Adding these and lowering the size of the amount of feature maps showed faster convergence, better performance and reduced training times. All of which are great for larger model applications for industry. My idea of skip connections came from my previous implementation of YOLOV5, which has a backbone of convolution layers that extract the main features in the inputted images, followed by many linear layers. In the case of YOLOV5, which is used for object detection as well as object classification, there is a need for a large number of linear layers so that the model can correctly classify the objects in the images. In our implementation, there is no need for a large number of linear layers as we will always be classifying our object as a circle, however adding skip connections makes the features which are extracted from different convolution layers come back into the linear layers so that the centre coordinates and radius can be calculated more efficiently, taking inputs from multiple feature maps, some of which could have less ‘noise’ than the other.
-	

To evaluate the model’s performance, I compared sample outputs from my model to the ground truth input visually. This visual evaluation can be seen in my evaluate_with_image function in the notebook and is a quick and effective way of comparing the output of the model. 
For a more in-depth performance metric, once the model was trained, I implemented the IoU calculation code given, to evaluate sample images with varying noise levels and calculated how many images had IoU thresholds of 50%, 75% and 90%.

The final model had around 2.6 million parameters. This is quite a small amount given I initially started with around 10 million. This decrease in parameters did not show a decrease in model performance and I think with further research I could have further decreased the number of tunable parameters even more. For my final model noise level for training, I chose 0.6 as it showed the most balance between less noisy-circle prediction and limiting overfitting on the very noisy images. 
This overfitting is an issue as the model just learns to fit small circles in the middle of the images so that it is the most likely to be included in a large circle of a noisy image – if that makes sense.
Looking at the bar plots in the training notebook, we can see that having the noise level in training set to 0.6 allows for great accuracy in less noisy images while still keeping decent accuracy for the noisy images. We have around 0.98% of the predictions within the IoU over 50% for images of noise levels 0.0 to 0.7 and around 95% over IoU 75%. The percentage decreases when we check for IoU of 90% is still quite high for images with a noise level comparable to 0.6. This is because the model was trained on this noise level so in the end it is the most effective when it is given input with this much noise. 

Limitations and applications to the project: 
-	Could have worse results if we input images with different shapes as the model is only trained on circles.
-	As observed, as the noise level becomes higher than the ones used for training, the model’s performance degrades.
-	The model is only trained to detect centre coordinates and radius. For further applications such as in object recognition and classification is larger and more varied images, a better model would look at trying to estimate bounding boxes for objects and a classification model architecture.
-	Some applications of this sort of algorithm could be used in areas such as medicine if we were to want to determine the location and size of a potential cell/ bacteria in a sample image from a lab. This is just an example and there could be many more examples – which would require a lot more research and fine-tuning of the model.

Overall, this project was challenging but rewarding, and I'm happy with the performance of the final model.
