
# Malaria Detection using Transfer Learning

## Abstract

Malaria is a contagious disease that affects millions of lives every year. Traditional diagnosis of malaria in laboratory requires an experienced person and careful inspection to discriminate healthy and infected red blood cells (RBCs). It is also very time-consuming and may produce inaccurate reports due to human errors. Cognitive computing and deep learning algorithms simulate human intelligence to make better human decisions in applications like sentiment analysis, speech recognition, face detection, disease detection, and prediction. Due to the advancement of cognitive computing and machine learning techniques, they are now widely used to detect and predict early disease symptoms in healthcare field. With the early prediction results, healthcare professionals can provide better decisions for patient diagnosis and treatment. Machine learning algorithms also aid the humans to process huge and complex medical datasets and then analyze them into clinical insights.

### Objective

The objective of this Project is to show how deep learning architecture such as convolutional neural network (CNN) which can be useful in real-time malaria detection effectively and accurately from input images and to reduce manual labor with a mobile application.

### Transfer Learning

**Transfer learning** is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. It is the improvement of learning in a new task through the **transfer of knowledge** from a related task that has already been learned. Also, transfer learning helps to deal with the issue of lack of data.

![App Screenshot](https://github.com/sonu275981/Malaria-Detection-using-Transfer-Learning/blob/6ef6b422579284eed7387e2a3070336e074d93ec/demo/1.png?raw=true)

### Task Description:

Create a project using Transfer Learning solving various problems like Face Detection, Image Classification, using existing Deep Learning models like VGG16, VGG19, ResNet, etc.

This project uses **VGG16** model for Detection of Malaria .The data-set contains infected and uninfected cell images. The cell images having some marks/dots in them signifies that it is an infected cell, and the cell image having no such marks means it is an uninfected cell. So, using Image Classification, we can detect whether a particular cell is infected or not, and hence detect malaria.

### VGG16 Architecture

VGG is a deep Convolutional Neural Network used to perform Image Classification. VGG16 is a variant of VGG model. The input of VGG is set to an RGB image of 224x244 size. The average RGB value is calculated for all images on the training set image, and then the image is input as an input to the VGG convolution network. A 3x3 or 1x1 filter is used, and the convolution step is fixed. . There are 3 VGG fully connected layers, which can vary from VGG11 to VGG19 according to the total number of convolutional layers + fully connected layers. The minimum VGG11 has 8 convolutional layers and 3 fully connected layers. The maximum VGG19 has 16 convolutional layers. +3 fully connected layers. In addition, the VGG network is not followed by a pooling layer behind each convolutional layer, or a total of 5 pooling layers distributed under different convolutional layers.

![App Screenshot](https://github.com/sonu275981/Malaria-Detection-using-Transfer-Learning/blob/0fc7e65cee2c4ef0d71fc67969e7717f3df7d7bb/demo/vgg%2016.png?raw=true)

### Steps:

- Here, I am using **jupyter notebook**, you can also use Google colab to do this project.

- Import all the libraries and methods required. And set the train and test path according to the location of data-set in jupyter notebook.

- The parameter include_top in method VGG16 is set to False as we don’t want to include the 3 fully-connected layers at the top of the network.
  Since we don’t want the model to retrain from scratch, so we write layers.trainable=False. This way the model remembers its previous weights.

- **model.summary()** is used to see the architecture/structure of the model. It gives information about all the layers of the network.

- Now, we compile the model using adam optimizer function and loss function=**binary_crossentropy**.

- Then, import the images from data-set using **ImageDataGenerator function**.

- Finally, fit the model using fit_generator method. Here, no. of **epochs=50** and after all the epochs, the accuracy achieved is **0.8363**.

-  You can plot the loss and accuracy functions of training and test set and In my case i have not plot anything. It can be seen that loss function decreases and accuracy increases.

- Now, save this state of the model as a h5 file using **model.save()**.
  Then, to do the prediction using test data, first load this model using load_model function.

  Here, I am loading a single image as input for the prediction.

  ![App Screenshot](https://github.com/sonu275981/Malaria-Detection-using-Transfer-Learning/blob/e61887fd420b5a298315fb6a922538508f3b62fd/demo/load%20model%20via%20h5.png?raw=true)

-  Use model.predict() to do the prediction. Here, I fed an uninfected cell image and the model also predicts the same, so it is a correct prediction.

- Also, you can use imshow() of matplotlib library, the input image for prediction can be seen and analysed manually. Here, as there are no marks/dots appearing in the cell image, it means it is uninfected and the model also predicts the same results. Hence, the model is trained successfully.

  This project is completed!!


