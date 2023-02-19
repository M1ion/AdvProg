- **Links:**
  - **Github:** <https://github.com/M1ion/AdvProg.git>
  - **Youtube:** <https://youtu.be/DjGHxV9PZ7c>
  - **Database: [https://www.kaggle.com/datasets/gpiosenka/balls-image-classifi cation](https://www.kaggle.com/datasets/gpiosenka/balls-image-classification)**
- **Introduction**
- **Problem**

In this report, we address the problem of classifying images of different types of balls in a dataset of 30 classes. Sports ball identification and classification is a common task in the sports industry, but it can be challenging due to the variety of shapes, sizes, and colors of different types of balls. Automating this task can improve efficiency and accuracy in sports equipment inventory management, coaching, and training. In this work, we apply deep learning to solve the problem, using the VGG16 model and data augmentation.

- **Literature Review**

There are many existing solutions for image classification, such as Convolutional Neural Networks (CNNs) and transfer learning. CNNs are a type of deep neural network that have been shown to be effective in image classification tasks. Transfer learning is a technique where a pre-trained model is used as a starting point for training a new model on a different task. The pre-trained model can be fine-tuned to the new task by adding new layers on top of it.

One of the most popular pre-trained models for image classification is VGG16, which is a CNN with 16 layers developed by the Visual Geometry Group at Oxford University. It has achieved state-of-the-art performance on various image recognition challenges:

- Theoretical part: [https://towardsdatascience.com/a-demonstration-of-transf](https://towardsdatascience.com/a-demonstration-of-transfer-learning-of-vgg-convolutional-neural-network-pre-trained-model-with-c9f5b8b1ab0a) [er-learning-of-vgg-convolutional-neural-network-pre-traine d-model-with-c9f5b8b1ab0a](https://towardsdatascience.com/a-demonstration-of-transfer-learning-of-vgg-convolutional-neural-network-pre-trained-model-with-c9f5b8b1ab0a)
- Practical part: <https://keras.io/api/applications/vgg/>

Also, Sweta Shaw developed a deep learning-based solution for ball classification on the same dataset, as used in this study. In her work, titled "Ball classification from scratch - 95% acc," she achieved an accuracy of 95% using MobileNetV2 (source: [https://www.kaggle.com/code/swetash/ball-classification-from-sc ratch-95-acc](https://www.kaggle.com/code/swetash/ball-classification-from-scratch-95-acc) ). MobileNetV2 is a neural network architecture that is optimized for mobile devices, with a small memory footprint and fast computation times. The model was trained using a batch size of 20 for 30 epochs, with the categorical cross-entropy loss function and the RMSprop optimizer. Her model was also able to identify the classes with high accuracy, including soccer ball, tennis ball, and volleyball. Overall, her work demonstrated the effectiveness of deep learning models for ball classification and the importance of data preprocessing and model optimization for achieving high accuracy.

- **Current Work**

In this work, I use the VGG16 model as a starting point and add new layers on top of it to perform classification on the dataset. I also use data augmentation to increase the size of the training set and improve the generalization of the model. The model trained using batch size of 16, image size 224x224 for 10 epochs, with the categorical cross-entropy loss function and the Adam optimizer.

- **Data and Methods**
- **Data Analysis**

Dataset obtained from kaggle: [https://www.kaggle.com/datasets/gpiosenka/balls-image-classifi cation](https://www.kaggle.com/datasets/gpiosenka/balls-image-classification). It consists of 30 classes of images with varying sizes and aspect ratios. However, the training set of the dataset is unbalanced, with some classes having much fewer samples than others. As the author of the dataset said: “This was done intentionally so notebook developers could test methods to deal with unbalanced data.”

- **Model Description**

The VGG16 model was used as a starting point and added our own fully connected layers on top of it. The VGG16 model has 16 layers, including 13 convolutional layers and 3 fully connected layers. I froze the pre-trained layers of the VGG16 model during the initial training to prevent overfitting and speed up the training process. In addition, I used the categorical cross-entropy loss and the Adam optimizer with a learning rate of 0.001. Model was trained for 10 epochs with early stopping if the validation loss did not improve for 5 consecutive epochs.

Also data augmentation was used to increase the size of the training set and improve the generalization of the model. I applied random transformations such as rotation, shear, zoom, and horizontal flipping to the images during training.

- **Results**

After training the model for 10 epochs, we achieved accuracy of 96.04%, validation accuracy of 91.33% and a test accuracy of 90%, indicating that the model was able to generalize well to new, unseen

images:

![](Aspose.Words.5a45df3d-d910-4834-b2e7-b3342c827cbe.001.jpeg)

Plot of the training and validation accuracy and loss:

![](Aspose.Words.5a45df3d-d910-4834-b2e7-b3342c827cbe.002.jpeg)Testing the model on a few individual images, and the model was able to correctly classify the images with the confidence score of 1.00:

![](Aspose.Words.5a45df3d-d910-4834-b2e7-b3342c827cbe.003.png)![](Aspose.Words.5a45df3d-d910-4834-b2e7-b3342c827cbe.004.png)

- **Discussion**

Results show that the VGG16 model with added layers and data augmentation is effective in classifying images of objects in this particular dataset.

The VGG16 model performed much better than a simple CNN model:

![](Aspose.Words.5a45df3d-d910-4834-b2e7-b3342c827cbe.005.png)

While the CNN model took over 30 minutes to train and only achieved a test accuracy of 65%:

![](Aspose.Words.5a45df3d-d910-4834-b2e7-b3342c827cbe.006.jpeg)

The VGG16 model took less than 10 minutes to train and achieved a test accuracy of 90%:

![](Aspose.Words.5a45df3d-d910-4834-b2e7-b3342c827cbe.007.jpeg)

I also tried to use other pre-trained models, such as ResNet50, VGG19, instead of VGG16, but with ResNet50 I got the accuracy of 25%, and with VGG19 the accuracy of 88%. Although the performance was on the same level. Therefore, I settled on VGG16.

However, there is still room for improvement, especially for the classes with fewer samples. One possible next step is to use transfer learning with **other** pre-trained models and compare their performance with the VGG16 model. Another possible next step is to collect more data for the classes with fewer samples or use techniques such as data synthesis to balance the dataset.

- **Conclusion**

In this project, VGG16 was used as a pre-trained model for image classification of 30 different sports balls types. Also used data augmentation techniques such as rotation, zooming and flipping to increase the diversity of the dataset. After training the model, I achieved a validation accuracy of 91.33% and a test accuracy of 90%. This model was able to correctly classify new, unseen images with a high degree of accuracy, indicating that the model was able to generalize well to new data. Overall, the use of pre-trained models such as VGG16 can greatly improve the accuracy of image classification tasks, while also reducing the amount of time needed for training.

- **Sources:**
- [https://towardsdatascience.com/a-demonstration-of-transfer-lea rning-of-vgg-convolutional-neural-network-pre-trained-model-wit h-c9f5b8b1ab0a](https://towardsdatascience.com/a-demonstration-of-transfer-learning-of-vgg-convolutional-neural-network-pre-trained-model-with-c9f5b8b1ab0a)
- <https://keras.io/api/applications/vgg/>
- [https://www.kaggle.com/code/swetash/ball-classification-from-s cratch-95-acc](https://www.kaggle.com/code/swetash/ball-classification-from-scratch-95-acc)
