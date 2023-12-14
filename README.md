# NEURAL NETWORKS (DEEP LEARNING)

## Introduction - Basic Concepts

Neural networks are becoming increasingly popular and are responsible for some of the most cutting-edge advancements in data science including image and speech recognition. They have also been transformative in reducing the need for intensive and often time-intensive feature engineering for traditional supervised learning tasks. In this lesson, we'll investigate the architecture of neural networks.

![image](https://github.com/MarvinAgumba/NEURAL-NETWORKS-WALKTHROUGH/assets/122484885/02b73ee7-4ce2-44c3-8a8a-bbbf4edd3b33)

Types of neural networks: 
- Standard neural networks
- Convolutional neural networks (input = images, video)
- Recurrent neural networks (input = audio files, text, time series data)
- Generative adversarial networks

The **loss function** is used to measure the inconsistency between the predicted value $(\hat y)$ and the actual label $y$.

**Keras**, a package that has prebuilt many of the building blocks of neural networks
- Scalars = 0D tensors
- Vectors = 1D tensors
- Matrices = 2D tensors
- 3D tensors

A ***tensor*** is defined by three key attributes: rank or number of axes; the shape; the data type

These are the three main operations that you will see in future implementations. ***Element-wise addition*** (or other operations) simply updates each element with the corresponding element from another tensor; ***Broadcasting*** operations can be used when tensors are of different dimensions; 

### Steps Taken To Build a Neural Network with Keras
- Importing the packages
- Decide on the network architecture `model = models.Sequential()`
- Add layers `model.add(layers.Dense(units, activation, input_shape))`
- Compile The Model `model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='mse',metrics=['accuracy'])`
- Train The Model `history = model.fit(x_train,y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))`
- Plot `history.history['loss']`
- Make Predictions `y_hat = model.predict(x)`
- Evaluate the Model `model.evaluate(X_test, X_test_labels)`

### Activation Functions

The **`sigmoid function`** is nearly always the go-to in the output layer of a binary classification problem but in hidden layers (outputs activation values somewhere between 0 and 1.)

![image](https://github.com/MarvinAgumba/NEURAL-NETWORKS-WALKTHROUGH/assets/122484885/e688006b-2ff7-42e4-932c-c38353c4a181)

The **`hyperbolic tangent (or tanh) function`** goes between -1 and +1, and is a shifted version of the sigmoid function. The means of the activations coming out are closer to zero! A disadvantage of both tanh and sigmoid activation functions is that when $z$ gets quite large or small, the derivative of the slopes of these functions becomes very small, generally 0.0001. This will slow down gradient descent. You can see in the tanh plot that this already starts happening for values of $z > 2$ or $z < 2$.

![image](https://github.com/MarvinAgumba/NEURAL-NETWORKS-WALKTHROUGH/assets/122484885/bbace24d-c15c-4ab3-88e3-1b1eb17734e2)

The **`inverse tangent (arctan) function`** has a lot of the same qualities that tanh has, but the range roughly goes from -1.6 to 1.6, and  the slope is more gentle than the one we saw using the tanh function.

![image](https://github.com/MarvinAgumba/NEURAL-NETWORKS-WALKTHROUGH/assets/122484885/0b1a2b6c-bb53-4529-86a4-123861ebe6ae)

**`The RectifiedLinear Unit Function (ReLU)`**: This is probably the most popular activation function, along with the tanh! The fact that the activation is exactly 0 when $z <0$  is slightly cumbersome when taking derivatives though. 

![image](https://github.com/MarvinAgumba/NEURAL-NETWORKS-WALKTHROUGH/assets/122484885/bcea8e76-9513-4057-8229-02fa4870b91f)

The **`leaky ReLU`** solves the derivative issue by allowing for the activation to be slightly negative when $z <0$.

![image](https://github.com/MarvinAgumba/NEURAL-NETWORKS-WALKTHROUGH/assets/122484885/5cc59a68-9a4d-484c-b5be-2bbe5b186169)

## CNN - Convolutional Neural Networks
CNNs are a useful model for image recognition due to their ability to recognize visual patterns at varying scales\
CNNs are great for the following tasks:
- Image classification
- Object detection in images
- Picture neural style transfer

- Padding can be used to prevent shrinkage and make sure pixels at the edge of an image receive the necessary attention
- Max pooling is typically used between convolutional layers to reduce the dimensionality
- After developing the convolutional and pooling layers to form a base, the end of the network architecture still connects back to a densely connected network to perform classification
  
**Building CNNs with Kerras**

![image](https://github.com/MarvinAgumba/NEURAL-NETWORKS-WALKTHROUGH/assets/122484885/4a5b4aa5-4f49-4d94-ba00-1403bcb898b6)

### ADDITIONAL RESOURCES

- https://towardsdatascience.com/multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2-bf464f09eb7f

- https://playground.tensorflow.org/

- [Visualising activation functions in neural networks](https://dashee87.github.io/data%20science/deep%20learning/visualising-activation-functions-in-neural-networks/)

- https://keras.io/getting-started/
 
- https://keras.io/getting-started/sequential-model-guide/#compilation

- https://www.coursera.org/learn/deep-neural-network/lecture/BhJlm/rmsprop

- https://www.coursera.org/learn/deep-neural-network/lecture/qcogH/mini-batch-gradient-descent

- A full book on Keras by the author of Keras himself: https://www.manning.com/books/deep-learning-with-python
