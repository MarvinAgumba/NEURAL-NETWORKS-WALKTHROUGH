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

# ADDITIONAL RESOURCES

- https://towardsdatascience.com/multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2-bf464f09eb7f

- https://playground.tensorflow.org/
