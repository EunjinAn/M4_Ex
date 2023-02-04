# **M4: Applied deep learning and artificial intelligence**


# Exercise 1 
## Task 
- Build, train, and evaluate a neural network with Pytorch.
- It should have minimum *2 hidden* layers
- Experiment with at least *5 different variations* of hyperparameters (n layers / neurons, activation function, epochs, optimizers, learning rate etc.).
- Use gradio to build a simple interactive demo (in the notebook).

## Description
- We used HR attrition Dataset, intially we selcted the important feature with attrition as our target. 
- Experimented simple Neural Network model with different variations : 
  - n layer = 3 (include output layer: 2 hidden layer and 1 output layer), 4 (include output layer: 3 hidden layer and 1 output layer)
  - epochs = 50 and 100
  - hidden activation function =ReLU / Sigmoid / Tanh
  - output layer function = Sigmoid / Softmax Activation
  - learning rate = 0.01 / 0.05
- The best results were found with following hyperparameters : 
  - n_layer = 3
  - epochs = 100 (best epochs = 60)
  - hidden layers function = Tanh
  - Output layer function = Sigmoid
  - learning rate = 0.01
  - optimizer = Adam
- We tested the model with best hyperparameters. 

## Our result
From training result, we select the model with the result **min_error = 0.1192**, and the following conditions.
n_layer = 3, learning_rate = 0.01, hidden_Activation function = Tanh, Output_Activation function = Sigmoid, Optimizer = Adam.

![image](https://user-images.githubusercontent.com/112074208/216768787-0886487a-b788-411d-985c-743fced5636b.png)

