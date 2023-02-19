# **M4: Applied deep learning and artificial intelligence**
# Exercise 3
## SBERT : we did three kinds of work with SBERT
### Dataset
- we create data, chatbot made sentences related to denmark.
- Youtube description dataset using from hugging dataset.
- 
### Pre-trained Model
- 'sentence-transformers/all-MiniLM-L6-v2'
- 'AI-Growth-Lab/PatentSBERTa'
- 


# Exercise 2
## CNN
- we used MNIST DATA with 60,000 train data and 10,000 test data form torch utilities.
- we found good model with <u>99.19%</u> accuracy.
- the model has hyperparameters with 32 filters, 50 epoch, 0.001 learning rate and Adam optimizer.
![image](https://user-images.githubusercontent.com/112074208/218149091-771d4930-f8b6-42a7-93a7-f08291386d6a.png)

## LSTM
- We used Political tweet dataset but only used the first 1000 data as using the full dataset the training time was too high.
- We first preprocessed the text using a tweet-preprocessor and we also added some code to replace remaining symbols.
- After splitting the dataset into train and test sets, we created a vocab from the train set text.
- Then using the vocab converted the text tokens into numbers, added padding to make all sequence of same length then a torch dataloader was created with token_ids and labels tensors.
- Then we trained a simple LSTM model with a embedding layer, lstm and a linear layer.
- Then we experimented with some hyperparameters mainly Optimiser and number of hidden layers.


- Results : 

  | Epoches 	| Hidden Layers 	| Learning rate| Optimiser 	| MSE   	|
  |---------	|---------------	|-----------	 |-----------	|-------	|
  | 50      	| 6             	|0.01          | SGD       	| **0.236** 	|
  | 50      	| 6             	|0.01          | AdamW     	| 0.240 	|
  | 50      	| 3             	|0.01          | SGD       	 | **0.239** 	|
  | 50      	| 3             	|0.01          | AdamW     	| 0.278 	|
  

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
From training result, we select the model with the result <font color = 'red'>**min_error = 0.1192**</font>, and the following conditions.
n_layer = 3, learning_rate = 0.01, hidden_Activation function = Tanh, Output_Activation function = Sigmoid, Optimizer = Adam.

![image](https://user-images.githubusercontent.com/112074208/216768787-0886487a-b788-411d-985c-743fced5636b.png)

