# M4: Applied deep learning and artificial intelligence
## Exercise 1 
**Trained and Tested simple neural network model using pytorch**
- We used HR attrition Dataset, intially we selcted the important feature with attrition as our target. 
- Trained a simple model with 4 inputs and 1 output, the results were good but we explored more compleax model. 
- Trained model with 4 inputs and 1 hidden layer. For this model we tested different number of epochs and learning rate. 
- We tested the model with best hyperparameters. 

## Our result
From training result, we select the model with the result **min_error = 0.1192**, and the following conditions.
n_layer = 3, learning_rate = 0.01, hidden_Activation function = Tanh, Output_Activation function = Sigmoid, Optimizer = Adam.

![image](https://user-images.githubusercontent.com/112074208/216768787-0886487a-b788-411d-985c-743fced5636b.png)

