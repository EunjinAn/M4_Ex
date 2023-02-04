# M4: Applied deep learning and artificial intelligence
## Exercise 1 
**Trained and Tested simple neural network model using pytorch**
- We used HR attrition Dataset, intially we selcted the important feature with attrition as our target. 
- Trained a simple model with 4 inputs and 1 output, the results were good but we explored more compleax model. 
- Trained model with 4 inputs and 1 hidden layer. For this model we tested different number of epochs and learning rate. 
- We tested the model with best hyperparameters. 

## our result
From training result, we select the model with following conditions.
n_layer = 3, learning_rate = 0.01, hidden_Activation function = Tanh, Output_Activation function = Sigmoid, Optimizer = Adam.
![image](https://user-images.githubusercontent.com/112074208/216768592-af142bbd-29be-4330-b338-62f9be7e352d.png)



test		N_layer	epoch	learning_rate	Hidden_Activation	Output _Activation	optimize	Min_error
test_01(N_layer)	default	3	100	0,01	ReLU	Sigmoid	RMSprop	0,1267
	case_01	4	100	0,01	ReLU	Sigmoid	RMSprop	0,1274
								
test_02(learning_rate)	default	3	100	0,01	ReLU	Sigmoid	RMSprop	0,1267
	case_02	3	100	0,05	ReLU	Sigmoid	RMSprop	0,1364
								
test_03(hidden_F)	default	3	100	0,01	ReLU	Sigmoid	RMSprop	0,1267
	case_03	3	100		Tanh	Sigmoid	RMSprop	0,1214
								
test_04(Output_F)	default	3	100	0,01	ReLU	Sigmoid	RMSprop	0,1267
	case_04	3	100		ReLU	Softmax	RMSprop	0,84
								
test_05()	default	3	100	0,01	ReLU	Sigmoid	RMSprop	0,1267
	case_05	3	100	0,01	ReLU	Sigmoid	Adam	0,1265
![image](https://user-images.githubusercontent.com/112074208/216768568-b6651453-08a4-4ca9-920a-e07c69670fe9.png)
