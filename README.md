# **M4: Applied deep learning and artificial intelligence**
# Exercise 4
##  **How to use Inference APIs for connecting to HuggingFace models**
## summarization 
In this project, I challenged the text summary task. The model("google/mt5-small") and dataset("arize-ai/beer_reviews_label_drift_neg") were downloaded and used from Hugging Face, and the Rounge score was used to evaluate the summary task. 
note that, Our assignments can be found [https://huggingface.co/EJaalborg2022/mt5-small-finetuned-beer-ctg-en] of HF.
**(file : M4_Exercise_04_TextSummarization.ipynb)**

## Text emotion analysis
### Dataset
1. [CARER: Contextualized Affect Representations for Emotion Recognition](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)
### Pre-trained Model
1. 'bert-base-uncased'

### Fine Tuning
- We created a pytorch dataset class to feed the data into the model.
- By using BertTokenizer we then tokenized the data to convert them to input_ids, token_type_ids and attention_mask.
- Then using the transformer library's BertForSequenceClassification we fetched the pretrained 'bert-base-uncased' model from hugging face.
- We performed fine tuning on the model to perform our task.

### Gradio & Huggingface
- We have also created a transformer pipeline and gradio demo in the Notebook itself which allows us to use the model to make predictions.
- Then we also created a space in huggingface with a very simple gradio interface allowing to make predictions.

- Model : **[Sadiksha/sentiment_analysis_bert](https://huggingface.co/Sadiksha/sentiment_analysis_bert)**
- Dataset : **[Sadiksha/sentiment_analysis_data](https://huggingface.co/datasets/Sadiksha/sentiment_analysis_data)**
- Space : **[spaces/Sadiksha/Sentiment_analysis_bert](https://huggingface.co/spaces/Sadiksha/Sentiment_analysis_bert)**



# Exercise 3
## SBERT : We did SBERT with three kinds of samples(sentences, youtube transcription, image)
### Dataset
1. we created data, chatbot made sentences related to denmark.
2. Downloaded Youtube description(jamescalam/youtube-transcriptions) dataset from hugging-face dataset.
3. 25k photos dataset from Unsplash  
### Pre-trained Model
1. 'all-MiniLM-L6-v2'
2. 'AI-Growth-Lab/PatentSBERTa'
3. 'clip-ViT-B-32'

### Task Description : Our all SBERT exercises have a gradio interface that allows us to semantic search using text prompt. 

1. We tried simple SBERT using the pre-trained model 'all-MiniLM-L6-v2'. we created some simple sentences and imported about 25 sentences related to Denmark from chat GPT, we did word embedding. The result was very reasonable even though we used very small samples.
**(file : M4_Exercise_03_SBERT.ipynb)**
2. We did second attempt using data(youtube transcriptions from Hugging face) and model'AI-Growth-Lab/PatentSBERTa'(made of Aalborg university), our data size was heavy, so we used 100 train dataset, A trained model found similar sentences through question(query operation).
**(file : M4_Exercise_03_SBERT.ipynb)**
3. CLIP Model encodes text and images to a shared vector space. For our usecase we encode the unsplash images and then use sentence-transformers semantic search to find similar images for a given images or prompt. 
**(file : M4_Exercise_03_image_search.ipynb)**



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

