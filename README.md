# Next Word Prediction using FFNN, RNN, and LSTM

## Overview

This project aims to implement and compare different language models (FFNN, RNN, LSTM) for next-word prediction. The models will be trained on two datasets:

- **Pride and Prejudice**
- **Ulysses**

Each model will be evaluated based on perplexity and generalization performance.

## Project Structure

```
📂 Project Root
│── generator.ipynb  
│── FFNN_pp.ipynb    # Trains FFNN model on Pride and Prejudice (n=3,5)
│── FFNN_u.ipynb     # Trains FFNN model on Ulysses (n=3,5)
│── lstmpp.ipynb    # Trains LSTM model on Pride and Prejudice
│── lstm_u.ipynb     # Trains LSTM model on Ulysses
│── rnnpp.ipynb     # Trains RNN model on Pride and Prejudice
│── rnn_u.ipynb      # Trains RNN model on Ulysses
```

---

## **Model Implementations**

### **Feedforward Neural Network (FFNN)**

- Implemented for **n-grams (n=3,5)**
- Architecture:
  - Input layer: Tokenized context words converted to embeddings
  - Hidden layers: Fully connected layers with ReLU activations
  - Output layer: Softmax over vocabulary to predict the next word
- Training:
  - Optimizer: Adam
  - Loss: Cross-entropy loss

---

### **Recurrent Neural Network (RNN)**

- Implemented using a standard **Elman RNN**
- Uses **GloVe embeddings** for input representation
- Architecture:
  - Input layer: Tokenized context words converted to embeddings
  - Recurrent layer: Simple RNN with tanh activation
  - Fully connected layer with softmax activation
- Training:
  - Optimizer: Adam
  - Loss: Cross-entropy loss

---

### **Long Short-Term Memory (LSTM)**

- Implemented with **GloVe embeddings**
- Uses LSTM cells to capture long-range dependencies
- Architecture:
  - Input layer: Tokenized context words converted to embeddings
  - LSTM layer: Multi-layer LSTM with dropout
  - Fully connected layer with softmax activation
- Training:
  - Optimizer: Adam
  - Loss: Cross-entropy loss

---

## **How to Run**

### **1. Load the Models in Google Colab**

1. Open **Generator.ipynb** in Google Colab.
2. Mount Google Drive to access saved models:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Load the required model:
   ```python
   model_path= "https://drive.google.com/drive/folders/1908ZxuAKv_F4bdLTP8qdeiAaO9WHeN42?usp=sharing"
   <!-- Link to google drive saved models -->
   model = torch.load(model_path, map_location=torch.device('cpu'))
   ```
