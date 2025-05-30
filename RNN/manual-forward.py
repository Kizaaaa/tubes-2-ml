# Load weights from keras training
# RUN keras-reference.py first

import os
# (for disabling warnings)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.api.layers import Embedding, SimpleRNN, Bidirectional, Dropout, Dense, TextVectorization
from keras.api.models import Sequential
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

import numpy as np

data = pd.read_csv("../datasets/NusaX-Sentiment-Indonesian/train.csv")
data_valid = pd.read_csv("../datasets/NusaX-Sentiment-Indonesian/valid.csv")

texts = data['text'].values

# Konfigurasi TextVectorization
tokenizer = TextVectorization(max_tokens=10000, output_mode="int", output_sequence_length=100)
tokenizer.adapt(texts)  # Pelajari vocabulary dari teks

# TextVectorization pada text
tokenized_texts = tokenizer(texts)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['label'].values)
labels_valid = label_encoder.transform(data_valid['label'].values)
texts_valid = data_valid['text'].values

# Tokenizer (TextVectorization)
tokenized_texts_valid = tokenizer(texts_valid)

# ----------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------- #
from myrnn import bidirectional_rnn_forward, embedding_forward, simple_rnn_forward, dense_forward, softmax

# Buat new model dengan arsitektur yang sama yang hanya memuat bobot model sebelumnya
new_model1 = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    SimpleRNN(32,),
    Dropout(0.2), 
    Dense(3, activation="softmax")
])

new_model1.build(input_shape=(None, 100))  # (batch_size, sequence_length (from output of TextVectorization))
new_model1.load_weights("rnn.weights.h5")

# Extract individual layer weights
# print(new_model.layers)
embedding_matrix = new_model1.layers[0].get_weights()[0]  # Shape: (10000, 128)
Wx, Wh, b_rnn = new_model1.layers[1].get_weights()  # Wx: (128, 32), Wh: (32, 32), b_rnn: (32,)
W_dense, b_dense = new_model1.layers[3].get_weights()  # W_dense: (32, 3), b_dense: (3,)

# Forward pass manual untuk seluruh batch validasi
manual_preds = []
for i in range(tokenized_texts_valid.shape[0]):
    emb = embedding_forward(tokenized_texts_valid[i], embedding_matrix)  # (seq_len, emb_dim)
    h = simple_rnn_forward(emb, Wx, Wh, b_rnn)  # (hidden_dim,)
    # Dropout (non-deterministic)
    h_drop = h * (1.0)  # Untuk inference, dropout biasanya tidak dipakai (atau dikalikan 1.0)
    logits = dense_forward(h_drop, W_dense, b_dense)  # (num_classes,)
    probs = softmax(logits)
    manual_preds.append(probs)

manual_preds = np.array(manual_preds)  # (num_samples, num_classes)
manual_pred_classes = manual_preds.argmax(axis=1)

# Evaluasi hasil manual forward
print("Manual Forward Macro F1_score:", f1_score(labels_valid, manual_pred_classes, average='macro'))

# ----------------------------------------------------------------------------------- #
# ----------------------------------- Bidirectional --------------------------------- #
# ----------------------------------------------------------------------------------- #

# Buat new model dengan arsitektur yang sama yang hanya memuat bobot model sebelumnya
new_model2 = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Bidirectional(SimpleRNN(32)),
    Dropout(0.2), 
    Dense(3, activation="softmax")
])

new_model2.build(input_shape=(None, 100))  # (batch_size, sequence_length)
new_model2.load_weights("rnn.bidirectional.weights.h5")

# Extract individual layer weights for bidirectional RNN
embedding_matrix = new_model2.layers[0].get_weights()[0]  # Shape: (10000, 128)

# Get weights from the bidirectional layer
bidirectional_layer = new_model2.layers[1]
forward_rnn = bidirectional_layer.forward_layer
backward_rnn = bidirectional_layer.backward_layer

# Forward RNN weights
Wx_f, Wh_f, b_f = forward_rnn.get_weights()  # Wx: (128, 32), Wh: (32, 32), b: (32,)

# Backward RNN weights
Wx_b, Wh_b, b_b = backward_rnn.get_weights()  # Same shapes as forward

# Dense layer weights
W_dense, b_dense = new_model2.layers[3].get_weights()  # W_dense: (64, 3), b_dense: (3,)

# Forward pass manual untuk seluruh batch validasi
manual_preds = []
for i in range(tokenized_texts_valid.shape[0]):
    emb = embedding_forward(tokenized_texts_valid[i], embedding_matrix)  # (seq_len, emb_dim)
    # Bidirectional RNN layer
    h = bidirectional_rnn_forward(emb, Wx_f, Wh_f, b_f, Wx_b, Wh_b, b_b)  # (hidden_dim * 2,)
    # Dropout (non-deterministic)
    h_drop = h * (1.0)  # For inference, dropout is not used
    # Dense layer
    logits = dense_forward(h_drop, W_dense, b_dense)  # (num_classes,)
    probs = softmax(logits)
    manual_preds.append(probs)

manual_preds = np.array(manual_preds)  # (num_samples, num_classes)
manual_pred_classes = manual_preds.argmax(axis=1)

# Evaluasi hasil manual forward
print("Manual Forward (Bidirectional) Macro F1_score:", f1_score(labels_valid, manual_pred_classes, average='macro'))