import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

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
labels = data['label'].values

# Konfigurasi TextVectorization
tokenizer = TextVectorization(max_tokens=10000, output_mode="int", output_sequence_length=100)
tokenizer.adapt(texts)  # Pelajari vocabulary dari teks

# TextVectorization pada text
tokenized_texts = tokenizer(texts)

model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    SimpleRNN(32),
    Dropout(0.5), 
    Dense(3, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Konversi label menjadi integer (0, 1, 2)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['label'].values)
labels_valid = label_encoder.transform(data_valid['label'].values)

texts_valid = data_valid['text'].values

# Tokenizer (TextVectorization)
tokenized_texts_valid = tokenizer(texts_valid)

model.fit(
    tokenized_texts, labels,  # Data latih
    validation_data=(tokenized_texts_valid, labels_valid),  # Data validasi
    epochs=3
)

predictions = model.predict(tokenized_texts_valid)
# Mengambil kelas dengan probabilitas tertinggi
predicted_classes = predictions.argmax(axis=1)

val_f1_score = f1_score(labels_valid, predicted_classes, average='macro')
print(f"Validation F1_Score: {val_f1_score:.2f}")

print("Classification Report:")
print(classification_report(labels_valid, predicted_classes, target_names=label_encoder.classes_))

model.save_weights("rnn.weights.h5")

# ----------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------- #

# Buat new model dengan arsitektur yang sama yang hanya memuat bobot model sebelumnya
new_model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    SimpleRNN(32),
    Dropout(0.5), 
    Dense(3, activation="softmax")
])

new_model.build(input_shape=(None, 100))  # (batch_size, sequence_length (from output of TextVectorization))
new_model.load_weights("rnn.weights.h5")

# Extract individual layer weights
# print(new_model.layers)
embedding_matrix = new_model.layers[0].get_weights()[0]  # Shape: (10000, 128)
Wx, Wh, b_rnn = new_model.layers[1].get_weights()  # Wx: (128, 32), Wh: (32, 32), b_rnn: (32,)
W_dense, b_dense = new_model.layers[3].get_weights()  # W_dense: (32, 3), b_dense: (3,)

# Shape debug
# print("Embedding matrix shape:", embedding_matrix.shape)
# print("Wx shape:", Wx.shape)
# print("Wh shape:", Wh.shape)
# print("b_rnn shape:", b_rnn.shape)
# print("W_dense shape:", W_dense.shape)
# print("b_dense shape:", b_dense.shape)

from myrnn import embedding_forward, simple_rnn_forward, dense_forward, softmax

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
print("Manual Forward Validation F1_score:", (manual_pred_classes == labels_valid).mean())
print("Manual Forward Classification Report:")
print(classification_report(labels_valid, manual_pred_classes, target_names=label_encoder.classes_))