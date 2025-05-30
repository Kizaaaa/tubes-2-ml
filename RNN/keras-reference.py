import os
# (for disabling warnings)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras.api.layers import Embedding, SimpleRNN, Bidirectional, Dropout, Dense, TextVectorization
from keras.api.models import Sequential
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../datasets/NusaX-Sentiment-Indonesian/train.csv")
data_valid = pd.read_csv("../datasets/NusaX-Sentiment-Indonesian/valid.csv")

texts = data['text'].values
labels = data['label'].values

# Konfigurasi TextVectorization
tokenizer = TextVectorization(max_tokens=10000, output_mode="int", output_sequence_length=100)
tokenizer.adapt(texts)  # Pelajari vocabulary dari teks

# TextVectorization pada text
tokenized_texts = tokenizer(texts)

model1 = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    SimpleRNN(32), 
    Dropout(0.2), 
    Dense(3, activation="softmax")
])

model1.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Konversi label menjadi integer (0, 1, 2)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['label'].values)
labels_valid = label_encoder.transform(data_valid['label'].values)

texts_valid = data_valid['text'].values

# Tokenizer (TextVectorization)
tokenized_texts_valid = tokenizer(texts_valid)

model1.fit(
    tokenized_texts, labels,  # Data latih
    validation_data=(tokenized_texts_valid, labels_valid),  # Data validasi
    epochs=10,
	verbose=0
)

predictions = model1.predict(tokenized_texts_valid)

# Mengambil kelas dengan probabilitas tertinggi
predicted_classes = predictions.argmax(axis=1)

val_f1_score = f1_score(labels_valid, predicted_classes, average='macro')
print(f"Macro F1_Score: {val_f1_score:.2f}")

model1.save_weights("rnn.weights.h5")

model1 = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Bidirectional(SimpleRNN(32)), 
    Dropout(0.2), 
    Dense(3, activation="softmax")
])

model1.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Konversi label menjadi integer (0, 1, 2)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['label'].values)
labels_valid = label_encoder.transform(data_valid['label'].values)

texts_valid = data_valid['text'].values

# Tokenizer (TextVectorization)
tokenized_texts_valid = tokenizer(texts_valid)

model1.fit(
    tokenized_texts, labels,  # Data latih
    validation_data=(tokenized_texts_valid, labels_valid),  # Data validasi
    epochs=10,
	verbose=0
)

predictions = model1.predict(tokenized_texts_valid)

# Mengambil kelas dengan probabilitas tertinggi
predicted_classes = predictions.argmax(axis=1)

val_f1_score = f1_score(labels_valid, predicted_classes, average='macro')
print(f"Macro F1_Score (Bidirectional): {val_f1_score:.2f}")

model1.save_weights("rnn.bidirectional.weights.h5")



