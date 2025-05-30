import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
print(sys.executable)


import keras
from keras.api.layers import Embedding, LSTM, Bidirectional, Dropout, Dense, TextVectorization
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

# Build LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    LSTM(64),  # 64 hidden units
    Dropout(0.5), 
    Dense(3, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['label'].values)
labels_valid = label_encoder.transform(data_valid['label'].values)

texts_valid = data_valid['text'].values
tokenized_texts_valid = tokenizer(texts_valid)

# Train model
print("Training LSTM model...")
model.fit(
    tokenized_texts, labels,
    validation_data=(tokenized_texts_valid, labels_valid),
    epochs=3,
    verbose=1
)

# Get Keras predictions for comparison
predictions = model.predict(tokenized_texts_valid)
predicted_classes = predictions.argmax(axis=1)

keras_f1_score = f1_score(labels_valid, predicted_classes, average='macro')
print(f"Keras LSTM Validation F1_Score: {keras_f1_score:.4f}")

# Save model weights
model.save_weights("lstm.weights.h5")

# ----------------------------------------------------------------------------------- #
# Extract LSTM weights for manual implementation
# ----------------------------------------------------------------------------------- #

# Create new model with same architecture
new_model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    LSTM(64),
    Dropout(0.5), 
    Dense(3, activation="softmax")
])

new_model.build(input_shape=(None, 100))
new_model.load_weights("lstm.weights.h5")

# Extract layer weights
embedding_matrix = new_model.layers[0].get_weights()[0]  # (10000, 128)

# LSTM layer weights - Keras combines all gates into single matrices
lstm_weights = new_model.layers[1].get_weights()
W_combined = lstm_weights[0]  # (input_dim, 4 * hidden_size) - Combined W for all gates
U_combined = lstm_weights[1]  # (hidden_size, 4 * hidden_size) - Combined U for all gates  
b_combined = lstm_weights[2]   # (4 * hidden_size,) - Combined biases

# Dense layer weights
W_dense, b_dense = new_model.layers[3].get_weights()

print(f"LSTM weights shapes:")
print(f"W_combined: {W_combined.shape}")
print(f"U_combined: {U_combined.shape}") 
print(f"b_combined: {b_combined.shape}")

# Split combined weights into individual gates
# Keras order: [input, forget, cell, output] (i, f, c, o)
hidden_size = 64
input_dim = 128

# Split W matrix (input-to-hidden weights)
Wi = W_combined[:, :hidden_size]                    # Input gate
Wf = W_combined[:, hidden_size:2*hidden_size]       # Forget gate  
Wc = W_combined[:, 2*hidden_size:3*hidden_size]     # Cell/Candidate gate
Wo = W_combined[:, 3*hidden_size:]                  # Output gate

# Split U matrix (hidden-to-hidden weights)
Ui = U_combined[:, :hidden_size]                    # Input gate
Uf = U_combined[:, hidden_size:2*hidden_size]       # Forget gate
Uc = U_combined[:, 2*hidden_size:3*hidden_size]     # Cell/Candidate gate  
Uo = U_combined[:, 3*hidden_size:]                  # Output gate

# Split bias vector
bi = b_combined[:hidden_size]                       # Input gate
bf = b_combined[hidden_size:2*hidden_size]          # Forget gate
bc = b_combined[2*hidden_size:3*hidden_size]        # Cell/Candidate gate
bo = b_combined[3*hidden_size:]                     # Output gate

print(f"\nSplit gate weights shapes:")
print(f"Wi: {Wi.shape}, Ui: {Ui.shape}, bi: {bi.shape}")
print(f"Wf: {Wf.shape}, Uf: {Uf.shape}, bf: {bf.shape}")
print(f"Wc: {Wc.shape}, Uc: {Uc.shape}, bc: {bc.shape}")
print(f"Wo: {Wo.shape}, Uo: {Uo.shape}, bo: {bo.shape}")

# ----------------------------------------------------------------------------------- #
# Manual LSTM Forward Pass
# ----------------------------------------------------------------------------------- #

from mylstm import embedding_forward, lstm_forward, dense_forward, softmax

print("\nRunning manual LSTM forward pass...")

# Manual forward pass for validation data
manual_preds = []
for i in range(min(50, tokenized_texts_valid.shape[0])):  # Test on first 50 samples
    # Get embeddings
    emb = embedding_forward(tokenized_texts_valid[i], embedding_matrix)
    
    # LSTM forward pass
    h_final, _, _ = lstm_forward(emb, Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo, Wc, Uc, bc)
    
    # Dense layer (no dropout during inference)
    logits = dense_forward(h_final, W_dense, b_dense)
    
    # Softmax
    probs = softmax(logits)
    manual_preds.append(probs)

manual_preds = np.array(manual_preds)
manual_pred_classes = manual_preds.argmax(axis=1)

# Compare with Keras predictions (first 50 samples)
keras_pred_classes_subset = predicted_classes[:len(manual_pred_classes)]
labels_valid_subset = labels_valid[:len(manual_pred_classes)]

# Evaluation
manual_f1 = f1_score(labels_valid_subset, manual_pred_classes, average='macro')
keras_f1_subset = f1_score(labels_valid_subset, keras_pred_classes_subset, average='macro')

print(f"\nComparison (first {len(manual_pred_classes)} samples):")
print(f"Manual LSTM F1 Score: {manual_f1:.4f}")
print(f"Keras LSTM F1 Score:  {keras_f1_subset:.4f}")
print(f"Difference: {abs(manual_f1 - keras_f1_subset):.4f}")

# Check if predictions match
matches = (manual_pred_classes == keras_pred_classes_subset).sum()
print(f"Matching predictions: {matches}/{len(manual_pred_classes)} ({matches/len(manual_pred_classes)*100:.1f}%)")

# Show some example predictions
print(f"\nExample predictions (first 5):")
print("Manual:", manual_pred_classes[:5])
print("Keras: ", keras_pred_classes_subset[:5])
print("True:  ", labels_valid_subset[:5])