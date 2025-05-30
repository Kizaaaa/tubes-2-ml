import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

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

tokenizer = TextVectorization(max_tokens=10000, output_mode="int", output_sequence_length=100)
tokenizer.adapt(texts)

tokenized_texts = tokenizer(texts)

model_uni = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    LSTM(64),
    Dropout(0.5), 
    Dense(3, activation="softmax")
])

model_uni.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model_bi = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(3, activation="softmax")
])

model_bi.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['label'].values)
labels_valid = label_encoder.transform(data_valid['label'].values)

texts_valid = data_valid['text'].values
tokenized_texts_valid = tokenizer(texts_valid)

model_uni.fit(
    tokenized_texts, labels,
    validation_data=(tokenized_texts_valid, labels_valid),
    epochs=3,
    verbose=1
)

model_bi.fit(
    tokenized_texts, labels,
    validation_data=(tokenized_texts_valid, labels_valid),
    epochs=3,
    verbose=1
)

predictions_uni = model_uni.predict(tokenized_texts_valid, verbose=0)
predicted_classes_uni = predictions_uni.argmax(axis=1)

predictions_bi = model_bi.predict(tokenized_texts_valid, verbose=0)
predicted_classes_bi = predictions_bi.argmax(axis=1)

keras_f1_uni = f1_score(labels_valid, predicted_classes_uni, average='macro')
keras_f1_bi = f1_score(labels_valid, predicted_classes_bi, average='macro')

# Save weights
model_uni.save_weights("lstm_uni.weights.h5")
model_bi.save_weights("lstm_bi.weights.h5")

# Extract weights
def extract_lstm_weights(model, is_bidirectional=False):
    
    embedding_matrix = model.layers[0].get_weights()[0]  # (10000, 128)
    
    if is_bidirectional:
        bi_layer = model.layers[1]
        
        forward_weights = bi_layer.forward_layer.get_weights()
        W_combined_f = forward_weights[0]
        U_combined_f = forward_weights[1]
        b_combined_f = forward_weights[2]
        
        backward_weights = bi_layer.backward_layer.get_weights()
        W_combined_b = backward_weights[0]
        U_combined_b = backward_weights[1]
        b_combined_b = backward_weights[2]
        
        W_dense, b_dense = model.layers[3].get_weights()
        
        return (embedding_matrix, 
                W_combined_f, U_combined_f, b_combined_f,
                W_combined_b, U_combined_b, b_combined_b,
                W_dense, b_dense)
    else:
        lstm_weights = model.layers[1].get_weights()
        W_combined = lstm_weights[0]
        U_combined = lstm_weights[1]
        b_combined = lstm_weights[2]
        
        W_dense, b_dense = model.layers[3].get_weights()
        
        return (embedding_matrix, W_combined, U_combined, b_combined, W_dense, b_dense)

def split_lstm_gates(W_combined, U_combined, b_combined, hidden_size):
    
    Wi = W_combined[:, :hidden_size]
    Wf = W_combined[:, hidden_size:2*hidden_size]
    Wc = W_combined[:, 2*hidden_size:3*hidden_size]
    Wo = W_combined[:, 3*hidden_size:]

    Ui = U_combined[:, :hidden_size]
    Uf = U_combined[:, hidden_size:2*hidden_size]
    Uc = U_combined[:, 2*hidden_size:3*hidden_size]
    Uo = U_combined[:, 3*hidden_size:]

    bi = b_combined[:hidden_size]
    bf = b_combined[hidden_size:2*hidden_size]
    bc = b_combined[2*hidden_size:3*hidden_size]
    bo = b_combined[3*hidden_size:]
    
    return Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf, bc, bo

new_model_uni = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    LSTM(64),
    Dropout(0.5), 
    Dense(3, activation="softmax")
])
new_model_uni.build(input_shape=(None, 100))
new_model_uni.load_weights("lstm_uni.weights.h5")

weights_uni = extract_lstm_weights(new_model_uni, is_bidirectional=False)
embedding_matrix, W_combined, U_combined, b_combined, W_dense_uni, b_dense_uni = weights_uni

Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf, bc, bo = split_lstm_gates(W_combined, U_combined, b_combined, 64)

new_model_bi = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(3, activation="softmax")
])
new_model_bi.build(input_shape=(None, 100))
new_model_bi.load_weights("lstm_bi.weights.h5")

weights_bi = extract_lstm_weights(new_model_bi, is_bidirectional=True)
(embedding_matrix_bi, W_combined_f, U_combined_f, b_combined_f,
 W_combined_b, U_combined_b, b_combined_b, W_dense_bi, b_dense_bi) = weights_bi

Wi_f, Wf_f, Wc_f, Wo_f, Ui_f, Uf_f, Uc_f, Uo_f, bi_f, bf_f, bc_f, bo_f = split_lstm_gates(
    W_combined_f, U_combined_f, b_combined_f, 32)

Wi_b, Wf_b, Wc_b, Wo_b, Ui_b, Uf_b, Uc_b, Uo_b, bi_b, bf_b, bc_b, bo_b = split_lstm_gates(
    W_combined_b, U_combined_b, b_combined_b, 32)

# Manual LSTM Forward

from mylstm import embedding_forward, lstm_forward, bidirectional_lstm_forward, dense_forward, softmax

N_SAMPLES = 20
test_indices = range(min(N_SAMPLES, tokenized_texts_valid.shape[0]))

manual_preds_uni = []
for i in test_indices:
    emb = embedding_forward(tokenized_texts_valid[i], embedding_matrix)
    
    h_final, _, _ = lstm_forward(emb, Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo, Wc, Uc, bc)
    
    logits = dense_forward(h_final, W_dense_uni, b_dense_uni)
    
    probs = softmax(logits)
    manual_preds_uni.append(probs)

manual_preds_uni = np.array(manual_preds_uni)
manual_pred_classes_uni = manual_preds_uni.argmax(axis=1)

manual_preds_bi = []
for i in test_indices:
    emb = embedding_forward(tokenized_texts_valid[i], embedding_matrix_bi)
    
    h_combined = bidirectional_lstm_forward(
        emb,
        Wf_f, Uf_f, bf_f, Wi_f, Ui_f, bi_f, Wo_f, Uo_f, bo_f, Wc_f, Uc_f, bc_f,
        Wf_b, Uf_b, bf_b, Wi_b, Ui_b, bi_b, Wo_b, Uo_b, bo_b, Wc_b, Uc_b, bc_b
    )
    
    logits = dense_forward(h_combined, W_dense_bi, b_dense_bi)
    
    probs = softmax(logits)
    manual_preds_bi.append(probs)

manual_preds_bi = np.array(manual_preds_bi)
manual_pred_classes_bi = manual_preds_bi.argmax(axis=1)

keras_pred_classes_uni_subset = predicted_classes_uni[:len(test_indices)]
keras_pred_classes_bi_subset = predicted_classes_bi[:len(test_indices)]
labels_valid_subset = labels_valid[:len(test_indices)]

manual_f1_uni = f1_score(labels_valid_subset, manual_pred_classes_uni, average='macro')
manual_f1_bi = f1_score(labels_valid_subset, manual_pred_classes_bi, average='macro')
keras_f1_uni_subset = f1_score(labels_valid_subset, keras_pred_classes_uni_subset, average='macro')
keras_f1_bi_subset = f1_score(labels_valid_subset, keras_pred_classes_bi_subset, average='macro')
    
print(f"\n=== BIDIRECTIONAL vs UNIDIRECTIONAL COMPARISON ===")
print(f"Manual Implementation:")
print(f"  Unidirectional F1: {manual_f1_uni:.4f}")
print(f"  Bidirectional F1:  {manual_f1_bi:.4f}")

print(f"\nKeras Implementation:")
print(f"  Unidirectional F1: {keras_f1_uni:.4f}")
print(f"  Bidirectional F1:  {keras_f1_bi:.4f}")