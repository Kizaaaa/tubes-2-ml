import numpy as np

def embedding_forward(token_indices, embedding_matrix):
    return embedding_matrix[token_indices]

def simple_rnn_forward(inputs, Wx, Wh, b):
    hidden_size = Wh.shape[0]
    h_t = np.zeros((hidden_size,))  # Initial hidden state
    
    for t in range(inputs.shape[0]):  # Iterate over sequence length
        h_t = np.tanh(np.dot(inputs[t], Wx) + np.dot(h_t, Wh) + b)
    
    return h_t

def bidirectional_rnn_forward(inputs, Wx_f, Wh_f, b_f, Wx_b, Wh_b, b_b):
    hidden_size = Wh_f.shape[0]
    h_f = np.zeros((hidden_size,))
    h_b = np.zeros((hidden_size,))
    
    for t in range(inputs.shape[0]):
        h_f = np.tanh(np.dot(inputs[t], Wx_f) + np.dot(h_f, Wh_f) + b_f)
        h_b = np.tanh(np.dot(inputs[-(t+1)], Wx_b) + np.dot(h_b, Wh_b) + b_b)
    
    return np.concatenate([h_f, h_b], axis=0)

# def dropout_forward(inputs, dropout_rate):
#     mask = np.random.binomial(1, 1 - dropout_rate, size=inputs.shape)
#     return inputs * mask / (1 - dropout_rate)

def dense_forward(inputs, W_dense, b_dense):
    return np.dot(inputs, W_dense) + b_dense

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Stability trick
    return exp_logits / np.sum(exp_logits)