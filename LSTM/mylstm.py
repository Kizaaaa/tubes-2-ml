import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(np.clip(x, -500, 500))

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def lstm_forward(inputs, Wf, Uf, bf, Wi, Ui, bi, Wo, Uo, bo, Wc, Uc, bc):
    
    seq_len, input_dim = inputs.shape
    hidden_size = Wf.shape[1]
    
    # Inisialisasi states
    h_t = np.zeros((hidden_size,))
    c_t = np.zeros((hidden_size,))
    
    all_hidden_states = []
    all_cell_states = []
    
    for t in range(seq_len):
        x_t = inputs[t]
        
        # 1. Forget Gate
        # f_t = σ(W_f * x_t + U_f * h_{t-1} + b_f)
        f_t = sigmoid(np.dot(x_t, Wf) + np.dot(h_t, Uf) + bf)
        
        # 2. Input Gate
        # i_t = σ(W_i * x_t + U_i * h_{t-1} + b_i)  
        i_t = sigmoid(np.dot(x_t, Wi) + np.dot(h_t, Ui) + bi)
        
        # 3. Candidate Values
        # c̃_t = tanh(W_c * x_t + U_c * h_{t-1} + b_c)
        c_tilde = tanh(np.dot(x_t, Wc) + np.dot(h_t, Uc) + bc)
        
        # 4. Update Cell State
        # c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        c_t = f_t * c_t + i_t * c_tilde
        
        # 5. Output Gate
        # o_t = σ(W_o * x_t + U_o * h_{t-1} + b_o)
        o_t = sigmoid(np.dot(x_t, Wo) + np.dot(h_t, Uo) + bo)
        
        # 6. Update Hidden State
        # h_t = o_t ⊙ tanh(c_t)
        h_t = o_t * tanh(c_t)
        
        # Store states
        all_hidden_states.append(h_t.copy())
        all_cell_states.append(c_t.copy())
    
    return h_t, np.array(all_hidden_states), np.array(all_cell_states)

def bidirectional_lstm_forward(inputs, Wf_f, Uf_f, bf_f, Wi_f, Ui_f, bi_f, Wo_f, Uo_f, bo_f, Wc_f, Uc_f, bc_f, Wf_b, Uf_b, bf_b, Wi_b, Ui_b, bi_b, Wo_b, Uo_b, bo_b, Wc_b, Uc_b, bc_b):    
    # Forward LSTM
    h_f, _, _ = lstm_forward(inputs, Wf_f, Uf_f, bf_f, Wi_f, Ui_f, bi_f, Wo_f, Uo_f, bo_f, Wc_f, Uc_f, bc_f)
    
    # Backward LSTM
    inputs_reversed = inputs[::-1]
    h_b, _, _ = lstm_forward(inputs_reversed, Wf_b, Uf_b, bf_b, Wi_b, Ui_b, bi_b, Wo_b, Uo_b, bo_b, Wc_b, Uc_b, bc_b)
    
    return np.concatenate([h_f, h_b], axis=0)

def embedding_forward(token_indices, embedding_matrix):
    return embedding_matrix[token_indices]

def dense_forward(inputs, W_dense, b_dense):
    return np.dot(inputs, W_dense) + b_dense