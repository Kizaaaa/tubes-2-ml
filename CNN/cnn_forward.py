import numpy as np
from tqdm import tqdm

def conv2d_forward(input_data, kernel, bias, strides=(1, 1), padding=0):
    N, H_in, W_in, C_in = input_data.shape
    K_H, K_W, _, C_out = kernel.shape
    S_H, S_W = strides

    if padding == 0:
        P_H, P_W = 0, 0
        H_out = (H_in - K_H) // S_H + 1
        W_out = (W_in - K_W) // S_W + 1
    else:
        raise ValueError("Belum mendukung Padding!")
    output_tensor = np.zeros((N, H_out, W_out, C_out))
    for n in tqdm(range(N)):
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * S_H
                h_end = h_start + K_H
                w_start = w_out * S_W
                w_end = w_start + K_W
                input_patch = input_data[n, h_start:h_end, w_start:w_end, :]
                for c_out in range(C_out):
                    current_kernel = kernel[:, :, :, c_out]
                    current_bias = bias[c_out]
                    conv_sum = np.sum(input_patch * current_kernel)
                    output_tensor[n, h_out, w_out, c_out] = relu(conv_sum + current_bias)
                    
    return output_tensor

def max_pooling2d_forward(input_data, pool_size=(2, 2), strides=None, padding=0):
    N, H_in, W_in, C_in = input_data.shape
    P_H, P_W = pool_size

    if strides is None:
        S_H, S_W = P_H, P_W
    else:
        S_H, S_W = strides

    if padding == 0:
        H_out = (H_in - P_H) // S_H + 1
        W_out = (W_in - P_W) // S_W + 1
        padded_input = input_data
    else:
        raise ValueError("Belum mendukung Padding!")

    output_tensor = np.zeros((N, H_out, W_out, C_in))

    for n in tqdm(range(N)):
        for c in range(C_in):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * S_H
                    h_end = h_start + P_H
                    w_start = w_out * S_W
                    w_end = w_start + P_W
                    input_patch = padded_input[n, h_start:h_end, w_start:w_end, c]
                    max_val = np.max(input_patch)
                    output_tensor[n, h_out, w_out, c] = max_val

    return output_tensor

def flatten(pooled):
  batch_size = pooled.shape[0]
  output_tensor = pooled.reshape(batch_size, -1)
  return output_tensor

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def dense_forward(input_data, kernel, bias, activation_fn=None):
    z = np.dot(input_data, kernel)
    z_plus_bias = z + bias
    if activation_fn is not None:
        output_tensor = activation_fn(z_plus_bias)
    else:
        output_tensor = z_plus_bias
    return output_tensor