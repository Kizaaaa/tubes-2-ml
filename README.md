# Tugas Besar 2 IF3270 Pembelajaran Mesin

## Deskripsi singkat

Tugas Besar II kuliah IF3270 Pembelajaran Mesin mengimplementasikan Convolutional Neural Network (CNN) dan Recurrent Neural Network. Pada tugas besar ini, diimplementasikan modul forward propagation CNN dan RNN from scratch.

## Struktur repository

Repository terdiri dari 5 Folder utama

1. CNN : Berisi source code bagian Convolutional Neural Network
2. RNN : Berisi source code bagian Simple Recurrent Neural Network
3. LSTM : Berisi source code bagian Long-Short Term Memory Network
4. datasets : Berisi dataset [NusaX-Sentiment (Bahasa Indonesia)](https://github.com/IndoNLP/nusax/tree/main/datasets/sentiment/indonesian)
5. doc : Berisi laporan tugas besar 2

## Cara setup dan run program

Prerequisite : Python, modul keras

Jalankan file .ipynb pada masing masing folder

Untuk CNN, dapat dijalankan dengan :
1. cnn_forward.py ; ini adalah modul implementasi forward propagation CNN yang bisa dipanggil.
2. cnn_scratch_nb.ipynb : ini adalah notebook pengujian hasil implementasi scratch dengan tensorflow.
3. notebook_eksperimen.ipynb : ini adalah notebook pengujian dengan tensorflow. 

Untuk RNN, lakukan ini setelah menjalankan file .ipynb :

1. Jalankan keras-reference.py
2. Jalankan manual-forward.py

Untuk LSTM, lakukan ini setelah menjalankan file .ipynb :

1. Jalankan main.py

## Pembagian tugas tiap anggota kelompok

Berikut adalah pembagian tugas tiap anggota kelompok:

| NIM | Nama | Tugas |
|-|-|-|
| 12822058 | Konstan Aftop Anewata Ndruru | CNN |
| 13522059 | Dzaky Satrio Nugroho | LSTM |
| 13522065 | Rafiki Prawhira Harianto | Simple RNN |
