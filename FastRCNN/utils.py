import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os,sys
import pickle
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

def load_data_to_numpy(folder):

    X = []
    Y = []
    feat = "image_embeddings.npy"
    composed_transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    label_map = {label: idx for idx, label in enumerate(os.listdir(folder))}
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            i=0
            for filename in os.listdir(label_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    i=i+1
                    img_path = os.path.join(label_path, filename)
                    # Load image and convert to numpy array
                    img = Image.open(img_path)
                    img = composed_transform(img)
                    img_array = np.array(img)
                    # Assign label based on folder name
                    y = label_map[label]
                    X.append(img_array)
                    Y.append(y)
                    if(i>=45):
                        break
    # try:
    #     X_feats = np.load(os.path.join(folder, feat))
    # except FileNotFoundError:
    #     print("Embeddings are absent in the input folder")
    #     X_feats = images_to_embeddings(X)
    # X_feats = []
    # print(len(X[0]))

    print(len(X[1]))    
    X = np.array(X)
    Y = np.array(Y)
    return X, X, Y


def get_various_data(X, Y, X_feats, temp_len, validation_size = 100, test_size = 200, L_size = 100, U_size = None):
    
    print("Unlablled")
    if U_size == None:
        U_size = X.shape[0]- L_size - validation_size - test_size

    print(U_size)
    index = np.arange(X.shape[0])
    print(index.size)
    print("second")
    index = np.random.permutation(index)
    print("third")
    X = X[index,:,:,:]
    Y = Y[index]
    X_feats = X_feats[index]

    print("validation")
    X_V = X[-validation_size:]
    Y_V = Y[-validation_size:]
    X_feats_V = X_feats[-validation_size:]
    R_V = np.zeros((validation_size, temp_len))

    print("test")
    X_T = X[-(validation_size+test_size):-validation_size]
    Y_T = Y[-(validation_size+test_size):-validation_size]
    X_feats_T = X_feats[-(validation_size+test_size):-validation_size]
    R_T = np.zeros((test_size,temp_len))

    print("Labeled")
    X_L = X[-(validation_size+test_size+L_size):-(validation_size+test_size)]
    Y_L = Y[-(validation_size+test_size+L_size):-(validation_size+test_size)]
    X_feats_L = X_feats[-(validation_size+test_size+L_size):-(validation_size+test_size)]
    R_L = np.zeros((L_size,temp_len))

    # X_U = X[:-(validation_size+test_size+L_size)]
    X_U = X[:U_size]
    X_feats_U = X_feats[:U_size]
    # Y_U = Y[:-(validation_size+test_size+L_size)]
    R_U = np.zeros((U_size,temp_len))

    return X_V,Y_V,X_feats_V,R_V, X_T,Y_T,X_feats_T,R_T, X_L,Y_L,X_feats_L,R_L, X_U,X_feats_U,R_U

def get_test_U_data(X, Y, X_feats, temp_len, test_size = 200, U_size = None):
    if U_size == None:
        U_size = X.shape[0]- test_size
    index = np.arange(X.shape[0])
    index = np.random.permutation(index)
    X = X[index,:,:,:]
    Y = Y[index]
    X_feats = X_feats[index]

    X_T = X[-(test_size):,:,:,:]
    Y_T = Y[-(test_size):]
    X_feats_T = X_feats[-(test_size):,:,:,:]
    R_T = np.zeros((test_size,temp_len))

    # X_U = X[:-(validation_size+test_size+L_size)]
    X_U = X[:U_size]
    X_feats_U = X_feats[:U_size]
    # Y_U = Y[:-(validation_size+test_size+L_size)]
    R_U = np.zeros((U_size,temp_len))

    return X_T,Y_T,X_feats_T,R_T, X_U,X_feats_U,R_U



