import numpy as np
import re
import enum
from spear.labeling import labeling_function, LFSet, ABSTAIN, preprocessor
import torch
from torchvision.models import resnet18,vgg16,resnext101_32x8d
import timm
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os,sys
import pickle
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
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
            for filename in os.listdir(label_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(label_path, filename)
                    # Load image and convert to numpy array
                    img = Image.open(img_path)
                    img = composed_transform(img)
                    img_array = np.array(img)
                    # Assign label based on folder name
                    y = label_map[label]
                    X.append(img_array)
                    Y.append(y)
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

composed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model1=resnet18(pretrained=False)
model1.fc = torch.nn.Linear(model1.fc.in_features, 21)

# model2=vgg16(pretrained=False)
# num_features = model2.classifier[6].in_features
# model2.classifier[6] = torch.nn.Linear(num_features, 21)

state_dict1 = torch.load('/raid/nlp/pranavg/pavan/azeem/RnD/resnet18_weights.pth')
model1.load_state_dict(state_dict1)
model1.eval()
# X, X_feats, Y = load_data_to_numpy("/raid/nlp/pranavg/pavan/azeem/RnD/train_data")
# print(X[0].shape)
image = Image.open("/raid/nlp/pranavg/pavan/azeem/RnD/train_data/aeroplane/image_121.jpg")
# image=np.array(image)
print(type(image))
image=composed_transform(image)
# Display the image
# print("Showing Image")
# plt.imshow(image)
# plt.axis('off')  # Optional: turn off axis
# # plt.show()
# image=torch.tensor(image)
image=torch.unsqueeze(image,0)
# image = image.float()
# image = image.permute(0, 3, 1, 2)
prediction = model1(image)
prediction = torch.argmax(prediction)
print(prediction)