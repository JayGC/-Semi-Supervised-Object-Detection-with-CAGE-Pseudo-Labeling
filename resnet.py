# import torch
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from torchvision.models import resnet18
# from torchvision import datasets
# from torch import optim
# from tqdm import tqdm

# torch.cuda.empty_cache()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# # torch.backends.cudnn.benchmark = True 
# model = resnet18()
# model = model.to(device)

# def dataloaders(data, transform, batch_size):
#     data = datasets.ImageFolder(data, transform=transform)
#     classes = data.classes
#     dataloader = DataLoader(data, batch_size=batch_size)
#     return dataloader, classes

# composed_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# batch_size = 64
# train_loader, classes = dataloaders('train_data', composed_transform, batch_size)
# val_loader, classes = dataloaders('val_data', composed_transform, batch_size)

# lr = 1e-4
# optimizer = optim.Adam(model.parameters(), lr=lr)
# num_epochs = 5
# loss_fn = torch.nn.CrossEntropyLoss().to(device)

# batches = len(train_loader)
# print(f"Number of batches: {batches}")

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     batch = 0
#     with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
#         for x, y in t:
#             batch=batch+1
#             x, y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             output = model(x)
#             loss = loss_fn(output, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             if batch % 20 == 0:
#                 t.set_postfix(loss=total_loss / batch)

#     epoch_loss = total_loss / batches
#     print(f"Train Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

#     #Validation
#     model.eval()
#     val_loss = 0.0

#     with torch.no_grad():
#         batch=0
#         with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as v:
#             for x, y in v:
#                 batch=batch+1
#                 x, y = x.to(device), y.to(device)
#                 output = model(x)
#                 loss = loss_fn(output, y)
#                 val_loss += loss.item()
#                 t.set_postfix(loss=val_loss / batch)
#     val_loss /= batch
#     print(f"Validation Epoch {epoch+1}/{num_epochs}, Loss: {val_loss}")

# torch.save(model.state_dict(), 'resnet18_weights.pth')

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import datasets
from torch import optim
from PIL import Image
import numpy as np
import os,sys
import pickle
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")

model = resnet18(pretrained=True)
num_classes = 21  # Change this to your desired number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.to(device)


def dataloaders(data,transform,batch_size,shuffle):
    
    data = datasets.ImageFolder(data,transform=transform)
    classes = data.classes
    print(data[0])
    dataloader = DataLoader(data,batch_size=batch_size,shuffle=shuffle)

    return dataloader,classes

composed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


batch_size = 64
train_loader,classes = dataloaders('train_data',composed_transform,batch_size,shuffle=True)
# val_loader,classes = dataloaders("val_data",composed_transform,batch_size,shuffle=False)

lr = 1e-4
optimizer = optim.Adam(model.parameters(),lr)
num_epochs = 5
loss_fn = torch.nn.CrossEntropyLoss().to(device)

batches = len(train_loader)
print(f"Number of batches : {batches}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    batch =0

    for x,y in train_loader:
        batch=batch+1
        x,y = x.to(device),y.to(device)
        optimizer.zero_grad()
        output = model(x)
        # print(output)
        # print(y)
        loss = loss_fn(output,y).to(device)
       
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
        if (batch%50 ==0):
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch}/{batches}, Loss: {total_loss/batch}")

    epoch_loss = total_loss/batches

    print(f"Train Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    # model.eval()
    # val_loss = 0.0

    # for x,y in val_loader:
    #     x,y = x.to(device),y.to(device)
    #     output = model(x)
    #     loss = loss_fn(output,y)
    #     val_loss+=loss

    # val_loss = val_loss/batches

    # print(f"Validation Epoch {epoch+1}/{num_epochs}, Loss: {val_loss}")

torch.save(model.state_dict(), 'resnet18_weights.pth')
print("Saved model")
