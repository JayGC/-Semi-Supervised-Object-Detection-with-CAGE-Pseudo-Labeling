

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18,vgg16
from torchvision import datasets
import torch.nn as nn
from torch import optim

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")

model = vgg16(pretrained=True)
num_classes = 21  # Change this to your desired number of classes
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, num_classes)
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
val_loader,classes = dataloaders("val_data",composed_transform,batch_size,shuffle=False)

lr = 1e-4
optimizer = optim.Adam(model.parameters(),lr)
num_epochs = 3
loss_fn = torch.nn.CrossEntropyLoss().to(device)

batches = len(train_loader)
print(f"Number of batches : {batches}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    batch=0
    for x,y in train_loader:
        batch=batch+1
        x,y = x.to(device),y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output,y)
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

torch.save(model.state_dict(), 'vgg16_weights.pth')
print("Saved model")