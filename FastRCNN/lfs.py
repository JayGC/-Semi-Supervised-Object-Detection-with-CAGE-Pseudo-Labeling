import numpy as np
import re
import enum
import torchvision.transforms as transforms
from PIL import Image
import sys
from torch.nn.functional import softmax
from con_scorer import prob
sys.path.append('../../')

print(sys.path)
from spear.labeling import labeling_function, LFSet, ABSTAIN, preprocessor
import torch
from torchvision.models import resnet18,vgg16,resnext101_32x8d
import timm
num_classes=21
#Resnet18
model1=resnet18(pretrained=False)
model1.fc = torch.nn.Linear(model1.fc.in_features, num_classes)
# model1.to(device)
#vgg
model2=vgg16(pretrained=False)
num_features = model2.classifier[6].in_features
model2.classifier[6] = torch.nn.Linear(num_features, num_classes)
# model2.to(device)
#inception
model3=timm.create_model('inception_v4', pretrained=False,num_classes=21)
# model3.to(device)
#resnext
model4=resnext101_32x8d(pretrained=False)
model4.fc = torch.nn.Linear(model4.fc.in_features, num_classes)
# model4.to(device)

state_dict1 = torch.load('/raid/nlp/pranavg/pavan/azeem/RnD/resnet18_weights.pth')
state_dict2 = torch.load('/raid/nlp/pranavg/pavan/azeem/RnD/vgg16_weights.pth')
state_dict3 = torch.load('/raid/nlp/pranavg/pavan/azeem/RnD/inceptionv4_weights.pth')
state_dict4 = torch.load('/raid/nlp/pranavg/pavan/azeem/RnD/resnext101_32x8d_weights.pth')
#Loading the saved pth files in each of these models
model1.load_state_dict(state_dict1)
model2.load_state_dict(state_dict2)
model3.load_state_dict(state_dict3)
model4.load_state_dict(state_dict4)
print("Loaded all of the prelabel models ...")
model1.eval()
model2.eval()
model3.eval()
model4.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model1.to(device)
model2.to(device)
model3.to(device)
model4.to(device)

composed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa','train', 'tvmonitor', 'background']

class ClassLabels(enum.Enum):
    aeroplane = 0
    bicycle = 1
    bird = 2
    boat = 3
    bottle = 4
    bus = 5
    car = 6
    cat = 7
    chair = 8
    cow = 9
    diningtable = 10
    dog = 11
    horse = 12
    motorbike = 13
    person = 14
    pottedplant = 15
    sheep = 16
    sofa = 17
    train = 18
    tvmonitor = 19
    background = 20

# discrete labelling functions

# for class 0 ('aeroplane')--------------------------
@labeling_function(label=ClassLabels.aeroplane )
def lf1d0(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image=torch.unsqueeze(image,0)
    
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'aeroplane':
        return ClassLabels.aeroplane
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.aeroplane )
def lf2d0(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'aeroplane':
        return ClassLabels.aeroplane
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.aeroplane )
def lf3d0(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'aeroplane':
        return ClassLabels.aeroplane
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.aeroplane )
def lf4d0(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'aeroplane':
        return ClassLabels.aeroplane
    else :
        return ABSTAIN
    
# for class 1 ('bicycle')--------------------------
@labeling_function(label=ClassLabels.bicycle )
def lf1d1(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'bicycle':
        return ClassLabels.bicycle
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.bicycle )
def lf2d1(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'bicycle':
        return ClassLabels.bicycle
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.bicycle )
def lf3d1(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'bicycle':
        return ClassLabels.bicycle
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.bicycle )
def lf4d1(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'bicycle':
        return ClassLabels.bicycle
    else :
        return ABSTAIN
    
# for class 2 ('bird')--------------------------
@labeling_function(label=ClassLabels.bird )
def lf1d2(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'bird':
        return ClassLabels.bird
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.bird ) 
def lf2d2(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'bird':
        return ClassLabels.bird
    else :
        return ABSTAIN
@labeling_function(label=ClassLabels.bird )
def lf3d2(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'bird':
        return ClassLabels.bird
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.bird )
def lf4d2(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'bird':
        return ClassLabels.bird
    else :
        return ABSTAIN
    
# for class 3 ('boat')--------------------------
@labeling_function(label=ClassLabels.boat )
def lf1d3(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'boat':
        return ClassLabels.boat
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.boat )
def lf2d3(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'boat':
        return ClassLabels.boat
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.boat )
def lf3d3(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'boat':
        return ClassLabels.boat
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.boat )
def lf4d3(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'boat':
        return ClassLabels.boat
    else :
        return ABSTAIN

# for class 4 ('bottle')--------------------------
@labeling_function(label=ClassLabels.bottle )
def lf1d4(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'bottle':
        return ClassLabels.bottle
    else :
        return ABSTAIN
    
@labeling_function(label=ClassLabels.bottle )
def lf2d4(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'bottle':
        return ClassLabels.bottle
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.bottle )
def lf3d4(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'bottle':
        return ClassLabels.bottle
    else :
        return ABSTAIN
    
@labeling_function(label=ClassLabels.bottle )
def lf4d4(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'bottle':
        return ClassLabels.bottle
    else :
        return ABSTAIN
    
# for class 5 ('bus')--------------------------
@labeling_function(label=ClassLabels.bus )
def lf1d5(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'bus':
        return ClassLabels.bus
    else :
        return ABSTAIN
    
@labeling_function(label=ClassLabels.bus )
def lf2d5(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'bus':
        return ClassLabels.bus
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.bus )    
def lf3d5(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'bus':
        return ClassLabels.bus
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.bus ) 
def lf4d5(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'bus':
        return ClassLabels.bus
    else :
        return ABSTAIN

# for class 6 ('car')
@labeling_function(label=ClassLabels.car )
def lf1d6(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'car':
        return ClassLabels.car
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.car )
def lf2d6(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'car':
        return ClassLabels.car
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.car )
def lf3d6(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'car':
        return ClassLabels.car
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.car )
def lf4d6(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'car':
        return ClassLabels.car
    else :
        return ABSTAIN
## cat  
@labeling_function(label=ClassLabels.cat )
def lf1d7(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'cat':
        return ClassLabels.cat
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.cat )
def lf2d7(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'cat':
        return ClassLabels.cat
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.cat )
def lf3d7(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'cat':
        return ClassLabels.cat
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.cat )
def lf4d7(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'cat':
        return ClassLabels.cat
    else :
        return ABSTAIN

## chair
@labeling_function(label=ClassLabels.chair )
def lf1d8(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'chair':
        return ClassLabels.chair
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.chair )
def lf2d8(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'chair':
        return ClassLabels.chair
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.chair )
def lf3d8(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'chair':
        return ClassLabels.chair
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.chair )
def lf4d8(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'chair':
        return ClassLabels.chair
    else :
        return ABSTAIN

## cow
@labeling_function(label=ClassLabels.cow )
def lf1d9(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'cow':
        return ClassLabels.cow
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.cow )
def lf2d9(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'cow':
        return ClassLabels.cow
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.cow )
def lf3d9(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'cow':
        return ClassLabels.cow
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.cow )
def lf4d9(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'cow':
        return ClassLabels.cow
    else :
        return ABSTAIN

## diningtable
@labeling_function(label=ClassLabels.diningtable )
def lf1d10(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'diningtable':
        return ClassLabels.diningtable
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.diningtable )
def lf2d10(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'diningtable':
        return ClassLabels.diningtable
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.diningtable )
def lf3d10(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'diningtable':
        return ClassLabels.diningtable
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.diningtable )
def lf4d10(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'diningtable':
        return ClassLabels.diningtable
    else :
        return ABSTAIN

## dog
@labeling_function(label=ClassLabels.dog )
def lf1d11(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'dog':
        return ClassLabels.dog
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.dog )
def lf2d11(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'dog':
        return ClassLabels.dog
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.dog )
def lf3d11(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'dog':
        return ClassLabels.dog
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.dog )
def lf4d11(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'dog':
        return ClassLabels.dog
    else :
        return ABSTAIN

## horse
@labeling_function(label=ClassLabels.horse )
def lf1d12(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'horse':
        return ClassLabels.horse
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.horse )
def lf2d12(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'horse':
        return ClassLabels.horse
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.horse )
def lf3d12(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'horse':
        return ClassLabels.horse
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.horse )
def lf4d12(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'horse':
        return ClassLabels.horse
    else :
        return ABSTAIN

## motorbike
@labeling_function(label=ClassLabels.motorbike )   
def lf1d13(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'motorbike':
        return ClassLabels.motorbike
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.motorbike )
def lf2d13(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'motorbike':
        return ClassLabels.motorbike
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.motorbike )
def lf3d13(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'motorbike':
        return ClassLabels.motorbike
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.motorbike )
def lf4d13(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'motorbike':
        return ClassLabels.motorbike
    else :
        return ABSTAIN

## person
@labeling_function(label=ClassLabels.person )
def lf1d14(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'person':
        return ClassLabels.person
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.person )
def lf2d14(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'person':
        return ClassLabels.person
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.person )
def lf3d14(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'person':
        return ClassLabels.person
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.person )
def lf4d14(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'person':
        return ClassLabels.person
    else :
        return ABSTAIN

## pottedplant
@labeling_function(label=ClassLabels.pottedplant )
def lf1d15(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'pottedplant':
        return ClassLabels.pottedplant
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.pottedplant )
def lf2d15(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'pottedplant':
        return ClassLabels.pottedplant
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.pottedplant )
def lf3d15(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'pottedplant':
        return ClassLabels.pottedplant
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.pottedplant )
def lf4d15(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'pottedplant':
        return ClassLabels.pottedplant
    else :
        return ABSTAIN

## sheep
@labeling_function(label=ClassLabels.sheep )
def lf1d16(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'sheep':
        return ClassLabels.sheep
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.sheep )
def lf2d16(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'sheep':
        return ClassLabels.sheep
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.sheep )
def lf3d16(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'sheep':
        return ClassLabels.sheep
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.sheep )
def lf4d16(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'sheep':
        return ClassLabels.sheep
    else :
        return ABSTAIN

## sofa
@labeling_function(label=ClassLabels.sofa )    
def lf1d17(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'sofa':
        return ClassLabels.sofa
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.sofa )
def lf2d17(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'sofa':
        return ClassLabels.sofa
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.sofa )
def lf3d17(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'sofa':
        return ClassLabels.sofa
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.sofa )
def lf4d17(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'sofa':
        return ClassLabels.sofa
    else :
        return ABSTAIN

## train
@labeling_function(label=ClassLabels.train )
def lf1d18(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'train':
        return ClassLabels.train
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.train )
def lf2d18(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'train':
        return ClassLabels.train
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.train )
def lf3d18(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'train':
        return ClassLabels.train
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.train )
def lf4d18(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'train':
        return ClassLabels.train
    else :
        return ABSTAIN

## tvmonitor
@labeling_function(label=ClassLabels.tvmonitor )
def lf1d19(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2) 
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'tvmonitor':
        return ClassLabels.tvmonitor
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.tvmonitor )
def lf2d19(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'tvmonitor':
        return ClassLabels.tvmonitor
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.tvmonitor )
def lf3d19(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'tvmonitor':
        return ClassLabels.tvmonitor
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.tvmonitor )
def lf4d19(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'tvmonitor':
        return ClassLabels.tvmonitor
    else :
        return ABSTAIN

## background
@labeling_function(label=ClassLabels.background )
def lf1d20(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    prediction = torch.argmax(model1(image)) 
    if ClassLabels(prediction.item()).name == 'background':
        return ClassLabels.background
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.background )    
def lf2d20(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    prediction = torch.argmax(model2(image)) 
    if ClassLabels(prediction.item()).name == 'background':
        return ClassLabels.background
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.background )    
def lf3d20(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    prediction = torch.argmax(model3(image)) 
    if ClassLabels(prediction.item()).name == 'background':
        return ClassLabels.background
    else :
        return ABSTAIN

@labeling_function(label=ClassLabels.background )    
def lf4d20(image):
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)
    image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    prediction = torch.argmax(model4(image)) 
    if ClassLabels(prediction.item()).name == 'background':
        return ClassLabels.background
    else :
        return ABSTAIN

#continous labelling fns

threshold = 0.8

# for class 0 ('aeroplane')--------------------------
@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.aeroplane]),label=ClassLabels.aeroplane )
def lf1c0(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.aeroplane
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.aeroplane]),label=ClassLabels.aeroplane )
def lf2c0(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.aeroplane
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.aeroplane]),label=ClassLabels.aeroplane ) 
def lf3c0(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.aeroplane
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.aeroplane]),label=ClassLabels.aeroplane )   
def lf4c0(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.aeroplane
    else :
        return ABSTAIN
    
# for class 1 ('bicycle')--------------------------
@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.bicycle]),label=ClassLabels.bicycle )
def lf1c1(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bicycle
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.bicycle]),label=ClassLabels.bicycle )    
def lf2c1(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bicycle
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.bicycle]),label=ClassLabels.bicycle )    
def lf3c1(image,**kwargs):   
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bicycle
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.bicycle]),label=ClassLabels.bicycle )    
def lf4c1(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bicycle
    else :
        return ABSTAIN
    
# for class 2 ('bird')--------------------------
@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.bird]),label=ClassLabels.bird )
def lf1c2(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bird
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.bird]),label=ClassLabels.bird )    
def lf2c2(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bird
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.bird]),label=ClassLabels.bird )    
def lf3c2(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bird
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.bird]),label=ClassLabels.bird )    
def lf4c2(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bird
    else :
        return ABSTAIN
    
# for class 3 ('boat')--------------------------
@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.boat]),label=ClassLabels.boat )
def lf1c3(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.boat
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.boat]),label=ClassLabels.boat )    
def lf2c3(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.boat
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.boat]),label=ClassLabels.boat )    
def lf3c3(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.boat
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.boat]),label=ClassLabels.boat )    
def lf4c3(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.boat
    else :
        return ABSTAIN

# for class 4 ('bottle')--------------------------
@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.bottle]),label=ClassLabels.bottle )
def lf1c4(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bottle
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.bottle]),label=ClassLabels.bottle )    
def lf2c4(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bottle
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.bottle]),label=ClassLabels.bottle )    
def lf3c4(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bottle
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.bottle]),label=ClassLabels.bottle )    
def lf4c4(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bottle
    else :
        return ABSTAIN
    
# for class 5 ('bus')--------------------------
@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.bus]),label=ClassLabels.bus )
def lf1c5(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bus
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.bus]),label=ClassLabels.bus )    
def lf2c5(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bus
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.bus]),label=ClassLabels.bus )    
def lf3c5(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bus
    else :
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.bus]),label=ClassLabels.bus )    
def lf4c5(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.bus
    else :
        return ABSTAIN

# for class 6 ('car')
@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.car]),label=ClassLabels.car )
def lf1c6(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.car
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.car]),label=ClassLabels.car )
def lf2c6(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.car
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.car]),label=ClassLabels.car )
def lf3c6(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.car
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.car]),label=ClassLabels.car )
def lf4c6(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.car
    else:
        return ABSTAIN
    

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.cat]),label=ClassLabels.cat )
def lf1c7(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.cat
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.cat]),label=ClassLabels.cat )
def lf2c7(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.cat
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.cat]),label=ClassLabels.cat )
def lf3c7(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.cat
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.cat]),label=ClassLabels.cat )
def lf4c7(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.cat
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.chair]),label=ClassLabels.chair )
def lf1c8(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.chair
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.chair]),label=ClassLabels.chair )
def lf2c8(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.chair
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.chair]),label=ClassLabels.chair )
def lf3c8(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.chair
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.chair]),label=ClassLabels.chair )
def lf4c8(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.chair
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.cow]),label=ClassLabels.cow )
def lf1c9(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.cow
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.cow]),label=ClassLabels.cow )
def lf2c9(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.cow
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.cow]),label=ClassLabels.cow )
def lf3c9(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.cow
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.cow]),label=ClassLabels.cow )
def lf4c9(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.cow
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.diningtable]),label=ClassLabels.diningtable )
def lf1c10(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.diningtable
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.diningtable]),label=ClassLabels.diningtable )
def lf2c10(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.diningtable
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.diningtable]),label=ClassLabels.diningtable )
def lf3c10(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.diningtable
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.diningtable]),label=ClassLabels.diningtable )
def lf4c10(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.diningtable
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.dog]),label=ClassLabels.dog )
def lf1c11(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.dog
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.dog]),label=ClassLabels.dog )
def lf2c11(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.dog
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.dog]),label=ClassLabels.dog )
def lf3c11(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.dog
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.dog]),label=ClassLabels.dog )
def lf4c11(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.dog
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.horse]),label=ClassLabels.horse )
def lf1c12(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.horse
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.horse]),label=ClassLabels.horse )
def lf2c12(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.horse
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.horse]),label=ClassLabels.horse )
def lf3c12(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.horse
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.horse]),label=ClassLabels.horse )
def lf4c12(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.horse
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.motorbike]),label=ClassLabels.motorbike )    
def lf1c13(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.motorbike
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.motorbike]),label=ClassLabels.motorbike )
def lf2c13(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.motorbike
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.motorbike]),label=ClassLabels.motorbike )
def lf3c13(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.motorbike
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.motorbike]),label=ClassLabels.motorbike )
def lf4c13(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.motorbike
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.person]),label=ClassLabels.person )
def lf1c14(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.person
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.person]),label=ClassLabels.person )
def lf2c14(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.person
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.person]),label=ClassLabels.person )
def lf3c14(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.person
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.person]),label=ClassLabels.person )
def lf4c14(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.person
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.pottedplant]),label=ClassLabels.pottedplant )
def lf1c15(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.pottedplant
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.pottedplant]),label=ClassLabels.pottedplant )
def lf2c15(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.pottedplant
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.pottedplant]),label=ClassLabels.pottedplant )
def lf3c15(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.pottedplant
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.pottedplant]),label=ClassLabels.pottedplant )
def lf4c15(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.pottedplant
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.sheep]),label=ClassLabels.sheep )
def lf1c16(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.sheep
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.sheep]),label=ClassLabels.sheep )
def lf2c16(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.sheep
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.sheep]),label=ClassLabels.sheep )
def lf3c16(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.sheep
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.sheep]),label=ClassLabels.sheep )
def lf4c16(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.sheep
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.sofa]),label=ClassLabels.sofa )    
def lf1c17(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.sofa
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.sofa]),label=ClassLabels.sofa )
def lf2c17(image,**kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.sofa
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.sofa]),label=ClassLabels.sofa )
def lf3c17(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.sofa
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.sofa]),label=ClassLabels.sofa )
def lf4c17(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.sofa
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.train]),label=ClassLabels.train )
def lf1c18(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.train
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.train]),label=ClassLabels.train )
def lf2c18(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.train
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.train]),label=ClassLabels.train )
def lf3c18(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.train
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.train]),label=ClassLabels.train )
def lf4c18(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.train
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.tvmonitor]),label=ClassLabels.tvmonitor )
def lf1c19(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.tvmonitor
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.tvmonitor]),label=ClassLabels.tvmonitor )
def lf2c19(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.tvmonitor
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.tvmonitor]),label=ClassLabels.tvmonitor )
def lf3c19(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.tvmonitor
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.tvmonitor]),label=ClassLabels.tvmonitor )
def lf4c19(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.tvmonitor
    else:
        return ABSTAIN   

@labeling_function(cont_scorer=prob,resources=dict(keywords= [1,ClassLabels.background]),label=ClassLabels.background )    
def lf1c20(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model1.eval()
    #prediction = softmax(model1(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.background
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [2,ClassLabels.background]),label=ClassLabels.background )
def lf2c20(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model2.eval()
    #prediction = softmax(model2(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.background
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [3,ClassLabels.background]),label=ClassLabels.background )
def lf3c20(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model3.eval()
    #prediction = softmax(model3(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.background
    else:
        return ABSTAIN

@labeling_function(cont_scorer=prob,resources=dict(keywords= [4,ClassLabels.background]),label=ClassLabels.background )
def lf4c20(image, **kwargs):
    # image=Image.fromarray(image)
    # image = composed_transform(image)
    # image=image.to(device)
    # image = torch.unsqueeze(image,0)
    
    # image = image.permute(0, 3, 1, 2)
    #model4.eval()
    #prediction = softmax(model4(image),dim=1)
    if prob(image,**kwargs) >= threshold:
        return ClassLabels.background
    else:
        return ABSTAIN   
    
a = ['c','d']
LFS = [lf1c0, lf1c1, lf1c2, lf1c3, lf1c4, lf1c5, lf1c6, lf1c7, lf1c8, lf1c9, lf1c10, lf1c11, lf1c12, lf1c13, lf1c14, lf1c15, lf1c16, lf1c17, lf1c18, lf1c19, lf1c20, lf2c0, lf2c1, lf2c2, lf2c3, lf2c4, lf2c5, lf2c6, lf2c7, lf2c8, lf2c9, lf2c10, lf2c11, lf2c12, lf2c13, lf2c14, lf2c15, lf2c16, lf2c17, lf2c18, lf2c19, lf2c20, lf3c0, lf3c1, lf3c2, lf3c3, lf3c4, lf3c5, lf3c6, lf3c7, lf3c8, lf3c9, lf3c10, lf3c11, lf3c12, lf3c13, lf3c14, lf3c15, lf3c16, lf3c17, lf3c18, lf3c19, lf3c20, lf4c0, lf4c1, lf4c2, lf4c3, lf4c4, lf4c5, lf4c6, lf4c7, lf4c8, lf4c9, lf4c10, lf4c11, lf4c12, lf4c13, lf4c14, lf4c15, lf4c16, lf4c17, lf4c18, lf4c19, lf4c20, lf1d0, lf1d1, lf1d2, lf1d3, lf1d4, lf1d5, lf1d6, lf1d7, lf1d8, lf1d9, lf1d10, lf1d11, lf1d12, lf1d13, lf1d14, lf1d15, lf1d16, lf1d17, lf1d18, lf1d19, lf1d20, lf2d0, lf2d1, lf2d2, lf2d3, lf2d4, lf2d5, lf2d6, lf2d7, lf2d8, lf2d9, lf2d10, lf2d11, lf2d12, lf2d13, lf2d14, lf2d15, lf2d16, lf2d17, lf2d18, lf2d19, lf2d20, lf3d0, lf3d1, lf3d2, lf3d3, lf3d4, lf3d5, lf3d6, lf3d7, lf3d8, lf3d9, lf3d10, lf3d11, lf3d12, lf3d13, lf3d14, lf3d15, lf3d16, lf3d17, lf3d18, lf3d19, lf3d20, lf4d0, lf4d1, lf4d2, lf4d3, lf4d4, lf4d5, lf4d6, lf4d7, lf4d8, lf4d9, lf4d10, lf4d11, lf4d12, lf4d13, lf4d14, lf4d15, lf4d16, lf4d17, lf4d18, lf4d19, lf4d20]

rules = LFSet("RULES_LF")
print(rules.name)
print(LFS)
rules.add_lf_list(LFS)

