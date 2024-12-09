import sys
sys.path.append('../../')
import enum
from PIL import Image
import torchvision.transforms as transforms
from spear.labeling import continuous_scorer
import torch
from torchvision.transforms import ToPILImage
# from gensim.parsing.preprocessing import STOPWORDS
# from gensim.models.keyedvectors import KeyedVectors 
# import gensim.matutils as gm
from torch.nn.functional import softmax
# import torch
from torchvision.models import resnet18,vgg16,resnext101_32x8d
import timm
# from lfs import ClassLabels
num_classes=21
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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model1.to(device)
model2.to(device)
model3.to(device)
model4.to(device)

composed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
# print("model loading")
# model = KeyedVectors.load_word2vec_format('../../data/SMS_SPAM/glove_w2v.txt', binary=False)
# print("model loaded")

# def get_word_vectors(btw_words):
#     word_vectors= []
#     for word in btw_words:
#         try:
#             word_vectors.append(model[word])
#         except:
#             temp = 1
#             # store words not avaialble in glove
#     return word_vectors

# def get_similarity(word_vectors,target_word): # sent(list of word vecs) to word similarity
#     similarity = 0
#     target_word_vector = 0
#     try:
#         target_word_vector = model[target_word]
#     except:
#         # store words not avaialble in glove
#         return similarity
#     target_word_sparse = gm.any2sparse(target_word_vector,eps=1e-09)
#     for wv in word_vectors:
#         wv_sparse = gm.any2sparse(wv, eps=1e-09)
#         similarity = max(similarity,gm.cossim(wv_sparse,target_word_sparse))
#     return similarity

# def preprocess(tokens):
#     btw_words = [word for word in tokens if word not in STOPWORDS]
#     btw_words = [word for word in btw_words if word.isalpha()]
#     return btw_words


# @continuous_scorer()
# def word_similarity(sentence,**kwargs):
#     similarity = 0.0
#     words = sentence.split()
#     words = preprocess(words)
#     word_vectors = get_word_vectors(words)
#     for w in kwargs['keywords']:
#         similarity = min(max(similarity,get_similarity(word_vectors,w)),1.0)

#     return similarity

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def prediction(idx, image):
    
    # predictionn = 0
    if (idx == 1):
        predictionn = softmax(model1(image))
    elif(idx == 2):
        predictionn = softmax(model2(image))
    elif(idx == 3):
        predictionn = softmax(model3(image))
    else:
        predictionn = softmax(model4(image))

    return predictionn


@continuous_scorer()
def prob(image,**kwargs):
    idx=kwargs['keywords'][0]
    cl=kwargs['keywords'][1]
    image=Image.fromarray(image)
    image = composed_transform(image)
    image=image.to(device)  
    image = torch.unsqueeze(image,0)   
    predictionn= prediction(idx,image)
    print(predictionn)
    print(cl.value)
    print(predictionn[0][cl.value])
    predictionn= predictionn.cpu().detach()
    return predictionn[0][cl.value]

