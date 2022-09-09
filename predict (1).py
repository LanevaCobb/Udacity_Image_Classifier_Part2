from get_input_args import get_input_args
import matplotlib.pyplot as plt
import os, random, sys
import torch
import json
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
import random
import time
import numpy as np
from check_command_line import check_command_line_arguments

in_args = get_input_args()

check_command_line_arguments(in_args)

train_set = in_args.train_set

#def load_checkpoint(filepath):
    #checkpoint = torch.load('checkpoint.pth',map_location=lambda storage, loc:storage)
    #model = in_args.arch
    
    #classifier = nn.Sequential(OrderedDict([
                           #('fc1', nn.Linear(25088, in_args.hidden_units)),
                           #('dropout', nn.Dropout(p=0.5)), 
                           #('relu', nn.ReLU()),
                           #('fc2', nn.Linear(in_args.hidden_units, 102)),
                           #('output', nn.LogSoftmax(dim=1))
                           #]))
    #model.classifier : classifier
    #model.load_state_dict : checkpoint['state_dict']
    #return model, checkpoint['class_to_idx']

#checkpoint = load_checkpoint('checkpoint.pth')

checkpoint = torch.load('checkpoint.pth',map_location=lambda storage, loc:storage)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if checkpoint['arch'] == 'vgg19':
    model = models.vgg19(pretrained=True)
else:
    model = models.vgg16(pretrained=True)
        
for param in model.parameters():    
    param.requires_grad = False 
       

model.classifier : checkpoint['classifier']
model.class_to_idx : checkpoint['class_to_idx']
model.load_state_dict : checkpoint['state_dict']
#model.opitimizer = checkpoint['optimizer_state_dict']




#model.class_to_idx = train_set.class_to_idx
        
print(checkpoint.keys())
#print(model.class_to_idx)
#print(model.load_state_dict)
print(type(model))
print(type(train_set))


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    #im = Image.open(image)
    Image.open(image_path, encoding="utf8", errors='ignore')
    im = im.resize((256,256))
    crp = .5*(256-224)
    im = im.crop((crp, crp, 256-crp, 256-crp))
    im = np.array(im)/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]    
    im = (im-mean) / std  
    
    transp = im.transpose(2,0,1)

    return torch.from_numpy(transp),type(torch.FloatTensor)
    
    

image_path = in_args.image_path  




def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    model.eval()
   
    with torch.no_grad():
        image = process_image(image_path)
            
    #image = image.view(image.shape[0],-1)
    
    #image = np.expand_dims(image, 0)
    
        inputs = image.to(device)
        image = torch.from_numpy(np.array([image])).float()
         
    #inputs = Variable(image).to(device)
   
    
    
        logits = model.forward(inputs)
    
        ps = torch.exp(logits)
    
    #ps = F.softmax(logits, dim=1)
    
        prob = torch.topk(ps, topk)[0].tolist()[0]
        index = torch.topk(ps, topk)[1].tolist()[0]
    #topk = ps.cpu().topk(topk)
    
        ind = []
        for i in range(len(model.class_to_idx.items())):
            ind.append(list(model.class_to_idx.items())[i][0])
        
        label = []
        for i in range(5):
            label.append(ind[index[i]])
        
        return prob, label
    

#image_path = in_args.image_path
train_dataset = 'ImageClassifier/flowers/train'

class_names = train_dataset.classes
probs, classes = predict(image_path, model.to(device))
print(probs)
print(classes)
flower_name = [cat_to_name[x] for x in classes]
#flower_name = [cat_to_name[class_names[e]] for e in classes]
print(flower_name)

print(cat_to_name)
