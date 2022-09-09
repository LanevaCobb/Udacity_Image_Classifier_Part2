#matplotlib inline -- needed only for Jupyter to see plots
#config InlineBackend.figure_format = 'retina'
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



#Training a network	train.py successfully trains a new network on a dataset of images

data_dir = 'ImageClassifier/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                       std = [0.229, 0.224, 0.225])]
                                      )

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                       std = [0.229, 0.224, 0.225])])
                            
                                      
                
train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                           std = [0.229, 0.224, 0.225])])
                                      
                

# TODO: Load the datasets with ImageFolder

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
image_datasets = [train_dataset, test_dataset, valid_dataset]
                                      

# TODO: Using the image datasets and the trainforms, define the dataloaders

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
data_loaders = [train_loader, valid_loader, test_loader]

with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f) 
    
# TODO: Build and train your network

in_args = get_input_args()

check_command_line_arguments(in_args)

model = in_args.arch

if in_args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif in_args.arch == 'vgg19':
    model = models.vgg19(pretrained=True)


for param in model.parameters():    
    param.requires_grad = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = 'ImageClassifier/checkpoint.pth'

#model.classifier = checkpoint['classifier']
#odel.class_to_idx = checkpoint['class_to_idx']
classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(25088, in_args.hidden_units)),
                           ('dropout', nn.Dropout(p=0.5)), 
                           ('relu', nn.ReLU()),
                           ('fc2', nn.Linear(in_args.hidden_units, 102)),
                           ('output', nn.LogSoftmax(dim=1))
                           ]))
                           
    
    
                        
model.classifier = classifier
model
#Training validation log	The training loss, validation loss, and validation accuracy are printed out as a network trains
epochs = in_args.epochs
steps = 0

#print_every = 7

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.003) #update the weights with gradients and learning rate

train_losses, test_losses = [], []

for e in range(epochs):
    running_loss = 0
    
    #training loop
    for images, labels in train_loader:
        
        if model == 'alexnet':
            images = images.view(images.shape[0],-1)
        else:
            images, labels = images.to(device), labels.to(device)
        #images = images.view(images.shape[0],-1)
        #steps += 1
        model.to(device)
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad() # clears up the gradients so there is no accumulation of gradients through each pass.
        #output = model.forward(images) #forward pass
        log_ps = model(images) #images/input passing through the mode.
        loss = criterion(log_ps, labels)
        loss.backward() # backward pass
        optimizer.step() # optimizer step to update the gradients.
        
        running_loss += loss.item()
        
        
    else:
        #print(f"Training loss: {running_loss/len(train_loader)}")

        #if steps % print_every == 0:
        model.eval()
        test_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))            
            
    print(f"Epoch {e+1}/{epochs}"
          f" Training loss: {running_loss/(len(train_loader)):.3f}"
          f" Validation loss: {test_loss/len(valid_loader): 3f}"
          f" Accuracy: {accuracy.item()*100: 3f}%")
            
                           
    model.train()      
    
    
model.eval()
accuracy = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        log_ps = model(images)
        test_loss += criterion(log_ps, labels)
        
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
            
    print(f"Testing Accuracy: {accuracy.item()*100: 3f}%")
    
model.class_to_idx = train_dataset.class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'classifier' : classifier,
              'learning_rate': in_args.learning_rate,
              'batch_size': 64,
              'epochs': in_args.epochs,
              'arch': 'vgg16',
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }

torch.save(checkpoint, 'checkpoint.pth')   
