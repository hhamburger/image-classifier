import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}


def load_data(where  = "./flowers" ):
    '''
    Arguments : the datas' path
    Returns : The loaders for the train, validation and test datasets
    This function receives the location of the image files, applies the necessery transformations (rotations,flips,normalizations and crops) and converts the images to tensor in order to be able to be fed into the neural network
    '''

    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Apply the required transfomations to the test dataset in order to maximize the efficiency of the learning
    #process


    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Crop and Resize the data and validation images in order to be able to be fed into the network

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir ,transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # The data loaders are going to use to load the data to the NN(no shit Sherlock)
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 20, shuffle = True)

    return trainloader , validloader, testloader

def nn_setup(structure='vgg16',dropout=0.5, hidden_layer1 = 120,lr = 0.001,power='gpu'):
    '''
    Arguments: The architecture for the network(alexnet,densenet121,vgg16), the hyperparameters for the network (hidden layer 1 nodes, dropout and learning rate) and whether to use gpu or not
    Returns: The set up model, along with the criterion and the optimizer fo the Training
    '''

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model. Choose vgg16,densenet121,or alexnet.".format(structure))

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('dropout',nn.Dropout(dropout)),
                              ('inputs', nn.Linear(arch[structure], hidden_layer1)),
                              ('relu1', nn.ReLU()),
                              ('hidden1', nn.Linear(hidden_layer1, 90)),
                              ('relu2', nn.ReLU()),
                              ('hidden2', nn.Linear(90,80)),
                              ('relu3', nn.ReLU()),
                              ('hidden3', nn.Linear(80,102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    if torch.cuda.is_available() and power == 'gpu':
        model.cuda()

    return model, criterion, optimizer


def train_network(model, criterion, optimizer, epochs=3, print_every=20, trainloader='trainloader', power='gpu'):
    '''
    Arguments: The model, the criterion, the optimizer, the number of epochs, the dataset, and whether to use a gpu or not
    Returns: Nothing
    This function trains the model over a certain number of epochs and displays the training,validation and accuracy every "print_every" step using cuda if specified. The training method is specified by the criterion and the optimizer which are NLLLoss and Adam respectively
    '''
    steps = 0
    running_loss = 0
    print("Training is starting!")
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            print(steps)
            if torch.cuda.is_available() and power == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            #forward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
        
            #backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                for inputs2,labels2 in validloader:
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')
                    with torch.no_grad():
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, labels)
                        valid_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                

    print("Finished training.")
    print("The Neural Network machine trained your model. It required")
    print("Epochs: {}".format(epochs))
    print("Steps: {}".format(steps))
    
def save_checkpoint(file_path='checkpoint.pth',structure ='vgg16', hidden_layer1=120,dropout=0.5,learn_rate=0.001,epochs=12):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing
    This function saves the model at a specified by the user path
    '''
    
    model.class_to_idx = train_datasets.class_to_idx
    model.cpu
    torch.save({'structure' : structure,
                'dropout' : dropout,
                'hidden_layer': hidden_layer1,
                'learn_rate' : learn_rate,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                file_path)
    return model
                    
def load_checkpoint(file_path='checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases
    '''
    checkpoint = torch.load(file_path)
    structure = checkpoint['structure']
    dropout = checkpoint['dropout']
    hidden_layer = checkpoint['hidden_layer']
    learn_rate = checkpoint['learn_rate']
    model,_,_ = nn_setup(structure, dropout, hidden_layer, learn_rate)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image_path):
    '''
    Arguments: The image's path
    Returns: The image as a numpy array
    This function opens the image usign the PIL package, applies the  necessery transformations and returns the image as a numpy array
    '''
    image_loader = transforms.Compose([
                   transforms.Resize(256), 
                   transforms.CenterCrop(224), 
                   transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = image_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image

def predict(image_path, model, topk=5,power='gpu'):
    '''
    Arguments: The path to the image, the model, the number of prefictions and whether cuda will be used or not
    Returns: The "topk" most probable choices that the network predicts
    '''

    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda:0')

    np_array = process_image(image_path)
    image_tensor = torch.from_numpy(np_array)
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

        ps = torch.exp(output).data.topk(topk)
        classes = ps[1].cpu()
        probs = ps[0].cpu()
        class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
        mapped_classes = []
        for label in classes.numpy()[0]:
            mapped_classes.append(class_to_idx_inverted[label])
        return probs[0], mapped_classes