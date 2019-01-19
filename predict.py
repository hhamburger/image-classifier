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

import futils

#Command Line Arguments

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='/home/workspace/aipnd-project/flowers/test/1/image_06752.jpg', nargs='*', type=str, action="store")
ap.add_argument('checkpoint', default='/home/workspace/aipnd-project/checkpoint.pth', nargs='*', type=str, action="store")
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
checkpoint_path = pa.checkpoint

model = futils.load_checkpoint(checkpoint_path)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
# Process image and predict label via model  
img = futils.process_image(input_img)

probabilities = futils.predict(img, model, number_of_outputs, power)

# Display probabilities and labels for each output specified
labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])

print("\n\n**Results from image {} using pretrained model checkpoint {}**".format(path_image, checkpoint_path))
i = 0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print("Finished")