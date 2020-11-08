import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import PIL
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as prfs
import matplotlib.pyplot as plt
import pathlib
import sys

def filedirinput(s=sys.argv):
    """
    File input reader
    parameter: sys.argv
    return: Image address
    """
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(s) ==2:
        pred_dir = s[1]
    else:
        print("Wrong input. Choose to predict image in ./unknown")
        pred_dir = './unknown'
    return pred_dir
def imshow(inp, title=None):
    """
    Imshow for Tensor
    parameter: transformed image matrix,title=None
    return: showing the image
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
def logoclassifier(model, loc = './unknown'):
    """
    logo classifier
    parameter: Model, Image address
    return: Image class
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("class_names.txt", 'r') as f:
        class_names = [line.rstrip('\n') for line in f]
    unknown_img = datasets.ImageFolder(loc, data_transforms)
    loaded_img = torch.utils.data.DataLoader(unknown_img, batch_size = 1)
    model.eval()
    original_labels = []
    pred_lst = []
    
    for i, (inputs, labels) in enumerate(loaded_img):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        original_labels.extend(labels)
        pred_lst.extend(preds)
        #imshow(inputs.cpu().data[0])
        #print('Predicted: ', class_names[preds[0]])
    precision, recall, f1, support = prfs(original_labels, pred_lst, average='weighted')
    print("Precision: {:.2%}\nRecall: {:.2%}\nF1 score: {:.2%}".format(precision, recall, f1))


def main():
    # Image to be classified
    pred_dir = filedirinput(s=sys.argv)
    # Model input
    model = torch.load("./tlmodel")
    model.eval()
    logoclassifier(model, loc = pred_dir)

if __name__ == "__main__":
    main()