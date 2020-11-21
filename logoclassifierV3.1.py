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
from torch.utils.data import Dataset
import natsort
from PIL import Image
import matplotlib.pyplot as plt
import pathlib
import sys

def filedirinput(s=sys.argv):
    """
    File input reader
    parameter: sys.argv
    return: Image address
    """
    #print ('Number of arguments:', len(sys.argv), 'arguments.')
    #print ('Argument List:', str(sys.argv))
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
class CustomDataSet(Dataset):
    """
    Custom image loader for torch nn model
    parameter: image file address, image transform funciton 
    return: the transformed image files
    """
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
def logoclassifier(model, loc = './unknown'):
    """
    logo classifier
    parameter: Model, Image address, class names
    return: Image class
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open("class_names.txt", 'r') as f:
        class_names = [line.rstrip('\n') for line in f]
    data_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    unknown_img = CustomDataSet("./unknown", transform=data_transforms)
    loaded_img = torch.utils.data.DataLoader(unknown_img, batch_size = 1)
    model.eval()
    original_labels = []
    pred_lst = []
    for i, inputs in enumerate(loaded_img):
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        prob = nn.functional.softmax(outputs, 1)

        pred_lst.extend(preds)
        imshow(inputs.cpu().data[0])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[preds[0]], 100 * np.max(prob.cpu().detach().numpy()[0]))
        )


def main():
    # Image to be classified
    pred_dir = filedirinput(s=sys.argv)
    # Model input
    model = torch.load("./tlmodelv3.1")
    model.eval()
    logoclassifier(model, loc = pred_dir)

if __name__ == "__main__":
    main()

    """
    Logo classifier usage:
    python logoclassifierV3.0.py file address
    For example, python logoclassifierV3.0.py ./unkown
    """