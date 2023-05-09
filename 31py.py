# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:36:18 2023

@author: yagiz
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:40:07 2023

@author: yagiz
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:28:26 2023

@author: yagiz
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:53:03 2023

@author: yagiz
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ToPILImage, Resize
from PIL import Image
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import imageio
import numpy as np
from scipy.ndimage import zoom
from torchvision.transforms import Lambda
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

#%%
class RankIQASiameseNetwork(nn.Module):
    def __init__(self):
        super(RankIQASiameseNetwork, self).__init__()

        # Feature extraction network
        self.feature_extractor = models.vgg16(pretrained=True).features

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    

# Create the Siamese network
rank_iqa_siamese_network = RankIQASiameseNetwork().cuda()
print(rank_iqa_siamese_network)

class DividedTID2013RankDataset(Dataset):
    def __init__(self, image_score_pairs, divisions=4, transform=None):
        self.image_score_pairs = image_score_pairs
        self.divisions = divisions
        self.transform = transform

        self.image_paths = []
        self.scores = []
        for i in range(len(self.image_score_pairs)):
            img_path, score = self.image_score_pairs[i]
            self.image_paths.append(img_path)
            self.scores.append(float(score))

    def __len__(self):
        return len(self.image_paths) * self.divisions

    def to_tensor(img):
        img = np.array(img, dtype=np.float32)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        return img

    def resize_image31(img, size):
        factors = (size[0] / img.shape[0], size[1] / img.shape[1], 1)
        img_resized = zoom(img, factors, order=1)
        return img_resized

    def divide_image(self, image):
        h, w, _ = image.shape
        h_div = h // 2
        w_div = w // 2
        return [
            image[:h_div, :w_div, :],
            image[:h_div, w_div:, :],
            image[h_div:, :w_div, :],
            image[h_div:, w_div:, :],
        ]

    def __getitem__(self, idx):
        original_idx = idx // self.divisions
        division_idx = idx % self.divisions

        img_path = self.image_paths[original_idx]
        score = self.scores[original_idx]
        img = imageio.imread(img_path)

        if len(img.shape) == 2:  # Check if the image is grayscale
            img = np.stack((img,) * 3, axis=-1)  # Stack grayscale image into 3 channels

        divided_images = self.divide_image(img)
        divided_img = divided_images[division_idx]

        if self.transform:
            divided_img = self.transform(divided_img)

        return {'img': divided_img, 'score': np.float32(score)}

# Custom transforms
transform = transforms.Compose([
    transforms.Lambda(lambda img: DividedTID2013RankDataset.resize_image31(img, (224, 224))),
    transforms.Lambda(DividedTID2013RankDataset.to_tensor),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



tid2013_folder = 'dataset' # Replace this with the actual path to your TID2013 dataset folder
rank_list_file = 'tid2013_train.txt' # Replace this with the path to your TID2013 rank list file

# Define normalization parameters for VGG-16
# normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Define the data transformations
# transformations = Compose([Resize((224, 224)), ToTensor(), normalize])
# 
# tid2013_rank_dataset = TID2013RankDataset(tid2013_folder, rank_list_file, transform=transformations)
# dataloader = DataLoader(tid2013_rank_dataset, batch_size=32, shuffle=True, num_workers=0)
    
#%% test train etc

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import random_split



def lossl1v2(reel, predict):
    reel2 = reel.squeeze().numpy()
    predict2 = predict.squeeze().numpy()
    diff = abs(reel2-predict2)
    return diff.mean()

def lossl1(reel, predict):
    euclidean_distance = torch.abs(reel - predict)
    loss = torch.sum(euclidean_distance)
    return loss

def lossmape(reel,predict):
    euclidean_distance = torch.abs(reel - predict)
    loss = torch.sum(euclidean_distance/reel)
    return loss

def train(model, dataloader, optimizer, device):
    global global_step_train
    model.train()
    running_loss = 0.0
    i = 0
    for batch in dataloader:
        i+=1
        global_step_train += 1
        images = batch['img'].to(device)
        scores = batch['score'].to(device).view(-1, 1)  # Updated line
        # print("144")
        optimizer.zero_grad()
        outputs = model(images)
        # print(scores)
        # print(outputs)
        loss = lossmape(outputs, scores)
        losslone = lossl1(outputs, scores)
        print("epoch: " + str(epoch) +" batch: " +str(i) + " mape = " +str(loss.item()/32) + " l1loss = "+ str(losslone.item()/32))
        losslone.backward()
        optimizer.step()
        writer.add_scalar('Lossl1/train', losslone.item()/32, global_step_train)
        writer.add_scalar('Lossmape/train', loss.item()/32, global_step_train)
        # validation_loss = validate(model, val_loader, device)
        # print("valloss:" + str(validation_loss))
        # running_loss += loss.item()
    return running_loss / len(train_dataset)


def validate(model, dataloader, device):
    global global_step_val
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            global_step_val +=1
            images = batch['img'].to(device)
            scores = batch['score'].to(device).view(-1, 1)  # Updated line
            # print("162")
            outputs = model(images)
            loss = lossl1(outputs, scores)
            loss = lossmape(outputs, scores)
            # print(loss.item()/32)
            writer.add_scalar('Lossl1/val', loss.item()/32, global_step_val)
            writer.add_scalar('Lossmape/val', lossmape.item()/32, global_step_val)
            running_loss += loss.item()
    return running_loss / len(test_dataset)

def validate2(model, dataloader, device):
    model.eval()
    # running_loss = 0.0
    reel =  []
    out = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['img'].to(device)
            scores = batch['score'].to(device).view(-1, 1)  # Updated line
            print("162")
            outputs = model(images)
            reel.append(scores.squeeze().numpy())
            out.append(outputs.squeeze().numpy())
    return reel, out

def validate3(model, dataloader, device):
    model.eval()
    # running_loss = 0.0
    reel =  []
    out = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['img'].to(device)
            scores = batch['score'].to(device).view(-1, 1)  # Updated line
            print("162")
            outputs = model(images)
            reel.append(scores.squeeze().numpy())
            out.append(outputs.squeeze().numpy())
    df_a = pd.DataFrame(reel)
    df_b = pd.DataFrame(out)
    a1= df_a.to_numpy().flatten()
    b1= df_b.to_numpy().flatten()
    df_a1 = pd.DataFrame(a1)
    df_b1 = pd.DataFrame(b1)
    reellist =[]
    tot = 0
    for i in range (0, len(df_a1)):
        if i%4!=3:
            tot += df_a1[0][i]
        else: 
            tot += df_a1[0][i]
            reellist.append(tot/4)
            tot=0
            
    predlist =[]
    tot = 0
    for i in range (0, len(df_b1)):
        if i%4!=3:
            tot += df_b1[0][i]
        else: 
            tot += df_b1[0][i]
            predlist.append(tot/4)
            tot=0
    reelito = torch.tensor(reellist)
    predito = torch.tensor(predlist)
    
    l1loss = lossl1v2(reelito, predito)
    mapeloss = lossmape(reelito, predito)
    
    return l1loss.item(), mapeloss.item()
    
    
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model and send it to the device
model = RankIQASiameseNetwork().to(device)

# Load the dataset
tid2013_folder = 'dataset'
rank_list_train_file = 'microscopy_train.txt'
rank_list_val_file = 'microscopy_val.txt'
rank_list_test_file = 'microscopy_test.txt'

print("180")
transform = transforms.Compose([
    Lambda(lambda img: DividedTID2013RankDataset.resize_image31(img, (224, 224))),
    Lambda(DividedTID2013RankDataset.to_tensor),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Read the entire dataset and split it into train and test sets
with open(rank_list_train_file, 'r') as f:
    train_image_score_pairs = [line.strip().split(';') for line in f.readlines()]
    
with open(rank_list_val_file, 'r') as f:
    val_image_score_pairs = [line.strip().split(';') for line in f.readlines()]
    
with open(rank_list_test_file, 'r') as f:
    test_image_score_pairs = [line.strip().split(';') for line in f.readlines()]

# train_image_score_pairs, test_image_score_pairs = train_test_split(image_score_pairs, test_size=0.2, random_state=42)

print("187")
train_dataset = DividedTID2013RankDataset(train_image_score_pairs, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = DividedTID2013RankDataset(val_image_score_pairs, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

test_dataset = DividedTID2013RankDataset(test_image_score_pairs, transform=transform)  # New test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # New test loader

# Loss function and optimizer

# criterion = nn.L1Loss()
# criterion = lossl1()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
best_val_loss = float('inf')

writer = SummaryWriter('runs/experiment_1')
global_step_train = 0
global_step_val = 0

global_epoch =0

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    print("198")
    global_epoch +=1
    train_loss = train(model, train_loader, optimizer, device)
    writer.add_scalar('Lossl1/trainepoch', train_loss, global_epoch)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')
    # Validation
    validation_loss = validate(model, val_loader, device)
    writer.add_scalar('Lossl1/valepoch', validation_loss, global_epoch)
    
    vall1, valmape = validate3(model, val_loader, device)
    print(f'Validation Loss: {validation_loss:.4f}')
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        torch.save(model.state_dict(), 'best_model15epoch33.pth')
        print("Model saved.")
        

model_final = torch.load('best_model15epoch.pth')        
test_loss = validate(model, test_loader, device)
print(f'Test Loss: {test_loss:.4f}')
testl1, testmape = validate3(model, test_loader, device)
print(testl1, testmape)
reel, out =validate2(model, test_loader, device)
print(reel[1])


