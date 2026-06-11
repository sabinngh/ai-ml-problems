import os
import cv2
import numpy as np
import pandas as pd

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.0003
NUM_CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("train.csv")

label2idx = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
df['label'] = df['label'].map(label2idx)


test_df = pd.read_csv("test.csv")
train_df,val_df = train_test_split(df, test_size = 0.2, random_state = 42)

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class ChessDataset(Dataset):

    def __init__(self,dataframe,img_dir="images",transform=None):
        self.df = dataframe.reset_index(drop = True)
        self.img_dir = img_dir 
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    #Functia care atribuie efectiv fiecarei poze label ul ei 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join("images", row['image_path'])
        

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'Nu pot citi imaginea {img_path}')
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = row['label']
        label = torch.tensor(label, dtype=torch.long)

        return img, label
    
train_dataset = ChessDataset(train_df, img_dir = "images", transform = transform_train)
val_dataset = ChessDataset(val_df, img_dir = "images", transform = transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features,NUM_CLASSES)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")



idx2label = {v: k for k, v in label2idx.items()}

class ChessTestDataset(Dataset):
    def __init__(self, dataframe, img_dir="images", transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'])

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Nu pot citi imaginea {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, row['id']

test_dataset = ChessTestDataset(
    test_df,
    img_dir="images",
    transform=transform_val
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model.eval()

all_ids = []
all_preds = []

with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        preds = preds.cpu().numpy()

        for i in range(len(preds)):
            all_ids.append(ids[i])
            all_preds.append(idx2label[preds[i]])

submission = pd.DataFrame({
    "id": all_ids,
    "label": all_preds
})

submission.to_csv("submission_1_feb19th26.csv", index=False)

print("submission.csv generat cu succes!")