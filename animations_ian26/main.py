#Importuri

import json
import cv2 
import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from torchvision import models,transforms

from sklearn.model_selection import train_test_split

#Transform nume etichete/clase in numere

CLASS_TO_IDX = {
    "Pokemon" : 0,
    "Tarzan" : 1,
    "Snow White" : 2,
    "Winnie the Pooh" : 3
}

IDX_TO_CLASS = {value : key for key,value in CLASS_TO_IDX.items()}

#Citire json 

with open("train.json","r") as f:
    train_data = json.load(f)

train_df = pd.DataFrame(train_data)
train_df['label'] = train_df['cartoon_class'].map(CLASS_TO_IDX)

#train test split

train_df, val_df = train_test_split(train_df,
                                    test_size = 0.2,
                                    stratify=train_df['label'],
                                    random_state = 42)

#transformari imagine
#fata de alte probleme, aici imaginile sunt colorate si trebuie
#normalizate putin si resize +tensor ca sa le lucreze pytorch bine

train_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

val_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485,0.456,0.406],
        std = [0.229,0.224,0.225]
    )
])

class CartoonDataset(Dataset):
    def __init__(self,df,tfms):
        self.df = df.reset_index(drop = True)
        self.tfms = tfms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_path"])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = self.tfms(img)
        label = torch.tensor(row["label"], dtype = torch.long)

        return img,label
    
#DataLoader

train_loader = DataLoader(
    CartoonDataset(train_df, train_tfms),
    batch_size = 16,
    shuffle = True
)

val_loader = DataLoader(
    CartoonDataset(val_df,val_tfms),
    batch_size = 16,
    shuffle = False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.resnet18(weights = "IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features,4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4)

EPOCHS = 15

for epoch in range(EPOCHS):
    model.train()
    for imgs,labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Validation Accuracy: {val_acc:.4f}")


with open("test.json") as f:
    test_data = json.load(f)

test_df = pd.DataFrame(test_data)

class TestDataset(Dataset):
    def __init__(self,df,tfms):
        self.df = df.reset_index(drop = True)
        self.tfms = tfms
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        img = cv2.imread(self.df.iloc[idx]["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfms(img)
        return img
    
test_loader = DataLoader(
    TestDataset(test_df,val_tfms),
    batch_size=16,
    shuffle = False
)    

model.eval()
predictions = []

with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        predictions.extend(preds.tolist())

submission = pd.DataFrame({
    "image_path": test_df["image_path"],
    "cartoon_class": [IDX_TO_CLASS[p] for p in predictions]
})

submission.to_csv("submission_2_ian25th.csv", index=False)

