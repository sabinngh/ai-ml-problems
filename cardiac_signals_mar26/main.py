import numpy as np 
import pandas as pd
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

df_tr, df_val = train_test_split(train_df, test_size = 0.2)

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class data_model_imagini(Dataset):
    def __init__(self, df, test = False):
        self.test = test
        self.X_path = df['image_path'].values
        if self.test == False:
            self.y = df['label'].values
    
    def __len__(self):
        return len(self.X_path)
    
    def __getitem__(self, index):
        image = torchvision.io.read_image(self.X_path[index])
        
        if image.shape[0] == 4:
            image = image[:3]

        image = image.float() / 255.0   

        image = transform(image)

        if not self.test:
            label = torch.tensor(self.y[index]).long()
            return image, label
        else:
            return image
        

train_dataset= data_model_imagini(df_tr)
val_dataset = data_model_imagini(df_val)
test_dataset = data_model_imagini(test_df, test= True)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset,batch_size = 16)

import torch.nn as nn
import torchvision.models as models

class model_class_imagini(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = models.resnet50(pretrained=True)
        
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model_class_imagini().to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 2

for epoch in range(epochs):
    model.train()
    train_preds = []
    train_labels = []

    for X, y in tqdm(train_loader):
        X = X.to(device)
        y = y.to(device)

        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        preds = torch.argmax(y_hat, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(y.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)

    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for X, y in tqdm(val_loader):
            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)
            preds = torch.argmax(y_hat, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)

    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

model.eval()
preds = []

with torch.no_grad():
    for X in tqdm(test_loader):
        X = X.to(device)

        y_hat = model(X)
        pred = torch.argmax(y_hat, dim=1)

        preds.extend(pred.cpu().numpy())

pd.DataFrame({
    "image_path": test_df['image_path'],
    "label": preds
}).to_csv("submission_1_mar19th.csv", index=False)

