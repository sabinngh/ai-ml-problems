import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from PIL import Image
import torch 
import torchvision 
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim 
from tqdm import tqdm

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

classes = sorted(train_df['label'].unique())
class_to_idx = {c:i for i,c in enumerate(classes)}
idx_to_class = {i:c for c,i in class_to_idx.items()}

tr_df,val_df  = train_test_split(train_df, test_size =0.2, random_state = 42)

BATCH_SIZE = 16
EPOCHS = 12
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATAS = '.'

train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
          std=[0.229,0.224,0.225])
])

test_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
          std=[0.229,0.224,0.225])
])


class date_imagini(Dataset): 
    def __init__(self,df,root_dir,test = False, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.X_path = df['image_path'].values
        if not test: 
            self.y = df['label'].values
        else: 
            self.y = None

    def __len__(self):
        return len(self.X_path)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.root_dir, self.X_path[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if self.test:
            return image, self.X_path[idx]
        else: 
            label = class_to_idx[self.y[idx]]
            return image, torch.tensor(label, dtype = torch.long)
    

train_dts = date_imagini(tr_df, DATAS, test=False, transform=train_tfms)
val_dts = date_imagini(val_df, DATAS, test=False, transform=test_tfms)
test_dts = date_imagini(test_df, DATAS, test=True, transform=test_tfms)

train_loader = DataLoader(train_dts, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dts, batch_size = BATCH_SIZE, shuffle = False)
test_loader = DataLoader(test_dts, batch_size = BATCH_SIZE, shuffle = False)

class WordCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 20)
    
    def forward(self,x):
        return self.model(x)
    

model = WordCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)

def evaluate_model(model,loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            preds = torch.argmax(outputs, dim = 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return accuracy_score(all_labels, all_preds)

best_accuracy = 0

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0

    for images,labels in tqdm(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    val_accuracy = evaluate_model(model, val_loader)

    print(f'Epoch {epoch+1} / {EPOCHS}')
    print(f'Loss: {total_loss:.4f} | Accuracy: {val_accuracy:.4f}')

    if val_accuracy > best_accuracy: 
        best_accuracy = val_accuracy
    

model.eval()
preds = []
paths = []

with torch.no_grad():
    for images, img_paths in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds_batch = torch.argmax(outputs, dim=1).cpu().numpy()
        preds_batch = [idx_to_class[int(p)] for p in preds_batch]

        preds.extend(preds_batch)
        paths.extend(img_paths)


submission = pd.DataFrame({
    'image_path': paths,
    'label': preds
})

submission.to_csv('submission_2_mar28th.csv', index = False)
