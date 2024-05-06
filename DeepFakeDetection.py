import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

#----------------------------------------------------------------------------------
#                           Set Paths and Variables
#----------------------------------------------------------------------------------
device = torch.device("cpu")
AllData = ''
paths = {'Training': AllData + 'Training/', 'Validation': AllData + 'Validation/', 'Testing': AllData + 'Testing/'}
#----------------------------------------------------------------------------------
#               Transforming Settings For Generalizing Better
#----------------------------------------------------------------------------------
transformations = {
    'Training': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'Validation': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'Testing': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
datasets = {
    x: datasets.ImageFolder(paths[x], transform=transformations[x])
    for x in ['Training', 'Validation', 'Testing']
}

dataload = {
    x: DataLoader(datasets[x], batch_size=32, shuffle=(x == 'Training'), num_workers=0)
    for x in ['Training', 'Validation', 'Testing']
}
#----------------------------------------------------------------------------------
#                           Define Model Structure ResNet-18
#----------------------------------------------------------------------------------

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, 2))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#----------------------------------------------------------------------------------
#                                    Train Model
#----------------------------------------------------------------------------------

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    x = len(datasets['Training'])
    y = len(datasets['Validation'])
    
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataload['Training']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            predictions = outputs.argmax(dim=1)
            training_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predictions == labels.data)
        scheduler.step()
        epoch_loss = training_loss / x
        epoch_acc = running_corrects.double() / x
        print(f'Epoch {epoch+1}/{num_epochs} Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#----------------------------------------------------------------------------------
#                                Validate Model
#----------------------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in dataload['Validation']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(predictions == labels.data)
        val_loss /= y
        val_acc = val_corrects.double() / y
        print(f'Epoch {epoch+1}/{num_epochs} Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

#----------------------------------------------------------------------------------
#                        Test Data Set on Unseen Data
#----------------------------------------------------------------------------------
def evaluate_model(model, test_loader):
    model.eval()
    num_correct = 0
    x = len(datasets['Testing'])
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            num_correct += torch.sum(predictions == labels.data)
    
    test_acc = num_correct.double() / x
    print(f'Test Accuracy: {test_acc:.4f}')
#----------------------------------------------------------------------------------
#                         Run Everything
#----------------------------------------------------------------------------------

train_model(model, criterion, optimizer, scheduler, num_epochs=10)
evaluate_model(model, dataload['Testing'])

