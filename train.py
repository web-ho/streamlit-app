import os

import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from PIL import Image

from src.engine import Engine
from src.utils import find_classes, Checkpoint


def create_model(num_classes):
    # Load the ResNet50 model pre-trained on ImageNet
    model = torchvision.models.resnet50(pretrained=True)

    # Replace the fully connected layer with a custom one
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),  
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),  # to mitigate overfitting
        nn.Linear(
            1024, num_classes
        ), 
    )
    return model

# modified loader to load images in grayscale, default is RGB
def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
        return img 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True
    )
    parser.add_argument(
        "--device", type=str, default='cpu'
    )
    parser.add_argument(
        "--epochs", type=int, default=1
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    args = parser.parse_args()

    
    Root_dir = args.data_path
    Dir_train = f'{Root_dir}\\train'
    Dir_valid = f'{Root_dir}\\val'
    
    # uncomment the below line if using the BirdSpecies_m dataset from data.py
    #classes, class_to_int, int_to_class = find_classes(Dir_train)
    model = create_model(num_classes=25)
    model.to(args.device)


    train_transforms = T.Compose([
        T.Resize((224, 224)),
        T.RandomRotation(45),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        #T.CenterCrop(256),
        T.ToTensor(),
    ])
    
    
    valid_transforms= T.Compose([
            T.Resize((224, 224)),
            #T.CenterCrop(256),
            T.ToTensor(),
        ])
    
    train_dataset = ImageFolder(Dir_train, train_transforms, loader=pil_loader)
    valid_dataset = ImageFolder(Dir_valid, valid_transforms, loader=pil_loader)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.6)
    eng = Engine(model, criterion, optimizer, device=args.device)

    for epoch in range(args.epochs):
        train_loss = eng.train(train_dataloader)
        valid_loss, valid_acc = eng.evaluate(valid_dataloader)
        print(f'Epoch : {epoch}, Train_loss : {train_loss}, Valid_loss : {valid_loss}')

    checkpoint = Checkpoint(model, optimizer)
    model_name = f"{model.__class__.__name__}_{args.epochs}.pt"
    model_path = os.path.join('weights', model_name)
    checkpoint.save(model_path)

    


    