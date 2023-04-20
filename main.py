import os

import pandas as pd
import numpy as np
from collections import defaultdict

import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader

from sklearn import metrics

from src.engine import Engine
from src.data import BirdSpecies
from src.utils import find_classes, Checkpoint


def create_model(num_classes):
    # Load the ResNet50 model pre-trained on ImageNet
    model = torchvision.models.resnet50(pretrained=True)

    # Replace the fully connected layer with a custom one
    model.fc = nn.Sequential(
        nn.Linear(2048, 100),  
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),  # to mitigate overfitting
        nn.Linear(
            100, num_classes
        ), 
    )

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True
    )
    parser.add_argument(
        "--device", type=str,
    )
    parser.add_argument(
        "--epochs", type=int, default=1
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    args = parser.parse_args()

    #!kaggle datasets download -d arjunbasandrai/25-indian-bird-species-with-226k-images
    
    Root_dir = args.data_path
    Dir_train = f'{Root_dir}\\train'
    Dir_valid = f'{Root_dir}\\val'
    
    # uncomment the below line if using the BirdSpecies_m dataset from data.py
    #classes, class_to_int, int_to_class = find_classes(Dir_train)
    model = create_model(num_classes=25)
    model.to(args.device)


    train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.Resize((224, 224)),
            #T.CenterCrop(256),
            T.ToTensor(),
            ])
    
    
    valid_transforms= T.Compose([
            T.Resize((224, 224)),
            #T.CenterCrop(256),
            T.ToTensor(),
        ])
    
    train_dataset = BirdSpecies(Dir_train, train_transforms)
    valid_dataset = BirdSpecies(Dir_valid, valid_transforms)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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

    # final predictions on test set
    Dir_test = f'{Root_dir}\\test'
    classes, class_to_int, int_to_class = find_classes(Dir_test)

    test_transforms= T.Compose([
            T.Resize((224, 224)),
            #T.CenterCrop(256),
            T.ToTensor(),
        ])

    test_dataset = BirdSpecies(Dir_test, test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        )
    
    checkpoint.load('weights\\ResNet_1.pt')
    eng = Engine(model, criterion, optimizer, device=args.device)
    images, true_labels, pred_labels = eng.get_prediction(test_dataloader)


    # Initialize a dictionary to keep track of correct predictions per class
    correct_preds = defaultdict(int)

    for i, pred in enumerate(pred_labels):
            if pred == true_labels[i]:
                correct_preds[classes[true_labels[i].item()]] += 1
                
    # Print the number of correct predictions per class
    for class_name, num_correct in correct_preds.items():
        print(f"Class {class_name}: {num_correct} correct predictions")





    