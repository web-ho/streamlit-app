import os
from collections import defaultdict

import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from PIL import Image

from src.engine import Engine
from src.utils import find_classes



# load from the path
model_path = os.path.abspath("weights/Quantized_ResNet_21.pt")
model = torch.load(model_path)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

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
        "--batch_size", type=int, default=16
    )
    args = parser.parse_args()


    # final predictions on test set
    Root_dir = args.data_path
    Dir_test = f'{Root_dir}\\test'
    classes, class_to_int, int_to_class = find_classes(Dir_test)

    test_transforms= T.Compose([
            T.Resize((224, 224)),
            #T.CenterCrop(256),
            T.ToTensor(),
        ])

    test_dataset = ImageFolder(Dir_test, test_transforms, loader=pil_loader)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        )
    
    #checkpoint = Checkpoint(model, optimizer)
    #checkpoint.load('weights\\ResNet_21.pt')
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