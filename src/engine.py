from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T



class Engine:
    def __init__(
            self,
            model,
            criterion,
            optimizer,
            scheduler=None,
            model_fn=None,
            device=None,
    ):
        """
        model_fn should take batch of data, device and model and return loss
        for example:
            def model_fn(data, device, model):
                images, targets = data
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                _, loss = model(images, targets)
                return loss
        """
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_fn = model_fn

    def train(self, dataloader):
    
        self.model.train()
        epoch_loss = 0  # initialize epoch loss
        
        # iterate through the dataloader batches. tqdm keeps track of progress.
        for batch_n, batch in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):

            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(image)

            loss = self.criterion(outputs, label)
            epoch_loss += loss.item()  # accumulate epoch loss

            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

        train_loss = epoch_loss / len(dataloader)  # calculate average epoch loss
        
        return train_loss

    def evaluate(self, dataloader):

        self.model.eval()

        epoch_loss = 0
        epoch_acc = 0

        with torch.no_grad():

            for batch in tqdm(dataloader, total=len(dataloader)):

                image = batch[0]
                label = batch[1]

                image = image.to(self.device)
                label = label.to(self.device)
                
                outputs = self.model(image)
                loss = self.criterion(outputs, label)

                probs = nn.functional.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                acc = (preds == label).float().mean()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            valid_loss = epoch_loss / len(dataloader)
            valid_acc = epoch_acc / len(dataloader)
            
        return valid_loss, valid_acc

    def get_prediction(self, dataloader):

        self.model.eval()

        preds_collector = []
        true_labels = []
        images_collector = []

        with torch.no_grad():

            for batch in tqdm(dataloader, total=len(dataloader)):

                image = batch[0]
                label = batch[1]

                image = image.to(self.device)
                label = label.to(self.device)
                
                outputs = self.model(image)
                loss = self.criterion(outputs, label)

                probs = nn.functional.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                preds_collector.append(preds.cpu().numpy())
                true_labels.append(label.cpu().numpy())
                images_collector.append(image.cpu().numpy())

        true_labels = np.concatenate(true_labels)
        pred_labels = np.concatenate(preds_collector)
        images = np.concatenate(images_collector)
            
        return images, true_labels, pred_labels
    
    
# funtion to use in the app.py 
def predict_img(img_path, model, int_to_class):

    labels = {int(k): v for k, v in int_to_class.items()}
    # Load the image and apply transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)

    # Make prediction using the model
    model.eval()
    with torch.no_grad():
        answer = model(img)
        y_class = answer.argmax(dim=-1).cpu().numpy()
        prob = torch.softmax(answer, dim=-1)[0][y_class[0]].item()
        perc = prob * 100

    # Convert the prediction result to label
    res = labels[y_class[0]]
    
    #print(f'Image is of: {res}, with {perc:.3f} probability')

    return res