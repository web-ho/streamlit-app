import os
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision



class Checkpoint:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.epoch = 0
        self.best_loss = 0

    def save(self, filename):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_loss': self.best_loss
        }
        torch.save(state, filename)
        
    def load(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.epoch = state['epoch']
        self.best_loss = state['best_loss']


def seed_everything(seed, use_cuda = True):
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    random.seed(seed) # Python
    os.environ['PYTHONHASHSEED'] = str(seed) # Python hash building
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False



# Make function to find classes in target directory (pytorch method)
def find_classes(directory: str):
    """Finds the class folder names in a target directory.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    """
    # Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # Create a dict of index labels
    class_to_int = {cls_name: i for i, cls_name in enumerate(classes)}
    
    int_to_class = {v: k for k, v in class_to_int.items()}

    return classes, class_to_int, int_to_class


def check_file_type(image_folder):
    extension_type = []
    for _class in os.listdir(image_folder):
        if os.path.isdir(os.path.join(image_folder, _class)):
            # if the class is a directory, iterate over its contents
            for filename in os.listdir(os.path.join(image_folder, _class)):
                extension_type.append(filename.rsplit(".", 1)[1].lower())
        else:
            extension_type.append(_class.rsplit(".", 1)[1].lower())
        return extension_type
        


def check_image_size(image_folder):
    counter = 0
    for _class in os.listdir(image_folder):
        if os.path.isdir(os.path.join(image_folder, _class)):
            # if the class is a directory, iterate over its contents
            for image in os.listdir(os.path.join(image_folder, _class)):
                try:
                    img = Image.open(os.path.join(image_folder, _class, image))
                    width, height = img.size
                    channels = len(img.getbands())
                    if channels == 3:
                        continue
                    else:
                        print("Image name: {}, Size: {} x {}, Channels: {}".format(image, width, height, channels))
                        #print("Image size: {} x {}, Channels: {}".format(width, height, channels))
                except Exception as e:
                    print("This {} is problematic: {}".format(image, e))
                    counter += 1
    return counter


