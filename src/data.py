import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.utils import find_classes



class BirdSpecies(ImageFolder):
    
    def __init__(self, root, transforms=None):
        super().__init__(root, transforms)
        
        # Only keep images with 3 channels
        self.samples = [(path, target) for path, target in self.samples if self._is_valid_file(path)]
        
    def _is_valid_file(self, path):
        try:
            with Image.open(path) as img:
                return img.mode == 'RGB'
        except Exception:
            return False
        


## this code (works) but slowed down the training process significantly as it is running too many checks
class BirdSpecies_m(Dataset):
    
    def __init__(self, image_dir, transforms=None):

        self.image_dir = image_dir
        self.transforms = transforms

        # Find classes and map them to integer labels
        self.classes, self.class_to_int = find_classes(self.image_dir)

      # this function creates a list of (image path, class label) tuples
      # source - https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder  
    def make_dataset(self):
        instances = []
        for target_class in sorted(self.class_to_int.keys()):
            class_index = self.class_to_int[target_class]
            target_dir = os.path.join(self.image_dir, target_class)

            # Traverse the directory structure and get image paths
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    try:
                        image = Image.open(path)
                        channels = len(image.getbands())
                        if channels == 3:
                            item = path, class_index
                            instances.append(item)
                    except Exception as e:
                        print('There is some problem with the code: {}'.format(e))

        return instances

    
      # Get an image and its label at a given index
    def __getitem__(self, idx):
        images = []
        image_path, target = self.make_dataset()[idx]
        try:
            image = Image.open(image_path)
            channels = len(image.getbands())
            if channels == 3:
                images.append(image)
            else:
                images.append(torch.zeros((3, 224, 224)))  # return empty tensor with correct size
        except Exception as e:
            print('There is some problem with the code: {}'.format(e))
            images.append(torch.zeros((3, 224, 224)))  # return empty tensor with correct size

        # Apply transformations to the image
        if self.transforms is not None and len(images)>0:
            images = self.transforms(images[0])
        else:
            return torch.zeros((3, 224, 224)), target  # return empty tensor with correct size

        return images, target


    # Return the number of images in the dataset
    def __len__(self):
        return len(self.make_dataset())