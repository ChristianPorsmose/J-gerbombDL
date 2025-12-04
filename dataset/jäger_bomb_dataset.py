from typing import List
from torch.utils.data import Dataset
from torchvision.io import decode_image
import os
from torch import tensor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
empty_labels = torch.zeros((0, 5), dtype=torch.float32)

class JägerBombDataset(Dataset): # type: ignore
    def __init__(
        self,
        images_list: str,

        images_list_ext: str = "txt",

        labels_ext: str = "txt",
        transforms : list = None
        ):

        self.images_list = images_list
        self.images_list_ext = images_list_ext
        self.labels_ext = labels_ext

        self.image_files : List[str] = []
        file = open(images_list, "r")

        images_list_dir = os.path.dirname(images_list)

        i = 0
        for file_name in file.readlines():
            full_name = file_name.strip()
            self.image_files.append(full_name)
            i += 1

        self.label_files : List[str] = []
        self.labels_dir = os.path.join(images_list_dir, "labels")

        for image_path in self.image_files:
            label_path = self.__get_label_path_from_image_path(image_path)
            self.label_files.append(label_path)
            i += 1
        
        self.transforms = transforms
        self.mislabeled=[]

    def __len__(self):
        return len(self.image_files)

    def __get_label_path_from_image_path(self, image_path : str) -> str:
        return os.path.join(
            self.labels_dir,
            os.path.basename(image_path).replace(".jpg", f".{self.labels_ext}", 1),
        )
    
    def __getitem__(self, index : int):
        image_path = self.image_files[index]
        image = decode_image(image_path)
        
        #Check if label file exists
        label_path = self.__get_label_path_from_image_path(image_path)

        if not os.path.exists(label_path):
            # Negative sample (background image with no objects)
            if self.transforms:
                image, _ = self.transforms(image, empty_labels)
            return image.to(device), empty_labels.to(device)
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Invalid label format in file: {label_path}")
                        self.mislabeled.append(image_path)
                        return image.to(device), empty_labels.to(device)
                    
                    class_label, x_center, y_center, width, height = map(float, parts)

                    if not (0 <= class_label <= 1 and 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        print(f"Det er altså ikke ordenlige labels her: {label_path}")
                        self.mislabeled.append(image_path)
                        continue
                    
                    labels.append([class_label, x_center, y_center, width, height])
                except ValueError:
                    print(f"Non-numeric value found in label file: {label_path}")
                    self.mislabeled.append(image_path)
                    continue
            
            if len(labels) == 0:
                #print(f"No valid labels found for image: {image_path}\nAt path: {label_path}")
                labels_tensor = empty_labels
            else:
                labels_tensor = tensor(labels)
            
            # Apply transforms AFTER loading labels so we can transform both
            if self.transforms:
                image, labels_tensor = self.transforms(image, labels_tensor)

            return image.to(device), labels_tensor.to(device)