import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import numpy as np
from enum import Enum

class ImageNetDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        self.root = root
        self.samples_dir = None

        self._load_jsons()
        
        if split == "val":
            self.samples_dir = os.path.join(self.root, "Data/CLS-LOC/val")
            for entry in os.listdir(self.samples_dir):
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(self.samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
        
        elif split == "train":
            self.samples_dir = os.path.join(self.root, "Data/CLS-LOC/train")
            for entry in os.listdir(self.samples_dir):
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(self.samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
        else:
            raise ValueError("Inserted wrong split, only 'train' and 'val' allowed")
    
    def _load_jsons(self):
        with open(os.path.join(self.root, "ILSVRC2012_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(self.root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return np.transpose(np.array(x, dtype=np.uint8), (1,2,0)), self.targets[idx]


class ImageDepthDataset(ImageNetDataset):
    def _load_jsons(self):
        with open(os.path.join(self.root, "ILSVRC2012_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(self.root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        self.val_to_syn = {key.replace(".JPEG","-dpt_swin2_large_384.png") : val for key,val in self.val_to_syn.items()}
    
    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert('L')
        x = Image.fromarray(np.stack((x,x,x),axis=2)).convert('RGB')
        if self.transform:
            x = self.transform(x)
        return np.transpose(np.array(x,dtype=np.uint8), (1,2,0)), self.targets[idx]

class DatasetGetter(Enum):
    ImageNetDataset = ImageNetDataset
    ImageDepthDataset = ImageDepthDataset