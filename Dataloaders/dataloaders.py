from pathlib import Path
import torch
import torchvision.transforms as transforms

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder

from .utils import fast_collate, PrefetchedWrapper

def get_dataloader(dataset, data_root, split, batch_size, workers=5, _worker_init_fn=None, input_size=224):
    if split == "train":
        transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            ])
    elif split == "val":
        transforms = transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            ])
    else:
        raise ValueError(f"Split {split} has to be either 'train' or 'val'")

    _dataset = dataset(root=data_root,
                            split=split, 
                            transform=transforms)

    if torch.distributed.is_initialized():
        _sampler = torch.utils.data.distributed.DistributedSampler(_dataset)
    else:
        _sampler = None

    _loader = torch.utils.data.DataLoader(
            _dataset,
            batch_size=batch_size,
            shuffle=(_sampler is None),
            num_workers=workers,
            worker_init_fn=_worker_init_fn,
            pin_memory=True,
            sampler=_sampler,
            collate_fn=fast_collate)

    return PrefetchedWrapper(_loader), len(_loader)

def get_ffcv_dataloader(data_root:Path,
                        split:str,
                        device:torch.device,
                        batch_size:int, 
                        workers:int=5,
                        input_size:int=224
                        ):

    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device)]

    if split == "train":
        image_pipeline = [
            RandomResizedCropRGBImageDecoder((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            ]

    elif split == "val":
        image_pipeline = [
            CenterCropRGBImageDecoder((input_size,input_size),1),
            ]
    else:
        raise ValueError(f"Split {split} has to be either 'train' or 'val'")
    
    image_pipeline.extend([
        ToTensor(),
        ToTorchImage(),
        ToDevice(device, non_blocking=True)
    ])
    

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
    _loader = Loader(data_root, batch_size=batch_size, num_workers=workers,
                    order=OrderOption.RANDOM, pipelines=pipelines)
    
    return PrefetchedWrapper(_loader), len(_loader)