from pathlib import Path
import torch
import torchvision.transforms as transforms

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
