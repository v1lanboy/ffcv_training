import torch
import numpy as np

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        copy_array = np.copy(nump_array)
        tensor[i] += torch.from_numpy(copy_array)

    return tensor, targets

class PrefetchedWrapper(object):
    def prefetched_loader(loader):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target #This causes runtime error in backprob if device is allocated in FFCV
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)