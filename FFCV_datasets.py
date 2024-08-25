import argparse
import logging
from pathlib import Path

import torchvision.transforms as transforms

from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter, MIN_PAGE_SIZE

from Dataloaders import datasets

log = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tool for managing FFCV datasets"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="print log messages from this level up; debug, info, warning, error, or critical",
    )
    command_parsers = parser.add_subparsers(dest="command")

    create_parser = command_parsers.add_parser("create", help="create datasets")
    create_parser.add_argument(
        "--dataset-path",
        type=str,
        metavar="PATH",
        default=None,
        help="path to the directory containing dataset",
    )
    create_parser.add_argument(
        "--dataset-name",
        type=str,
        metavar="NAME",
        default=None,
        help="name of the original dataset. should be implemented in DATALOADERS",
    )
    create_parser.add_argument(
        "--dataset-splits",
        type=str,
        metavar="SPLITS",
        nargs="+",
        default=None,
        help="list of splits of the dataset. e.g 'train' 'val' 'test' ",
    )
    create_parser.add_argument(
        "--ffcv-directory",
        type=str,
        metavar="PATH",
        default=None,
        help="path to the directory for saving the new dataset",
    )
    create_parser.add_argument(
        "--workers",
        type=int,
        metavar="INT",
        default=8,
        help="number of workers",
    )

    return parser.parse_args()

def create_ffcv_dataset(args: argparse.Namespace):
    dataset_class = datasets.DatasetGetter[args.dataset_name].value
    dataset_dicts = {
        split: dataset_class(
            root = args.dataset_path,
            split = split,
            transform= transforms.Compose([transforms.PILToTensor()])
        ) 
        for split in args.dataset_splits
        }
    for (name, ds) in dataset_dicts.items():
        writer = DatasetWriter(Path(args.ffcv_directory) / f'{name}.beton', {
            'image': RGBImageField(write_mode="smart",
                                    max_resolution=500,
                                    jpeg_quality=90,
                                    compress_probability=0.5),
            'label': IntField()
        }, num_workers=args.workers)
        writer.from_indexed_dataset(ds, chunksize=100)
        log.info(f"Dataset written to {args.ffcv_directory}")

if __name__ == "__main__":
    args = parse_args()

    log_level = getattr(logging, args.log_level.upper(), "INFO")
    logging.basicConfig(level=log_level)

    if args.command == 'create':
        create_ffcv_dataset(args)