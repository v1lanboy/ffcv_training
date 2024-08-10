# FFCV_training

# Create FFCV dataset
```console
python FFCV_datasets.py create --dataset-path <path_to_directory> --dataset-name <DatasetClass> --dataset-splits <train/val> --ffcv-directory <path_to_directory> 
```
Issues:
- PAGE_SIZE being too low will give malloc error halt the process in memory allocator
- Default workers in the DatasetWriter is equal to num of cpu cores. For me this crashes the whole system time to time.