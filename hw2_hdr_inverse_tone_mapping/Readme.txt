# How to execute the code

0. Prepare data path:
- Training data and validation data are stored at the same location with source: ./data
- HDR results are stored at ./data/val/hdr_res

1. Train:
$ python Main --mode train

2. Validate and generate HDR images:
$ python Main --mode val
