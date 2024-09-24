# MSMGE-CNN: A multi-scale multi-graph embedding convolutional neural network for motor related EEG decoding
This is the PyTorch implementation of the MSMGE-CNN architecture for MI/ME-EEG classification.
![](https://github.com/a0304w99/MSMGE-CNN/blob/main/MSMGE-CNN.png)

# Resources
## Datasets
BCICIV-2a：(https://www.bbci.de/competition/iv/)

# Instructions
## Install the dependencies
It is recommended to create a virtual environment with python version 3.7 and running the following:
pip install -r requirements.txt

Obtain the raw dataset:
Download the raw dataset from the resources above, and save them to the same folder. 

braindecode package is directly copied from https://github.com/robintibor/braindecode/tree/master/braindecode for preparing datasets

Start:

setp 1 :Prepare dataset(Only needs to run once)

python tools/data_bciciv2a_tools.py --data_dir ~/dataset/bciciv2a/gdf -output_dir ~/dataset/bciciv2a/pkl

step 2 :Train model

python ho.py -data_dir ~/dataset/bciciv2a/pkl -id 1

# Results
The classification results for our method in three scenarios are as follows:
## BCICIV-2a
| Settings  | FBCSP |  Deep ConvNet | EEGNet | FBCNet | FBMSNet | EEG_GENet | MSMGE-CNN |
| :-------- | :---- | ------------- | ------ | ------ | ------- | --------- | --------- |
| HO | 67.75 | 73.15 | 70.45 | 76.23 | 78.36 | 77.89 | 79.59 |
| CV | 75.89 | 72.20 | 73.13 | 79.03 | 81.43 | 88.35 | 88.86 |


# Cite:
If used, please cite:



# Acknowledgment
We thank Huiyang Wang et al for their wonderful works.

Huiyang Wang, Hua Yu, and Haixian Wang. Eeg_genet: A feature-level graph embedding method for motor imagery classification based on eeg signals. Biocybernetics and Biomedical Engineering, 42(3):1023–1040, 2022.
