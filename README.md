# MSMGENet：Multi-scale multi-graph embedding based EEG decoding of motor intentions

Paper link:

Resources:openBMI:( http://dx.doi.org/10.5524/100542)

Environment:
Install the dependencies
It is recommended to create a virtual environment with python version 3.7 and running the following:
pip install -r requirements.txt

Obtain the raw dataset
Download the raw dataset from the resources above, and save them to the same folder. 
braindecode package is directly copied from https://github.com/robintibor/braindecode/tree/master/braindecode for preparing datasets

Start

setp 1 Prepare dataset(Only needs to run once)

python tools/data_bciciv2a_tools.py --data_dir ~/dataset/bciciv2a/gdf -output_dir ~/dataset/bciciv2a/pkl

step 2 Train model


python ho.py -data_dir ~/dataset/bciciv2a/pkl -id 1  or  -data_dir ~/dataset/openBMI -id 1

python cv.py -data_dir ~/dataset/bciciv2a/pkl -id 1  or  -data_dir ~/dataset/openBMI -id 1

Licence
For academtic and non-commercial usage
