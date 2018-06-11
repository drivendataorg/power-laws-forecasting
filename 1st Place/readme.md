# Project Title

Drivedata.org Power Laws: Forecasting Energy Consumption competition



### Prerequisites

Python 3
LightGBM 2.0.10 (should work on previous versions as well but not tested)
numpy, pandas, scikit-learn, psutil




### Hardware requirements:

At least 64Gb of RAM is highly recommended 
8+ virtual cores CPU is recommended
~10Gb of disk space



### Installing


Put original .csv files into 'data' subfolder and create empty 'data/my' folder for intermediate and output files. 
Or put them wherever you need and edit get_paths function in util.py file accordingly


## Running


just run solution.py 
It should do everything without human intervention, however this process is long and can take up to 24 hours on 16-vcores CPU



### Results

final_prediction.csv is generated in 'data/my' folder


