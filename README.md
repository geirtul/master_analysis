# Master's thesis analysis
Analysis work on scintillator data for master's thesis.
This repository contains notebooks and sample data used increating an analysis
framework for the scintillator data I'm working with in my master's thesis.

## Folder structure
Strict! The data scripts provided in github.com/geirtul/master_scripts assume
the folder structure in this repository is kept, and data is put where it should be.

├── data\
│   ├── real\
│   ├── sample\
│   └── simulated\
├── notebooks\
│   ├── classification\
│   ├── exploration\
│   ├── prediction\
│   └── pretrained\
└── README.md\
## data/
Contains datafiles for analysis.\
├── real - Real data from scintillator experiments. Not provided.\
├── sample - Simulated sample data. Provided.\
└── simulated - Simulated data. Not provided.\
\
As the structure in the data folder suggests, the different types of data are
kept in different folders. The sample data folder already contains sample data.

### Sample data
CeBr10kSingle are 10,000 rows of data and labels for single interactions in the detector\
CeBr10k_1.txt is 10,000 rows of data and labels with a mix of single interactions and double interactions\
CeBr10.txt is a small file I use which contains 10 single interactions.\

## notebooks/
All notebooks used in the analysis work.\
├── classification - classifying single and double events\
├── exploration - data exploration\
├── prediction - prediction of energy and positions of origin, single and double events\
└── pretrained - benchmarks with known models like VGG16\

