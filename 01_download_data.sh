#!/bin/bash

mkdir input; cd input;

# man data
TYPE=competitions
DATA=alaska2-image-steganalysis
kaggle ${TYPE} download --force -c ${DATA}
mkdir ${DATA}
unzip ${DATA} -d ${DATA} 


