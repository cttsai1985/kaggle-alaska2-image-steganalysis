#!/bin/bash

mkdir input; cd input;

# man data
TYPE=competitions
DATA=alaska2-image-steganalysis
kaggle ${TYPE} download --force -c ${DATA}
mkdir ${DATA}
unzip ${DATA} -d ${DATA} 

# external data: clean dift
TYPE=datasets
OWNER=cttsai
DATA=alaska2-image-steganalysis-image-quality
kaggle ${TYPE} download --force ${OWNER}/${DATA}
mkdir ${DATA}
unzip ${DATA} -d ${DATA} 


