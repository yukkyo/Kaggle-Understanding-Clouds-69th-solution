#!/bin/bash

sudo nvidia-docker run --rm \
    --shm-size=6G \
    -v `pwd`:/home/${USER}/siim2019 \
    -p 8758:8888 -p 8756:6006 -p 8752:22\
    --name ${USER}.kaggle \
    -itd ${USER}/kaggle
