#!/bin/bash

sudo nvidia-docker build \
     --build-arg UID=`id -u` --build-arg USERNAME=`whoami` \
     -t ${USER}/kaggle -f docker/Dockerfile ./docker
