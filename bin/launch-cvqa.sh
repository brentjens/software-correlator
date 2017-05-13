#!/usr/bin/bash

SAS_ID=$1

srun -N 1 docker run -it cvqa cvqa --output-dir="/data/projects/HOLOG_WINDMILL_TESTS" "/data/projects/HOLOG_WINDMILL_TESTS/$SAS_ID/cs/"

