#!/usr/bin/bash

SAS_ID=$1
# -u userid/groupid lofarsys

#srun -N 1 docker run -v /data:/data:rw -u lofarsys:lofarsys --rm=true -t cvqa cvqa --output-dir="/data/projects/HOLOG_WINDMILL_TESTS" "/data/projects/HOLOG_WINDMILL_TESTS/$SAS_ID/cs/"

#docker-run-slurm.sh 

srun -N 1 -J "xcflagavg-$SAS_ID-$USER" \
        docker run --rm -u 7149 \
        -e USER=$USER -e HOME=$HOME \
        -v /data:/data:rw \
        -v /globalhome:/globalhome:rw \
        --network host \
        -t cvqa:latest xcflagavg \
            --output-dir="/data/projects/HOLOG_WINDMILL_TESTS" \
            --max-mem-gb=128 \
           $2 $3 $4 $5 $6 $7 \
           "/data/projects/HOLOG_WINDMILL_TESTS/$SAS_ID-*SB???.hdf5"

