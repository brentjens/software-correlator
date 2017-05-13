#!/usr/bin/bash

IMG=cvqa.latest

docker tag -f $IMG nexus.cep4.control.lofar:18080/$IMG
docker push nexus.cep4.control.lofar:18080/$IMG

# 12 is max num of cpu nodes to prevent hangs
clush -f 12 -w cpu[01-50] docker pull nexus.cep4.control.lofar:18080/$IMG 
clush -f 12 -w cpu[01-50] docker tag -f nexus.cep4.control.lofar:18080/$IMG $IMG
