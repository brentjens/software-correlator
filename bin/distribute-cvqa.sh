#!/usr/bin/bash

IMG=cvqa
# -f option is only in docker version on head nodes, not cpu nodes which have newer version
docker tag -f $IMG nexus.cep4.control.lofar:18080/$IMG
docker push nexus.cep4.control.lofar:18080/$IMG
# 12 is max num of cpu nodes to prevent hangs
clush -f 12 -w cpu[01-50] docker pull nexus.cep4.control.lofar:18080/$IMG 
clush -f 12 -w cpu[01-50] docker tag  nexus.cep4.control.lofar:18080/$IMG $IMG
