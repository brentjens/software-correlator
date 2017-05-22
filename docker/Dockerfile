#
# base
#
FROM ubuntu:17.04

#
# common-environment
#
ENV USER=lofar
ENV INSTALLDIR=/opt

#
# environment
#
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.5

#
# set-uid
#
ENV UID=1000

#
# set-build-options
#
ENV J=8

#
# Base and runtime dependencies
#
#RUN sed -i 's/archive.ubuntu.com/osmirror.rug.nl/' /etc/apt/sources.list 
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y sudo rsync ssh \
        cython3 \
        ipython3 \
        libopenblas-dev \
        libpython3.5-dev \
        python3-dev \
        python3-h5py \
        python3-matplotlib \
        python3-numpy \
        python3-scipy 

#libcfitsio-bin libcfitsio-dev libwcs5 libfftw3-bin libhdf5-10 libboost-python${BOOST_VERSION}.0



WORKDIR "/root/"
COPY ["software-correlator.tar.gz", "."]
RUN  tar xvf software-correlator.tar.gz && \
     cd software-correlator*/ && \
     python3 setup.py install && \
     cd ../


#
# open security holes (allow smooth user switching, allow sudo)
#
RUN echo 'ALL ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    sed -i 's/requiretty/!requiretty/g' /etc/sudoers && \
    chmod a+rw /etc/group /etc/passwd

#
# setup install dir
#
RUN mkdir -p ${INSTALLDIR}

#
# entry
#
COPY ["bashrc",    "${INSTALLDIR}/"]
COPY ["bashrc.d",  "${INSTALLDIR}/bashrc.d/"]
COPY ["chuser.sh", "/usr/local/bin"]
WORKDIR "/"
ENTRYPOINT ["/usr/local/bin/chuser.sh"]

