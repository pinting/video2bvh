FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04

RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
git g++ wget make libprotobuf-dev protobuf-compiler \
libopencv-dev libgoogle-glog-dev libboost-all-dev libcaffe-cuda-dev libhdf5-dev libatlas-base-dev \
build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev \
libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev curl zlibc

RUN wget https://www.python.org/ftp/python/3.7.15/Python-3.7.15.tgz && \
tar xzf Python-3.7.15.tgz -C /opt && \
rm Python-3.7.15.tgz

WORKDIR /opt/Python-3.7.15
RUN ./configure && make install

WORKDIR /
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install numpy opencv-python
RUN python3 -m pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
rm cmake-3.16.0-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"

WORKDIR /openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

# Disabled CUDNN because of WSL
WORKDIR /openpose/build
RUN cmake -DBUILD_PYTHON=ON -DUSE_CUDNN=OFF .. && make -j `nproc`
RUN make install

WORKDIR /video2bvh
RUN cp -R /usr/local/python/* ./

RUN python3 -m pip install pyyaml easydict h5py matplotlib ipython

COPY . ./