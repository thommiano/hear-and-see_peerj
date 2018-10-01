from nvidia/cuda:8.0-cudnn7-runtime-ubuntu16.04

# from digital genius
# https://github.com/DigitalGenius/docker-pytorch/blob/master/Dockerfile

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libssl-dev \
        pkg-config \
        python3 \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        python3-tk \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# Install pip for python3
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        numpy \
        scipy \
        sklearn \
        scikit-image \
        matplotlib \
        jupyter \
        tqdm

# Updated for pytorch 0.2.0
# RUN pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
# RUN pip install torchvision
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
RUN pip install torchvision

# Install Pytorch Audio
RUN apt-get update && apt-get install git -y
RUN git clone https://github.com/pytorch/audio.git
WORKDIR /audio
RUN apt-get update && apt-get install sox -y libsox-dev libsox-fmt-all
RUN pip install cffi
RUN python setup.py install
RUN pip install librosa sounddevice
RUN apt-get install libav-tools -y
RUN apt-get install libportaudio2 -y
RUN pip install git+https://github.com/pytorch/tnt.git@master
RUN pip install opencv-python

# -y -2 -105
#RUN apt-get install python3-tk -y

WORKDIR /hear_and_see
