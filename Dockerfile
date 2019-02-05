FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing
RUN conda install -y -c pytorch magma-cuda100 torchvision
RUN conda install -y -c fastai fastai
RUN conda clean -ya

RUN pip install nvidia-ml-py3 dataclasses

WORKDIR /opt/src
