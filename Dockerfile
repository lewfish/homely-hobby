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
RUN conda install -y python=$PYTHON_VERSION numpy=1.15 pyyaml=3.13 scipy=1.2 \
    ipython=7.2 mkl=2019.1 mkl-include=2019.1 cython=0.29 typing=3.6
RUN conda install -y -c pytorch magma-cuda100=2.5 torchvision=0.2
RUN conda clean -ya

ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=$PYTHON_VERSION
RUN conda install -y -c fastai fastai=1.0.53
RUN conda install -y -c conda-forge awscli=1.16.* boto3=1.9.*
RUN conda install -y jupyter=1.0.*
RUN conda clean -ya

RUN pip install click==7.0
RUN pip install ptvsd==4.2.*
RUN pip install shapely==1.6.4
RUN pip install cython==0.29.*
RUN pip install pycocotools==2.0.*
RUN pip install line_profiler==2.1.*

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY mlx /opt/src/mlx
WORKDIR /opt/src
ENV PYTHONPATH=/opt/src:$PYTHONPATH

CMD ["bash"]