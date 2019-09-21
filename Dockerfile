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
         libpng-dev \
         wget && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=$PYTHON_VERSION numpy=1.15 pyyaml=3.13 scipy=1.2 \
    ipython=7.2 mkl=2019.1 mkl-include=2019.1 cython=0.29 typing=3.6 jupyter=1.0.*
RUN conda install -c pytorch -c fastai fastai
RUN conda install -y -c conda-forge awscli=1.16.* boto3=1.9.*
RUN conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN pip install click==7.0 ptvsd==4.2.* shapely==1.6.4 \
     cython==0.29.* pycocotools==2.0.* line_profiler==2.1.* \
     tensorboard==1.14.* tensorboardX==1.8.* yacs==0.1.*
RUN pip install albumentations==0.3.*

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY mlx /opt/src/mlx
WORKDIR /opt/src
ENV PYTHONPATH=/opt/src:$PYTHONPATH

CMD ["bash"]