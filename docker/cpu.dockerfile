# Use an official Ubuntu base image
FROM ubuntu:22.04

# Set up the working directory
WORKDIR /project

##################################
# Install system packages
##################################

# Set environment variables to avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        r-base \
        make \
        git \
        gcc \
        g++ \
        libhdf5-dev \
        libgl1 \
        libglu1-mesa \
        libglvnd-dev \
        mesa-utils \
        libxcursor1 \
        libxinerama1 \
        libxrender1 \
        libxext6 \
        libsm6 \
        libx11-xcb1 \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

##################################
# Install dcm2niix
##################################

# Download a specific version of dcm2niix to match the GPU Dockerfile
RUN curl -fLO https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20240202/dcm2niix_lnx.zip

# Create a directory for dcm2niix and unzip the executable
RUN mkdir -p /opt/dcm2niix \
    && unzip dcm2niix_lnx.zip -d /opt/dcm2niix \
    && rm dcm2niix_lnx.zip

# Add dcm2niix to the PATH
ENV PATH="/opt/dcm2niix:$PATH"

# Clean up unnecessary files
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

##################################
# Install Miniconda
##################################

ARG DEFAULT_ENV=fourflow

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda \
    && rm Miniconda3-latest-Linux-x86_64.sh

RUN ln -s /opt/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
RUN echo ". /opt/miniconda/etc/profile.d/conda.sh" >> /root/.bashrc

# Set up Conda environment
ENV PATH="/opt/miniconda/bin:$PATH"
ENV CONDA_DEFAULT_ENV ${DEFAULT_ENV}
ENV PATH /opt/miniconda/envs/${DEFAULT_ENV}/bin:$PATH

# Copy environment files
COPY ./environment.yaml /project/environment.yaml
COPY ./environment_paraview.yaml /project/environment_paraview.yaml

# Create Conda environments using mamba for faster install
RUN conda install -n base -c conda-forge mamba && \
    mamba env create -f environment.yaml && \
    mamba env create -f environment_paraview.yaml

# Activate the main environment by default
RUN echo "conda activate ${DEFAULT_ENV}" >> /root/.bashrc

# Copy source files
COPY ./src /project/src

# Set Python path
ENV PYTHONPATH "${PYTHONPATH}:/project/src"

# Start an interactive shell to keep the container running
CMD ["/bin/bash"]
