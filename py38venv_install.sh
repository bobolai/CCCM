#!/bin/bash

# Update and upgrade system
echo "Updating and upgrading system..."
sudo apt update -y
sudo apt upgrade -y

# Download Python 3.8.10
echo "Downloading Python 3.8.10..."
wget -q https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tar.xz

# Extract Python package
echo "Extracting Python..."
tar -xf Python-3.8.10.tar.xz
cd Python-3.8.10

# Install required dependencies
echo "Installing dependencies..."
sudo apt install -y build-essential zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev \
    libreadline-dev libffi-dev curl libbz2-dev pkg-config make

# Configure and compile Python
echo "Configuring and compiling Python..."
./configure --enable-optimizations --enable-shared
make -j$(nproc)

# Install Python
echo "Installing Python 3.8.10..."
sudo make altinstall
sudo ldconfig /usr/local/lib

# Install pip
echo "Installing pip..."
wget -q https://bootstrap.pypa.io/get-pip.py
python3.8 get-pip.py
python3.8 -m pip install --upgrade pip

# Install virtualenv
echo "Installing virtualenv..."
pip3.8 install virtualenv

# Create virtual environment
echo "Creating virtual environment..."
cd ~
python3.8 -m virtualenv CCCM_venv

echo "Setup complete. To activate the virtual environment, run:"
echo "source ~/CCCM_venv/bin/activate"
