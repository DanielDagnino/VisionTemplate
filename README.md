# VisionTemplate

## Introduction
This template serves as a personal framework designed to facilitate the training of models for image classification and semantic segmentation using PyTorch. Its core objective is to maintain a meticulously organized repository where custom functions—ranging from losses and metrics to optimizers—can be seamlessly integrated. This approach allows for seamless experimentation and implementation of novel methodologies, such as those introduced in research articles.

New methods find their place in dedicated folders within the structure. The training process remains highly configurable through a YAML file, enabling simple execution via a user-friendly CLI (Command Line Interface). Additionally, the results are systematically tracked through comprehensive logs and TensorBoard, providing a comprehensive overview of the training progress and outcomes.

## Installation
* Training:
```bash
cd <PATH_TO_VisionTemplate>/vision
mkvirtualenv VisionTemplate -p python3.10
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=.:..:../../submodules
```

### Docker
Test with 2CPUs and Push:

```bash
docker run \
    --name vision \
    --shm-size 1024M \
    --cpuset-cpus 0-1 \
    -p 8080:8080 \
    --rm \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    vision:v2.0.0

docker exec -it vision bash

docker login
docker tag vision:v2.0.0 dagnino/vision:v2.0.0
docker push dagnino/vision:v2.0.0
```

### Create an Azure VM running the model
```bash
# Push Docker Image to a Registry:
chmod 400 $HOME/.ssh/azure-vm-visiontemplate.pem

# SSH
ssh -i $HOME/.ssh/azure-vm-visiontemplate-clf.pem azureuser@IP

# Update package list and install required packages:
sudo apt update
sudo apt install -y docker.io

# Start the Docker service:
sudo systemctl start docker

# Add your user to the docker group to avoid using sudo with every Docker command:
sudo usermod -aG docker $USER

# Fix error when running docker pull ...
sudo chmod 666 /var/run/docker.sock

# Run Your Docker Container:
docker login
docker pull dagnino/vision:v2.0.0
docker run -d -p 8080:8080 dagnino/vision:v2.0.0

docker-compose up -d
```

### Submodules
#### Initialize
```bash
git submodule init
git submodule update
```

```bash
cd submodules/demucs
pip install -e .
```
