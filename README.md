# Hear and See: Evaluating the use of CNNs for real-time sound classification and visualization of classified sounds

Author: Thom Miano

Acknowledgements: Thank you to [RTI International](https://www.rti.org/) and Georgiy Bobashev for providing the funding to write this paper, and thank you to Rob Chew for reviewing manuscript. Also, thank you to Vicente Ordonez for providing guidance on the methodology.

## Reproducing

### Requirements

The instructions provided below require the following:

1. A machine with an NVIDIA GPU.
2. An internet connection.
3. Docker engine [installed](https://docs.docker.com/install/).
4. You execute commands in the root of the repository (i.e., `hear_and_see/`).

### Environment

This project uses a Docker environment. You can pull a pre-built [image](https://hub.docker.com/r/socraticdatum/hear_and_see/) from Dockerhub with the following:

```bash
docker pull socraticdatum/hear_and_see:latest
```

Alternatively, you can build the Docker image for the project manually by doing the following:

```bash
docker build ./Dockerfile/ -t hear_and_see:latest
```

Launch the Docker container with the following:

```bash
. ./scripts/docker_launch.sh
```

_Note_: The you'll need to update the default arguments in that script for your machine.

If you'd like to run a launch a Jupyter notebook, you can use the following:

```bash
. ./scripts/jupyter_launch.sh
```

_Note_: If you change the ports in the `docker_launch.sh` file, you'll need to make sure they match the port address in this script.

### Dataset

This project uses the open-source UrbanSound8k dataset. Complete the request form and download it [here](https://urbansounddataset.weebly.com/download-urbansound8k.html). Unzip `UrbanSound8k.tar.gz`. When you unzip the file, you should end up with `hear_and_see/data/UrbanSound8k`.
