#!/bin/bash

IMAGE="docker_detectron2:latest"

xhost local:root
docker run -it \
           --gpus all \
           --privileged \
           -v $(pwd)/../../:/detectron2 \
           -v /home/timho/data:/data \
           -w /detectron2/projects/CharGrid \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v /dev/video0:/dev/video0 \
           -e DISPLAY=$(echo $DISPLAY) \
           -e QT_X11_NO_MITSHM=1 \
           -e PYTHONPATH=/detectron2/projects/CharGrid \
           ${IMAGE} \
           /bin/bash