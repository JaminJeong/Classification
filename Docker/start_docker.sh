#!/bin/bash
/usr/bin/xhost +

HOME=$HOME
default=1.13.1-gpu-py3
tag=${1:-$default}
echo $tag
  #tensorflow/tensorflow:$tag \
NAME=tf${USER}$(date +%Y%m%d%H%M%S) 

if [[ "$(docker images -q tensorflow/tensorflow:$tag 2> /dev/null)" == "" ]] ; then
  sudo docker build -t tensorflow/tensorflow:$tag ./ --build-arg VERSION=$tag
fi

sudo nvidia-docker run -it \
  --name ${NAME}_${tag} \
	--privileged \
  -v /etc/group:/etc/group:ro \
  -v /etc/passwd:/etc/passwd:ro \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/video0:/dev/video0 \
  -v $HOME:$HOME \
  -w=$(pwd) \
  -u=$UID:$(id -g $USER) \
	-e DISPLAY=$DISPLAY \
	-e TZ=Asia/Seoul \
  -e QT_X11_NO_MITSHM=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 6006:6006 \
  tensorflow/tensorflow:$tag \
  /bin/bash

#docker exec -it ${NAME}_${tag} tmux
#  /bin/bash

