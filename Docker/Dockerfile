ARG VERSION

FROM tensorflow/tensorflow:$VERSION

RUN apt update && apt install -y vim

RUN pip3 install -U pip
RUN pip3 install opencv-python wget
