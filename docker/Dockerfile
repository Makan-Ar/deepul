FROM nvcr.io/nvidia/pytorch:23.07-py3

LABEL maintainer="makan.arastuie@gmail.com"

ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install build-essential
RUN apt-get -y install manpages-dev
RUN apt -y install screen

# Creating a user appuser and switching from root
ARG UNAME=appuser
ARG UID=1000
ARG GID=1000

RUN apt-get install sudo -y
RUN groupadd -g $GID -o $UNAME
RUN adduser --uid $UID --gid $GID --disabled-password --gecos '' --shell /bin/bash $UNAME
RUN adduser $UNAME sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $UNAME

WORKDIR /home/$UNAME

# Installing dependencies & adding $UNAME bin to PATH
ENV PATH="/home/$UNAME/.local/bin:${PATH}"

RUN mkdir misc
COPY requirements.txt misc
RUN pip install -r misc/requirements.txt