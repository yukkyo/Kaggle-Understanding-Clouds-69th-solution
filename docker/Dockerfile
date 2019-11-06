FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Basic setting library
RUN apt-get update \
  && apt-get install --yes --no-install-recommends \
    sudo \
    git \
    tig \
    cmake \
    pkg-config \
    unzip \
    wget \
    build-essential \
    libssl-dev\
    libbz2-dev\
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    curl \
    libsm6 \
    libgtk2.0-dev \
    vim \
    less \
    pv \
  && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Make sudo user
ARG UID
ARG USERNAME
RUN useradd ${USERNAME} -u ${UID} -G sudo -s /bin/bash -m  && \
    echo ${USERNAME}' ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV HOME /home/${USERNAME}

# Install pyenv & install python
ENV PYTHON_VERSION 3.6.6
ENV PYENV_ROOT /home/${USERNAME}/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

# Install python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip setuptools && \
    pip install -r /tmp/requirements.txt

# Install kaggle command
COPY kaggle_apikey.json /tmp/kaggle.json
RUN mkdir $HOME/.kaggle && \
    cp /tmp/kaggle.json $HOME/.kaggle/kaggle.json && \
    chmod 600 $HOME/.kaggle/kaggle.json

# Install fish shell
RUN sudo apt-get update && \
    sudo apt-get -y install software-properties-common && \
    sudo apt-add-repository ppa:fish-shell/release-3 && \
    sudo apt-get -y install fish && \
    curl https://git.io/fisher --create-dirs -sLo ~/.config/fish/functions/fisher.fish

# change the default shell
ENV SHELL /usr/bin/fish
# install your fish plugins
RUN /usr/bin/fish -c "fisher add oh-my-fish/eclm jethrokuan/z 0rax/fish-bd oh-my-fish/plugin-peco"
