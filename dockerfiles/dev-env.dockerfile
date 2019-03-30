FROM quay.io/fenicsproject/dev-env:2018.1.0 as dev

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
USER fenics
WORKDIR /home/fenics
RUN sudo chmod g+w -R ../fenics
ENV FENICS_VERSION=2018.1.0
ENV PATH=$PATH:/home/fenics/bin
ENV FENICS_BUILD_TYPE='Debug'
RUN sed -i 's/\(^ *make$\)/\1 -j/' bin/fenics-build
RUN cat bin/fenics-build && cat bin/fenics-update
RUN /bin/bash -c ". fenics.env.conf; env; fenics-update"

from dev as Deploy

USER root
#COPY --from=dev /home/fenics/local .
#COPY --from=dev /home/fenics/fenics.env.conf .
#ENV FENICS_VERSION=2018.1.0

# Dependencies
RUN apt-get update -y &&\
    apt-get install -y python3-tk git

# Shell
RUN apt-get install -y zsh

# Interface
RUN useradd --create-home --shell /bin/zsh --gid users --groups sudo,docker_env,fenics mtnlion
RUN echo "mtnlion ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER mtnlion
WORKDIR /home/mtnlion

# Install mtnlion
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/macklenc/mtnlion.git
RUN cd mtnlion && sudo -E python3 setup.py install && cd ..

# Nice shell
RUN wget -O .zshrc https://git.grml.org/f/grml-etc-core/etc/zsh/zshrc &&\
    wget -O .zshrc.local  https://git.grml.org/f/grml-etc-core/etc/skel/.zshrc

FROM deploy as test

RUN sudo sh -c "echo 'export PYTHONPATH=$PYTHONPATH:$HOME/mtnlion/buildup' >> /etc/profile"
RUN cd mtnlion &&\
    sudo -EH pip3 install -r requirements_dev.txt &&\
    git lfs pull &&\
    cd ..

FROM test as sde

USER root

# Fetch pycharm
RUN wget https://download.jetbrains.com/python/pycharm-professional-2018.3.5.tar.gz

# Install git flow
RUN wget https://raw.githubusercontent.com/petervanderdoes/gitflow-avh/develop/contrib/gitflow-installer.sh &&\
    bash gitflow-installer.sh install stable &&\
    rm gitflow-installer.sh

# Install gvim
RUN apt-get install -y vim-gtk3

# Install sublime
RUN wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | apt-key add - &&\
    apt-get install -y apt-transport-https &&\
    echo "deb https://download.sublimetext.com/ apt/stable/" | tee /etc/apt/sources.list.d/sublime-text.list &&\
    apt-get update &&\
    apt-get install -y sublime-text

# Install pycharm
RUN tar xfz pycharm-*.tar.gz -C /opt/ &&\
    rm pycharm-* &&\
    cd /usr/bin &&\
    ln -s /opt/pycharm-*/bin/pycharm.sh pycharm

USER mtnlion

ENTRYPOINT ["/bin/zsh"]
