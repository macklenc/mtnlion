FROM quay.io/fenicsproject/stable:2017.2.0

RUN apt-get -y update

RUN wget https://download.jetbrains.com/python/pycharm-professional-2018.3.5.tar.gz

RUN apt-get install -y xauth xorg openbox

# Install git and git flow
RUN apt-get -y install git &&\
    wget --no-check-certificate -q  https://raw.githubusercontent.com/petervanderdoes/gitflow-avh/develop/contrib/gitflow-installer.sh &&\
    bash gitflow-installer.sh install stable &&\
    rm gitflow-installer.sh

# Install gvim
RUN apt-get install -y vim-gtk3

# Install zsh
RUN wget -O .zshrc https://git.grml.org/f/grml-etc-core/etc/zsh/zshrc &&\
    wget -O .zshrc.local  https://git.grml.org/f/grml-etc-core/etc/skel/.zshrc &&\
    apt-get install -y zsh

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

RUN git clone https://github.com/macklenc/mtnlion /home/fenics/mtnlion

CMD ["/bin/zsh"]
