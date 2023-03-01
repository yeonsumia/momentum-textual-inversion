FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

ENV PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5;8.0" \
    DEBIAN_FRONTEND=noninteractive
ARG APT_INSTALL="apt-get install -y --no-install-recommends"
ARG USERNAME=hyeonseong
ARG UID=2071
ARG GID=${UID}

RUN rm -rf /var/lib/apt/lists/* && apt-get -y update && apt-get -y upgrade

RUN groupadd --gid ${GID} ${USERNAME} \
    && useradd --uid ${UID} --gid ${GID} -m -s /bin/bash ${USERNAME} \
    && $APT_INSTALL sudo nano curl wget unzip git vim python3-pip zsh locales language-pack-en lpips cmake dlib \
    && rm -rf /var/lib/apt/lists/* \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
ENV HOME /home/${USERNAME}
WORKDIR ${HOME}
COPY environment.yaml .
COPY setup.py .

# Install conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir ~/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && sudo ln -s ~/miniconda3/bin/conda /usr/local/bin/conda \
    && sudo rm -rf /var/lib/apt/lists/* \

# Install zsh for interactive container
RUN sudo locale-gen en_US.UTF-8 \
    && sudo update-locale \
    && sudo chsh -s /usr/bin/zsh \
    && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended > /dev/null \
    && git clone -q https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
    && git clone -q https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && sed -i 's/^ZSH_THEME.*/ZSH_THEME="agnoster"/' ~/.zshrc \
    && sed -i 's/^plugins.*/plugins=(git zsh-syntax-highlighting zsh-autosuggestions)/' ~/.zshrc \
    && sudo rm -rf /var/lib/apt/lists/* \

# Install python packages by conda
ENV CONDA_ENV=ldm
RUN conda init zsh \
    && conda env create -f environment.yaml \
    && echo "conda activate ${CONDA_ENV}" >> ~/.zshrc

EXPOSE 7777
CMD ["zsh"]
