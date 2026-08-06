FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG ELIXIR_VERSION=1.20.2
ARG OTP_VERSION=29.0

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
      build-essential \
      ca-certificates \
      clang-format \
      cmake \
      curl \
      g++-14 \
      gcc-14 \
      git \
      inotify-tools \
      ninja-build \
      openssh-client \
      pkg-config \
      python3 \
      python3-venv \
      sudo \
      unzip \
      wget \
      xz-utils \
      zsh && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 140 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 140 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-14 140 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-14 140 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN usermod --shell /bin/zsh ubuntu && \
    usermod --append --groups sudo ubuntu && \
    echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu && \
    mkdir -p /workspace && \
    chown ubuntu:ubuntu /workspace

USER ubuntu

ENV PATH="/home/ubuntu/.local/share/mise/shims:/home/ubuntu/.local/bin:${PATH}"

RUN curl https://mise.run | sh && \
    /home/ubuntu/.local/bin/mise use --global erlang@${OTP_VERSION} && \
    /home/ubuntu/.local/bin/mise use --global elixir@${ELIXIR_VERSION} && \
    git config --global --add safe.directory /workspace && \
    mix local.hex --force && \
    mix local.rebar --force

WORKDIR /workspace
