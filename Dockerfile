# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

FROM tensorflow/tensorflow:1.14.0-gpu-py3
COPY requirements.txt /
RUN apt-get update && apt-get install -y --no-install-recommends cmake \
    python3-pip language-pack-ja-base language-pack-ja\
 && apt-get clean\
 && rm -rf /var/lib/apt/lists/*\
 && pip3 install --upgrade pip setuptools\
 && pip3 install -r requirements.txt\
 && mkdir share\
 && locale-gen ja_JP.UTF-8
ENV PYTHONIOENCODING=utf-8 LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:en LC_ALL=ja_JP.UTF-8