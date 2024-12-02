#!/usr/bin/env bash

pip3 install transformers
pip3 install datasets
pip3 install tensorboard
pip3 install pytorch-crf==0.7.2
pip3 install seqeval==1.2.2
pip3 install byted-tracking -i https://bytedpypi.byted.org/simple/
pip3 install -U byted-wandb -i https://bytedpypi.byted.org/simple
# # lego
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org
# 改图工具THSEditor
wget http://d.scm.byted.org/api/v2/download/ceph:data.aml.thseditor_1.0.0.51.tar.gz && tar -zxvf ./ceph\:data.aml.thseditor_1.0.0.51.tar.gz && pip3 install ./thseditor-0.1.8-cp37-cp37m-linux_x86_64.whl --force-reinstall

# Lego Ops算子
pip3 install https://luban-source.byted.org/repository/scm/data.aml.lego_ops_th110_cu113_cudnn820_sdist_1.0.0.236.tar.gz

# Lego Pipeline
pip install https://d.scm.byted.org/api/v2/download/data.aml.lego_1.0.0.152.tar.gz --force-reinstall
