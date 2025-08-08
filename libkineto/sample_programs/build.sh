#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

g++ \
  -g3 \
  -O0 \
  kineto_playground.cpp \
  time_converter.cc \
  kplay_cu.o \
  -o main \
  -I/usr/local/cuda/include \
  -I../src \
  -I../third_party/fmt/include \
  -I/pdlhome/install/kineto/include/kineto \
  -L/usr/local/lib \
  -L/usr/local/cuda/lib64 \
  -L/usr/local/cuda/extras/CUPTI/lib64 \
  -Wl,-rpath,/usr/local/cuda/lib64 \
  -Wl,-rpath,/usr/local/cuda/extras/CUPTI/lib64 \
  /pdlhome/install/kineto/lib/libkineto.a \
  -lpthread \
  -lcuda \
  -lcudart \
  -lcupti \
  -lnvperf_host \
  -lnvperf_target
