#!/usr/bin/env bash
set -e
if [ "$1" = "original" ]; then
  wget "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
  wget "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
  wget "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
  wget "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
else
  wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
  wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
  wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
  wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
fi
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
