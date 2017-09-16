#!/usr/bin/env bash
set -e
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
mkdir -p fashion_mnist
mv train-images-idx3-ubyte fashion_mnist
mv train-labels-idx1-ubyte fashion_mnist
mv t10k-images-idx3-ubyte fashion_mnist
mv t10k-labels-idx1-ubyte fashion_mnist
