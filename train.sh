#!/usr/bin/env bash

clear

case "$1" in
    0)
    echo "run resnet 101 debug"
    CUDA_VISIBLE_DEVICES=0,1 python train.py --model 'resnet101' --n_gpus=2 --batch_size 50 --optim='sgd' --learning_rate_decay_start=0
    ;;

    1)
    echo "run vgg19 debug"
    CUDA_VISIBLE_DEVICES=0,1 python train.py --model 'vgg19' --n_gpus=2 --batch_size 10 --optim='sgd' --learning_rate_decay_start=0
    ;;

    2)
    echo "run resnet101 debug"
    CUDA_VISIBLE_DEVICES=1 python train.py --model 'resnet101' --batch_size 10 --learning_rate 1e-4
    ;;

    *)
    echo
    echo "No input"
    ;;
esac

