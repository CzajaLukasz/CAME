#!/bin/bash

# Zatrzymanie skryptu w przypadku błędu
set -e

# echo "Rozpoczynam zautomatyzowane testy..."

# python run.py --dataset-name mnist --arch lenet5 --epochs 2

# python run.py --dataset-name cifar10 --arch resnet50 --epochs 2

# python run.py --dataset-name cifar10 --arch densenet121 --epochs 2

# echo "Wszystkie testy zakończone pomyślnie!"

# Scenariusz 1: Testy pojemności i typu architektury (Architecture Robustness)
# Podscenariusz 1A: Głęboka sieć rezydualna (resnet50 + SimCLR)

CUDA_VISIBLE_DEVICES=0 python run.py --data ./datasets/ --dataset-name cifar10 --arch resnet50 --batch-size 1024 --epochs 200 --learning-rate 0.4 --optimizer SGD --momentum 0.09 --scheduler MultiStep --milestones 100,150 --gamma 0.1 --cl-model SimCLR --data-setup cifar --fp16-precision --workers 32 --save-dir ./checkpoints/CIFAR10/resnet50_simclr
python meta_set/synthesize_set_cifar.py --cifar-path ./datasets/ --dataset-name cifar10 --metaset-size 500 --sampleset-size 10000 --metaset-dir ./metasets/CIFAR10/resnet50_meta --workers 32
python regression.py --data ./datasets/ --meta-dataset-name cifar10 --metaset-dir ./metasets/CIFAR10/resnet50_meta/dataset_default/ --arch resnet50 --batch-size 512 --save-dir ./checkpoints/CIFAR10/resnet50_simclr --restore-file ./checkpoints/CIFAR10/resnet50_simclr/cifar10/checkpoint_0199.pth --cl-model SimCLR --data-setup cifar --fp16-precision --workers 32
