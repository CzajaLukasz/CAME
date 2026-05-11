#!/bin/bash

# Zatrzymanie skryptu w przypadku błędu
set -e

echo "Rozpoczynam zautomatyzowane testy..."

python run.py --dataset-name mnist --arch lenet5 --epochs 2

python run.py --dataset-name cifar10 --arch resnet50 --epochs 2

python run.py --dataset-name cifar10 --arch densenet121 --epochs 2

echo "Wszystkie testy zakończone pomyślnie!"