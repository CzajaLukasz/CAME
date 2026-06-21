import os
import numpy as np
import torchvision

# Utwórz strukturę katalogów, jeśli nie istnieje
os.makedirs('./datasets/CIFAR100', exist_ok=True)

print("Pobieranie CIFAR100 przez torchvision...")
train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)

print("Konwersja i zapisywanie plików .npy...")
# torchvision przechowuje obrazy jako numpy array w `.data`
np.save('./datasets/CIFAR100/train_data.npy', train_set.data)
np.save('./datasets/CIFAR100/test_data.npy', test_set.data)

# Na wypadek, gdyby Twój skrypt szukał też etykiet (częsty przypadek):
np.save('./datasets/CIFAR100/train_labels.npy', np.array(train_set.targets))
np.save('./datasets/CIFAR100/test_labels.npy', np.array(test_set.targets))

print("Sukces! Pliki .npy zostały przygotowane w ./datasets/CIFAR100/")