# IMPORTANTE!!!
# Todo o código foi realizado e executado no ambiente de desenvolvimento Google Colab

# ------------------------------------------ INSTAÇÃO E IMPORTAÇÃO ------------------------------------------

# !pip install numpy --upgrade
# !pip install autokeras
# !pip install hyperopt
# !pip install mahotas
# !pip install git+https://github.com/hyperopt/hyperopt-sklearn.git
# !pip install rembg
# !pip install Pillow

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mahotas as mh
import mahotas.features
import cv2
import seaborn as sns
from tensorflow import keras
from keras import layers
from sklearn import svm
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from skimage import data, exposure
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray
from skimage.filters import laplace, gaussian
from skimage.morphology import skeletonize
from skimage.exposure import histogram
from skimage.util import invert
from hpsklearn import HyperoptEstimator, svc, linear_svc
from hyperopt import tpe, hp
from rembg import remove, new_session
from PIL import Image

path1 = 'AcerPalmatum-SEI_SICITE\Conjuntos\Conjunto 1'
path2 = 'AcerPalmatum-SEI_SICITE\Conjuntos\Conjunto 2'
path3 = 'AcerPalmatum-SEI_SICITE\Conjuntos\Conjunto 3'

# ------------------------------------------ FUNÇÕES ------------------------------------------

# Separar as imagens e labels do dataset
def Extract_image_label(dataset):
  dataset_image = []
  dataset_label = []
  for image, label in dataset:
    dataset_image.append(image)
    dataset_label.append(label)

  return dataset_image, dataset_label

# Transformar a imagem grayscale para binário
def Grayscale_to_binary(image):

  uint8_image = image.astype('uint8')
  threshold = mh.otsu(uint8_image)
  binary = image > threshold

  return binary

# Virar a imagem para esquerda e direita
def Flip_left_right(train_ds):
  flip_lr_ds = train_ds
  flip_lr_ds = flip_lr_ds.map(lambda image, label: (tf.image.flip_left_right(image), label))

  return flip_lr_ds

# Virar a imagem para cima e baixo
def Flip_up_down(train_ds):
  flip_ud_ds = train_ds
  flip_ud_ds = flip_ud_ds.map(lambda image, label: (tf.image.flip_up_down(image), label))

  return flip_ud_ds

# Aproximar a imagem de forma aleatória
def Random_zoom_in(train_ds, seed):
  Zoom = tf.keras.Sequential([
    layers.RandomZoom((-0.3, -0.2), (-0.3, -0.2), 'wrap', 'bilinear', seed)
  ])

  random_zoom_in_ds = train_ds
  random_zoom_in_ds = random_zoom_in_ds.map(lambda image, label: (Zoom(image), label))

  return random_zoom_in_ds

# Aumentar o brilho de forma aleatória
def Random_bright(train_ds, seed):
  random_bright_ds = train_ds
  random_bright_ds = random_bright_ds.map(lambda image, label: (tf.image.stateless_random_brightness(image, 0.2, seed), label))

  return random_bright_ds

# Aumentar o contraste de forma aleatória
def Random_contrast(train_ds, seed):
  random_contrast_ds = train_ds
  random_contrast_ds = random_contrast_ds.map(lambda images, label: (tf.image.stateless_random_contrast(images, 0.2, 0.3, seed), label))

  return random_contrast_ds

# Agrupando dois datasets
def Group_dataset(first_ds, second_ds):
  train_image = []
  train_label = []
  for image, label in first_ds:
    train_image.append(image)
    train_label.append(label)
  for image, label in second_ds:
    train_image.append(image)
    train_label.append(label)

  grouped_ds = tf.data.Dataset.from_tensor_slices((train_image,train_label))

  return grouped_ds

# Realizando o data augmentation
def Data_Augmentation(train_ds):
  seed = (1, 2)

  flip_lr_ds = Flip_left_right(train_ds)
  train_ds = Group_dataset(train_ds, flip_lr_ds)

  flip_ud_ds = Flip_up_down(train_ds)
  train_ds = Group_dataset(train_ds, flip_ud_ds)

  random_zoom_in_1_ds = Random_zoom_in(train_ds, seed)
  random_bright_ds = Random_bright(train_ds, seed)
  random_contrast_ds = Random_contrast(train_ds, seed)

  train_ds = Group_dataset(train_ds, random_zoom_in_1_ds)
  train_ds = Group_dataset(train_ds, random_bright_ds)
  train_ds = Group_dataset(train_ds, random_contrast_ds)

  return train_ds

# Preparando para classificar as imagens originais
def Sem_data_augmentation(dataset_image, dataset_label, image_height, image_width):
  dataset_image = np.array(dataset_image)
  dataset_label = np.array(dataset_label)

  dataset_image = dataset_image.reshape(-1, image_height*image_width*3)

  return dataset_image, dataset_label

# Preparando para classificar as imagens com data augmentation
def Com_data_augmentation(dataset_image, dataset_label, image_height, image_width, augment):

  dataset = tf.data.Dataset.from_tensor_slices((dataset_image,dataset_label))
  if augment == True:
    dataset = Prepare_train_ds(dataset)
  else:
    dataset = Prepare_test_ds(dataset)

  dataset_image, dataset_label = Extract_image_label(dataset)

  dataset_image = np.array(dataset_image)
  dataset_label = np.array(dataset_label)

  dataset_image = dataset_image.reshape(-1, image_height*image_width*3)

  return dataset_image, dataset_label

# Extraindo as cores com histograma e preparando para classificar
def Histogram_extractor(dataset_image, dataset_label, image_height, image_width):

  nbins = 256
  histogram_image = []

  for image in dataset_image:
    hist, bin_center = histogram(image, nbins = nbins, channel_axis = 2)
    hist = hist.reshape(-1)
    histogram_image.append(hist)

  histogram_image = np.array(histogram_image)
  histogram_label = np.array(dataset_label)

  return histogram_image, histogram_label

# Extraindo o formato com zernike moment e preparando para classificar
def Zernike_extractor(dataset_image, dataset_label):

  radius = 120
  degree = 13
  zernike_image = []

  for image in dataset_image:
    grayscale = rgb2gray(image)
    zernike = mh.features.zernike_moments(grayscale, radius, degree)
    zernike_image.append(zernike)

  zernike_image = np.array(zernike_image)
  zernike_label = np.array(dataset_label)

  return zernike_image, zernike_label

# Preparando o dataset de treinamento
def Prepare_train_ds(train_ds):
  rescaling = tf.keras.Sequential([
    layers.Rescaling(1./255)
  ])
  train_ds = train_ds.map(lambda image, label: (rescaling(image), label))
  train_ds = Data_Augmentation(train_ds)

  return train_ds

# Preparando o dataset de teste
def Prepare_test_ds(test_ds):
  rescaling = tf.keras.Sequential([
    layers.Rescaling(1./255)
  ])
  test_ds = test_ds.map(lambda image, label: (rescaling(image), label))

  return test_ds

# Preparando o dataset
def Prepare_dataset(dataset):
  rescaling = tf.keras.Sequential([
    layers.Rescaling(1./255)
  ])
  dataset = dataset.map(lambda image, label: (rescaling(image), label))

  return dataset


# ------------------------------------------ MAIN ------------------------------------------

# Modo    Descrição                           |   Conjunto
# 1       Sem data augmentation               |   1
# 2       Com data augmentation               |   2
# 3       Histogram (color)                   |   3
# 4       Zernike Moments (shape)             |

modo = 1
conjunto = 1

if conjunto == 1:
  path = path1
elif conjunto == 2:
  path = path2
elif conjunto == 3:
  path = path3

  # Altura e largura da imagem
image_height = 180
image_width = 180

# Dataset para treinamento e teste do classificador
dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    labels = 'inferred',
    label_mode = 'int',
    color_mode = 'rgb',
    batch_size = None,
    image_size = (image_height, image_width),
    shuffle = False,
    seed = 0,
    validation_split = None,
    subset = None)

class_names = dataset.class_names

if (modo!=5):
  dataset = Prepare_dataset(dataset)

dataset_image, dataset_label = Extract_image_label(dataset)

X_train, X_test, y_train, y_test = train_test_split(dataset_image,
                                                    dataset_label,
                                                    test_size = 0.2,
                                                    random_state = 0,
                                                    shuffle = True,
                                                    stratify = dataset_label)

if modo == 1:
  X_train, y_train = Sem_data_augmentation(X_train, y_train, image_height, image_width)
  X_test, y_test = Sem_data_augmentation(X_test, y_test, image_height, image_width)
elif modo == 2:
  X_train, y_train = Com_data_augmentation(X_train, y_train, image_height, image_width, True)
  X_test, y_test = Com_data_augmentation(X_test, y_test, image_height, image_width, False)
elif modo == 3:
  X_train, y_train = Histogram_extractor(X_train, y_train, image_height, image_width)
  X_test, y_test = Histogram_extractor(X_test, y_test, image_height, image_width)
elif modo == 4:
  X_train, y_train = Zernike_extractor(X_train, y_train)
  X_test, y_test = Zernike_extractor(X_test, y_test)

models = []
skf = StratifiedKFold(n_splits = 3)
estim = HyperoptEstimator(classifier=svc('mySVC', probability=True))

for train, test in skf.split(X_train, y_train):
  estim.fit(X_train[train], y_train[train])
  # print(f'\nscore: {estim.score(X_train[test], y_train[test])}\n')
  models.append(estim.best_model()['learner'])
  print(models[-1])
  print('----------------------------------------------------------------------')

eclf = VotingClassifier(estimators=[('best_model_1',models[0]),
                                    ('best_model_2',models[1]),
                                    ('best_model_3',models[2])],
                                    voting='soft')

eclf = eclf.fit(X_train, y_train)

pred = eclf.predict(X_test)
print(pred)
print(y_test)

print(classification_report(y_test, pred, zero_division=0))

if conjunto != 1:
  labelY = ['$\it{atropurpureum}$', '$\it{bihou}$', '$\it{common}$', '$\it{deshojo}$', '$\it{dissectum}$', '$\it{dissectum}$ $\it{atropurpureum}$', '$\it{dissectum}$ $\it{rubrum}$']
  labelX = ['$\it{atropurpureum}$', '$\it{bihou}$', '$\it{common}$', '$\it{deshojo}$', '$\it{dissectum}$', '$\it{diss.}$ $\it{a.}$', '$\it{diss.}$ $\it{r}$.']
else:
  labelY = ['$\it{atropurpureum}$','$\it{beni}$ $\it{kawa}$','$\it{bihou}$','$\it{bloodgood}$', '$\it{common}$', '$\it{deshojo}$', '$\it{dissectum}$', '$\it{dissectum}$ $\it{atropurpureum}$', '$\it{dissectum}$ $\it{rubrum}$','$\it{hogyoku}$','$\it{jordan}$','$\it{katsura}$','$\it{orange}$ $\it{dream}$','$\it{sango}$ $\it{kaku}$']
  labelX = ['$\it{atr.}$','$\it{ben.}$','$\it{bih.}$','$\it{blo.}$', '$\it{com.}$', '$\it{des.}$', '$\it{dis.}$', '$\it{dis.A}$', '$\it{dis.R}$','$\it{hog.}$','$\it{jor.}$','$\it{kat.}$','$\it{ora.}$','$\it{san.}$']

fig, ax = plt.subplots(figsize=(13, 11))
c_matrix = confusion_matrix(y_test, pred)
sns.set(font_scale=1.0)

sns.heatmap(c_matrix, annot=True, linewidths=5, cbar_kws={"shrink": 1}, square=True, xticklabels = labelX, yticklabels = labelY)

plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(rotation=0)
plt.xlabel('')
plt.ylabel('')

if conjunto != 1:
  labelY = ['$\it{atropurpureum}$', '$\it{bihou}$', '$\it{common}$', '$\it{deshojo}$', '$\it{dissectum}$', '$\it{dissectum}$ $\it{atropurpureum}$', '$\it{dissectum}$ $\it{rubrum}$']
  labelX = ['$\it{atropurpureum}$', '$\it{bihou}$', '$\it{common}$', '$\it{deshojo}$', '$\it{dissectum}$', '$\it{diss.}$ $\it{a.}$', '$\it{diss.}$ $\it{r}$.']
else:
  labelY = ['$\it{atropurpureum}$','$\it{beni}$ $\it{kawa}$','$\it{bihou}$','$\it{bloodgood}$', '$\it{common}$', '$\it{deshojo}$', '$\it{dissectum}$', '$\it{dissectum}$ $\it{atropurpureum}$', '$\it{dissectum}$ $\it{rubrum}$','$\it{hogyoku}$','$\it{jordan}$','$\it{katsura}$','$\it{orange}$ $\it{dream}$','$\it{sango}$ $\it{kaku}$']
  labelX = ['$\it{atr.}$','$\it{ben.}$','$\it{bih.}$','$\it{blo.}$', '$\it{com.}$', '$\it{des.}$', '$\it{dis.}$', '$\it{dis.A}$', '$\it{dis.R}$','$\it{hog.}$','$\it{jor.}$','$\it{kat.}$','$\it{ora.}$','$\it{san.}$']

fig, ax = plt.subplots(figsize=(13, 11))
c_matrix = confusion_matrix(y_test, pred, normalize="true")
sns.set(font_scale=1.0)
sns.heatmap(c_matrix, annot=True, linewidths=5, cbar_kws={"shrink": 1}, square=True, xticklabels = labelX, yticklabels = labelY)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(rotation=0)
plt.xlabel('')
plt.ylabel('')