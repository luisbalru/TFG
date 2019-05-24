# SCRIPT PARA EL EXPERIMENTO 3
# Elegido sagital y C1 (materia blanca), llevo a cabo un ensemble learner (stacking)
# para elegir las slices más determinantes en el diagnóstico
# Author: Luis Balderas Ruiz

from subjects import Sujeto
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

pacientes = []
imagenes_pd = os.listdir('./Datos/PD')
# Set the parameters by cross-validation


C_range = np.logspace(-2,11,13)
gamma_range = np.logspace(-9,3,13)
#C_range = np.logspace(-5, 15, 11)
#gamma_range = np.logspace(3, -15, 10)
param_grid = dict(gamma=gamma_range, C=C_range)
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

scores = ['recall']
clasificadores = []

print("LECTURA DE IMAGENES")

print("Leyendo pacientes enfermos")

for i in range(len(imagenes_pd)):
    imagenes_pd[i] = './Datos/PD/' + imagenes_pd[i]
    paciente = Sujeto(imagenes_pd[i],0)
    pacientes.append(paciente)

pde = len(pacientes)
print("Enfermos: ",pde)
imagenes_control = os.listdir('./Datos/Control')

print("Leyendo pacientes control")

for i in range(len(imagenes_control)):
    imagenes_control[i] = './Datos/Control/' + imagenes_control[i]
    paciente = Sujeto(imagenes_control[i],1)
    pacientes.append(paciente)

sanos = len(pacientes) - pde
print("Sanos: ",sanos)


## REPARTO LOS PACIENTES ENTRE CONJUNTO DE TRAINING Y TEST ESTRATIFICADO PARA LAS DOS FASES DEL
## STACKING (70%-30%). PARA TRAINING, 114-55. PARA TEST, 50-23 (ENFERMO-SANO).
pacientes_training = []
train_enfermos = 0
train_sanos = 0
test_enf = 0
test_sanos = 0
pacientes_test = []
np.random.shuffle(pacientes)

for i in range(len(pacientes)):
    if paciente[i].get_label() == 0:
        if train_enfermos <= 114:
            train_enfermos = train_enfermos + 1
            pacientes_training.append(paciente[i])
        else:
            test_enf = test_enf + 1
            pacientes_test.append(paciente[i])
    else:
        if train_sanos <= 50:
            train_enfermos = train_enfermos + 1
            pacientes_training.append(paciente[i])
        else:
            test_enf = test_enf + 1
            pacientes_test.append(paciente[i])



Y_shape = pacientes_training[0].get_shape(1)

for i in range(Y_shape):
    print("Slice ", i)
    dataset = []
    target = []
    np.random.shuffle(pacientes_training)
    for j in range(len(pacientes_training)):
        slice = pacientes[j].get_slice(1,i)
        label = pacientes[j].get_label()
        row = pacientes[j].get_wave2D(slice,'bior3.3',2)
        dataset.append(row)
        target.append(label)

    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    X_train, X_test, y_train, y_test = train_test_split(dataset,target,test_size = 0.2,stratify = target)
    y_train = np.array(y_train)
    pca = PCA(n_components = 0.95, svd_solver = 'full')
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    for score in scores:
        clf = GridSearchCV(SVC(kernel='rbf',class_weight = 'balanced'),param_grid = param_grid, cv=10,scoring = '%s_macro' % score)
        clf.fit(X_train,y_train)
"""
