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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model


def deep_learning(x_train,y_train,x_test,y_test):
    # network parameters
    num_labels = 2
    print(x_train.shape)
    print(len(y_train))
    print(image_size)
    input_shape = (image_size, image_size, 1)
    batch_size = 128
    kernel_size = 3
    filters = 64
    dropout = 0.3

    # use functional API to build cnn layers
    inputs = Input(shape=input_shape)
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu')(inputs)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu')(y)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu')(y)
    # image to vector before connecting to dense layer
    y = Flatten()(y)
    # dropout regularization
    y = Dropout(dropout)(y)
    outputs = Dense(num_labels, activation='softmax')(y)

    # build the model by supplying inputs/outputs
    model = Model(inputs=inputs, outputs=outputs)
    # network model in text
    model.summary()

    # classifier loss, Adam optimizer, classifier accuracy
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # train the model with input images and labels
    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              epochs=20)
    return model.predict(x_test)


pacientes = []
imagenes_pd = os.listdir('./Datos/PD')
# Set the parameters by cross-validation

clasificadores = []

print("LECTURA DE IMAGENES")

print("Leyendo pacientes enfermos")

for i in range(len(imagenes_pd)):
    imagenes_pd[i] = './Datos/PD/' + imagenes_pd[i]
    paciente = Sujeto(imagenes_pd[i],0)
    pacientes.append(paciente)

imagenes_control = os.listdir('./Datos/Control')

print("Leyendo pacientes control")

for i in range(len(imagenes_control)):
    imagenes_control[i] = './Datos/Control/' + imagenes_control[i]
    paciente = Sujeto(imagenes_control[i],1)
    pacientes.append(paciente)


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
    if pacientes[i].get_label() == 0:
        if train_enfermos < 114:
            train_enfermos = train_enfermos + 1
            pacientes_training.append(pacientes[i])
        else:
            test_enf = test_enf + 1
            pacientes_test.append(pacientes[i])
    else:
        if train_sanos < 55:
            train_sanos = train_sanos + 1
            pacientes_training.append(pacientes[i])
        else:
            test_sanos = test_sanos + 1
            pacientes_test.append(pacientes[i])

dataset_final = []
target_final = []
Y_shape = pacientes_training[0].get_shape(1)

for i in range(Y_shape):
    print("Slice ", i)
    dataset = []
    # *_t para pacientes del test y configurar el dataset de la segunda parte
    dataset_t = []
    target = []
    target_t = []
    np.random.shuffle(pacientes_training)
    np.random.shuffle(pacientes_test)
    for j in range(len(pacientes_training)):
        slice = pacientes_training[j].get_slice(1,i)
        label = pacientes_training[j].get_label()
        print(slice.shape)
        """
        row = pacientes_training[j].get_wave2D(slice,'bior3.3',2)
        dataset.append(row)
        target.append(label)
        """

    for k in range(len(pacientes_test)):
        slice = pacientes_test[k].get_slice(1,i)
        label = pacientes_test[k].get_label()
        print(slice.shape)
        """
        row =  pacientes_test[k].get_wave2D(slice,'bior3.3',2)
        dataset_t.append(row)
    dataset_t = np.array(dataset_t)

    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    dataset_t = scaler.fit_transform(dataset_t)
    X_train = dataset
    y_train = target
    pca = PCA(n_components = 0.95, svd_solver = 'full')
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    pca2 = PCA(n_components = 0.95, svd_solver = 'full')
    pca2.fit(dataset_t)
    dataset_t = pca.transform(dataset_t)
    for i in range(len(pacientes_test)):
        target_final.append(pacientes_test[i].get_label())
    target_final = np.array(target_final)
    pred = deep_learning(X_train,y_train,dataset_t,target_final)
    dataset_final.append(pred)


dataset_final = np.array(dataset_final)
dataset_final = dataset_final.transpose()
X_train, X_test, y_train, y_test = train_test_split(dataset_final,target_final, test_size = 0.2,stratify = target_final)
lr = LogisticRegressionCV(cv=10,multi_class='multinomial').fit(X_train,y_train)
sc_training = lr.score(X_train,y_train)
sc_test = lr.score(X_test,y_test)
params_lr = lr.get_params()
name_scores = "scores.txt"
name_params = "params.txt"
name_prob = "probs.txt"

scores = open(name_scores,"a")
scores.write("Score para training: ")
scores.write(str(sc_training))
scores.write("\n")
scores.write("Score para test: ")
scores.write(str(sc_test))
scores.close()

params = open(name_params,"a")
params.write(str(params_lr))
params.close()

mat = lr.coef_
with open(name_prob,'w') as f:
        for line in mat:
            f.write(str(line))
            f.write("\n")
"""
