# SCRIPT PARA EL EXPERIMENTO 4
# Elegido sagital y C1 (materia blanca), llevo a cabo un ensemble learner (stacking)
# para elegir las slices más determinantes en el diagnóstico
# Author: Luis Balderas Ruiz

from subjects import Sujeto
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


pacientes = []
imagenes_pd = os.listdir('./Datos/PD')
# Set the parameters by cross-validation

def deep_learning(X_train,y_train,X_test,y_test):
    model = Sequential()

    #add model layers
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(157,136,1)))
    model.add(MaxPooling2D(pool_size=(4, 4))) # 2,2 -> 4,4
    model.add(Dropout(0.5))
    model.add(Conv2D(16, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4))) # 2,2 -> 4,4
    model.add(Dropout(0.5))
    model.add(Conv2D(16, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4))) # 2,2 -> 4,4
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    #compile model using accuracy as a measure of model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    print(model.summary())
    #train model
    model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=100, callbacks=[EarlyStopping(monitor='val_loss')] )
    pred = model.predict(X_test)
    pred_fin = []
    for i in range(len(pred)):
        index = pred[i].argmax()
        pred_fin.append(index)
    return pred_fin



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
        #if train_enfermos < 8:
            train_enfermos = train_enfermos + 1
            pacientes_training.append(pacientes[i])
        else:
            test_enf = test_enf + 1
            pacientes_test.append(pacientes[i])
    else:
        #if train_sanos < 5:
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
    print("Slice ", i, "/", Y_shape)
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
        dataset.append(slice)
        target.append(label)


    for k in range(len(pacientes_test)):
        slice = pacientes_test[k].get_slice(1,i)
        label = pacientes_test[k].get_label()
        dataset_t.append(slice)

    dataset_t = np.array(dataset_t)
    print(dataset_t.shape)

    X_train = dataset
    y_train = target
    X_train = np.array(X_train)
    print(X_train.shape)
    y_train = np.array(y_train)
    X_train = X_train.reshape(169,157,136,1)
    #X_train = X_train.reshape(13,157,136,1)
    dataset_t = dataset_t.reshape(73,157,136,1)
    #dataset_t = dataset_t.reshape(12,157,136,1)
    y_train = to_categorical(y_train)
    target_final = []
    for i in range(len(pacientes_test)):
        target_final.append(pacientes_test[i].get_label())
    target_final = np.array(target_final)
    target_final = to_categorical(target_final)
    pred = deep_learning(X_train,y_train,dataset_t,target_final)
    print(pred)
    dataset_final.append(pred)


dataset_final = np.array(dataset_final)
dataset_final = dataset_final.transpose()
etiquetas = []
for i in range(len(target_final)):
    index = target_final[i].argmax()
    etiquetas.append(index)

etiquetas = np.array(etiquetas)
X_train, X_test, y_train, y_test = train_test_split(dataset_final,etiquetas, test_size = 0.2,stratify = etiquetas)
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
