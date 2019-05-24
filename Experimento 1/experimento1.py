# SCRIPT PARA EL EXPERIMENTO 1
# Determinar qué plano es más determinante en la clasificación
# Tratamiento por slices
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


C_range = np.logspace(-2,10,13)
gamma_range = np.logspace(-9,3,13)
#C_range = np.logspace(-5, 15, 11)
#gamma_range = np.logspace(3, -15, 10)
param_grid = dict(gamma=gamma_range, C=C_range)
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

scores = ['recall']


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

print("X (Plano 0)")


X_shape = pacientes[0].get_shape(0)
accuracy_Xslices = []

for j in range(X_shape):
    print("Slice ", j)
    dataset = []
    target = []
    np.random.shuffle(pacientes)
    for i in range(len(pacientes)):
        slice = pacientes[i].get_slice(0,j)
        label = pacientes[i].get_label()
        row = pacientes[i].get_wave2D(slice,'bior3.3',2)
        dataset.append(row)
        target.append(label)

    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    X_train, X_test, y_train, y_test = train_test_split(dataset,target,test_size=0.3,random_state = 77145416,stratify = target)
    y_train = np.array(y_train)
    pca = PCA(n_components = 0.95, svd_solver = 'full')
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    clf = 0
    for score in scores:
        clf = GridSearchCV(SVC(),param_grid=param_grid , cv=10,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

    mean_score_slice = 0
    for k in range(0,10):
        indices = np.random.choice(len(X_train),size=len(X_train),replace=False)
        dataset_shuffle = X_train[indices]
        target_shuffle = y_train[indices]
        svm = SVC(C = clf.best_params_['C'],gamma = clf.best_params_['gamma'])
        svm.fit(dataset_shuffle,target_shuffle)
        y_true, y_pred = y_test, svm.predict(X_test)
        mean_score_slice = mean_score_slice + accuracy_score(y_true,y_pred)

    mean_score_slice = mean_score_slice/10
    accuracy_Xslices.append(mean_score_slice)

f = open('planoX-d.txt','a')
f.write('X (Plano 0)')
f.write(str(np.argmax(accuracy_Xslices)))
f.write(str(accuracy_Xslices[np.argmax(accuracy_Xslices)]))
f.close()
accuracy_Xslices = np.array(accuracy_Xslices)
accuracy_Xslices.tofile('accuracy_Xslices-d.out',sep=",")

print("Y (Plano 1)")


Y_shape = pacientes[0].get_shape(1)
accuracy_Yslices = []

for j in range(Y_shape):
    print("Slice ", j)
    dataset = []
    target = []
    np.random.shuffle(pacientes)
    for i in range(len(pacientes)):
        slice = pacientes[i].get_slice(1,j)
        label = pacientes[i].get_label()
        row = pacientes[i].get_wave2D(slice,'bior3.3',2)
        dataset.append(row)
        target.append(label)

    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    X_train, X_test, y_train, y_test = train_test_split(dataset,target,test_size=0.3,random_state = 77145416,stratify = target)
    y_train = np.array(y_train)
    pca = PCA(n_components = 0.95, svd_solver = 'full')
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    clf = 0
    for score in scores:
        clf = GridSearchCV(SVC(), param_grid=param_grid, cv=10,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

    mean_score_slice = 0
    for k in range(0,10):
        indices = np.random.choice(len(X_train),size=len(X_train),replace=False)
        dataset_shuffle = X_train[indices]
        target_shuffle = y_train[indices]
        svm = SVC(C = clf.best_params_['C'],gamma = clf.best_params_['gamma'])
        svm.fit(dataset_shuffle,target_shuffle)
        y_true, y_pred = y_test, svm.predict(X_test)
        mean_score_slice = mean_score_slice + accuracy_score(y_true,y_pred)

    mean_score_slice = mean_score_slice/10
    accuracy_Yslices.append(mean_score_slice)

g = open("planoY-d.txt","a")
g.write("Y (Plano 1)")
g.write(str(np.argmax(accuracy_Yslices)))
g.write(str(accuracy_Yslices[np.argmax(accuracy_Yslices)]))
accuracy_Yslices = np.array(accuracy_Yslices)
accuracy_Yslices.tofile('accuracy_Yslices-d.out',sep=",")

print("Z (Plano 2)")

Z_shape = pacientes[0].get_shape(2)
accuracy_Zslices = []

for j in range(Z_shape):
    print("Slice ", j)
    dataset = []
    target = []
    np.random.shuffle(pacientes)
    for i in range(len(pacientes)):
        slice = pacientes[i].get_slice(2,j)
        label = pacientes[i].get_label()
        row = pacientes[i].get_wave2D(slice,'bior3.3',2)
        dataset.append(row)
        target.append(label)


    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    X_train, X_test, y_train, y_test = train_test_split(dataset,target,test_size=0.3,random_state = 77145416,stratify = target)
    y_train = np.array(y_train)
    pca = PCA(n_components = 0.95, svd_solver = 'full')
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    clf = 0
    for score in scores:
        clf = GridSearchCV(SVC(), param_grid=param_grid, cv=10,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

    mean_score_slice = 0
    for k in range(0,10):
        indices = np.random.choice(len(X_train),size=len(X_train),replace=False)
        dataset_shuffle = X_train[indices]
        target_shuffle = y_train[indices]
        svm = SVC(C = clf.best_params_['C'],gamma = clf.best_params_['gamma'])
        svm.fit(dataset_shuffle,target_shuffle)
        y_true, y_pred = y_test, svm.predict(X_test)
        mean_score_slice = mean_score_slice + accuracy_score(y_true,y_pred)

    mean_score_slice = mean_score_slice/10
    accuracy_Zslices.append(mean_score_slice)

h = open("planoZ-d.txt","a")
h.write("Z (Plano 2)")
h.write(str(np.argmax(accuracy_Zslices)))
h.write(str(accuracy_Zslices[np.argmax(accuracy_Zslices)]))
accuracy_Zslices = np.array(accuracy_Zslices)
accuracy_Zslices.tofile('accuracy_Zslices-d.out',sep=",")
