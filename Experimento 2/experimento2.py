# SCRIPT PARA EL EXPERIMENTO 2
# Sabiendo que el plano sagital ha sido el más determinante,
# elijo qué tipo de imagen da mejores resultados: whole, materia blanco o gris
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

nombres_PD = ['./Datos/PD','./Datos/PD-C1','./Datos/PD-C2']
nombres_C = ['./Datos/Control','./Datos/Control-C1','./Datos/Control-C2']

C_range = np.logspace(-2,10,13)
gamma_range = np.logspace(-9,3,13)
#C_range = np.logspace(-5, 15, 11)
#gamma_range = np.logspace(3, -15, 10)
param_grid = dict(gamma=gamma_range, C=C_range)

scores = ['recall']

print("LECTURA DE IMAGENES")

for i in range(len(nombres_PD)):

    pacientes = []
    imagenes_pd = os.listdir(nombres_PD[i])

    print("Leyendo pacientes enfermos ", nombres_PD[i])

    for i in range(len(imagenes_pd)):
        imagenes_pd[i] = nombres_PD + imagenes_pd[i]
        paciente = Sujeto(imagenes_pd[i],0)
        pacientes.append(paciente)

    imagenes_control = os.listdir(nombres_C[i])

    print("Leyendo pacientes control")

    for i in range(len(imagenes_control)):
        imagenes_control[i] = nombres_C[i] + imagenes_control[i]
        paciente = Sujeto(imagenes_control[i],1)
        pacientes.append(paciente)

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
            X_train, X_test, y_train, y_test = train_test_split(dataset,target,test_size=0.3,random_state = 77145416)
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

        name = nombres_PD[i]
        name = name[-2:]
        name_ac = name + "acc.out"
        h = open(name,"a")
        h.write(name)
        h.write(str(np.argmax(accuracy_Zslices)))
        h.write(str(accuracy_Zslices[np.argmax(accuracy_Zslices)]))
        accuracy_Zslices = np.array(accuracy_Zslices)
        accuracy_Zslices.tofile(name_ac,sep=",")
