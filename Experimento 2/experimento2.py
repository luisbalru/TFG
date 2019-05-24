# SCRIPT PARA EL EXPERIMENTO 2
# Sabiendo que el plano coronal ha sido el más determinante,
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

nombres_PD = ['./Datos/PD/','./Datos/PD-C1/','./Datos/PD-C2/']
nombres_C = ['./Datos/Control/','./Datos/Control-C1/','./Datos/Control-C2/']

C_range = np.logspace(-2,11,13)
gamma_range = np.logspace(-9,3,13)
#C_range = np.logspace(-5, 15, 11)
#gamma_range = np.logspace(3, -15, 10)
param_grid = dict(gamma=gamma_range, C=C_range)
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

scores = ['recall']

print("LECTURA DE IMAGENES")

for i in range(len(nombres_PD)):

    pacientes = []
    imagenes_pd = os.listdir(nombres_PD[i])

    print("Leyendo pacientes enfermos ", nombres_PD[i])

    for j in range(len(imagenes_pd)):
        imagenes_pd[j] = nombres_PD[i] + imagenes_pd[j]
        paciente = Sujeto(imagenes_pd[j],0)
        pacientes.append(paciente)

    imagenes_control = os.listdir(nombres_C[i])

    print("Leyendo pacientes control")

    for j in range(len(imagenes_control)):
        imagenes_control[j] = nombres_C[i] + imagenes_control[j]
        paciente = Sujeto(imagenes_control[j],1)
        pacientes.append(paciente)

    print("Y (Plano 1)")

    Y_shape = pacientes[0].get_shape(1)
    accuracy_Yslices = []
    components_Yslices = []

    for k in range(Y_shape):
        print("Slice ", k)
        dataset = []
        target = []
        np.random.shuffle(pacientes)
        for l in range(len(pacientes)):
            slice = pacientes[l].get_slice(1,k)
            label = pacientes[l].get_label()
            row = pacientes[l].get_wave2D(slice,'bior3.3',2)
            dataset.append(row)
            target.append(label)


        scaler = StandardScaler()
        dataset = scaler.fit_transform(dataset)
        X_train, X_test, y_train, y_test = train_test_split(dataset,target,test_size=0.3,random_state = 77145416, stratify = target)
        y_train = np.array(y_train)
        pca = PCA(n_components = 0.95, svd_solver = 'full')
        pca.fit(X_train)
        components_Yslices.append(pca.n_components_)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        clf = 0
        for score in scores:
            clf = GridSearchCV(SVC(kernel='rbf',class_weight='balanced'), param_grid=param_grid, cv=10,
                               scoring='%s_macro' % score)
            clf.fit(X_train, y_train)

        mean_score_slice = 0
        for m in range(0,10):
            indices = np.random.choice(len(X_train),size=len(X_train),replace=False)
            dataset_shuffle = X_train[indices]
            target_shuffle = y_train[indices]
            svm = SVC(C = clf.best_params_['C'],gamma = clf.best_params_['gamma'])
            svm.fit(dataset_shuffle,target_shuffle)
            y_true, y_pred = y_test, svm.predict(X_test)
            mean_score_slice = mean_score_slice + accuracy_score(y_true,y_pred)

        mean_score_slice = mean_score_slice/10
        accuracy_Yslices.append(mean_score_slice)

    name = nombres_PD[i]
    name = name[-3:-1] + "3"
    c = "componentsY-" + name +".txt"
    name_ac = name + "accY3.out"
    h = open(name,"a")
    h.write(name)
    h.write(str(np.argmax(accuracy_Yslices)))
    h.write(str(accuracy_Yslices[np.argmax(accuracy_Yslices)]))
    accuracy_Yslices = np.array(accuracy_Yslices)
    components_Yslices = np.array(components_Yslices)
    accuracy_Yslices.tofile(name_ac,sep=",")
    components_Yslices.tofile(c,sep=",")
