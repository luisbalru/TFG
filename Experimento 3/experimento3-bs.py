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
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
    '''
    @brief Función encargada de computar y preparar la impresión de la matriz de confusión. Se puede extraer los resultados normalizados o sin normalizar. Basada en un ejemplo de scikit-learn
    @param y_true Etiquetas verdaderas
    @param y_pred Etiquetas predichas
    @param classes Distintas clases del problema (vector)
    @param normalize Booleano que indica si se normalizan los resultados o no
    @param title Título del gráfico
    @param cmap Paleta de colores para el gráfico
    '''
    if not title:
        if normalize:
            title = 'Matriz de confusión normalizada'
        else:
            title = 'Matriz de confusión sin normalizar'

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    # Clases
    classes = [0,1,2,3,4,5,6,7,8,9]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalizar')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Etiquetas verdaderas',
           xlabel='Etiquetas predichas')

    # Rotar las etiquetas para su posible lectura
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Creación de anotaciones
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

pacientes = []
imagenes_pd = os.listdir('./Datos/PD')
# Set the parameters by cross-validation


C_range = np.logspace(-2,11,13)
gamma_range = np.logspace(-9,3,13)
#C_range = np.logspace(-5, 15, 11)
#gamma_range = np.logspace(3, -15, 10)
param_grid = dict(gamma=gamma_range, C=C_range)
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

scores = ['precision']
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

best_slices = [11, 1, 78, 162, 27, 50, 149, 82, 183, 23]

for i in best_slices:
    print("Slice ", i)
    dataset = []
    # *_t para pacientes del test y configurar el dataset de la segunda parte
    dataset_t = []
    target = []
    target_t = []
    np.random.shuffle(pacientes)
    np.random.shuffle(pacientes)
    for j in range(len(pacientes)):
        slice = pacientes[j].get_slice(1,i)
        label = pacientes[j].get_label()
        row = pacientes[j].get_wave2D(slice,'bior3.3',2)
        dataset.append(row)
        target.append(label)



scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)
X_train, X_test, y_train, y_test = train_test_split(dataset,target,stratify=target,test_size=0.3,random_state=77145416)
pca = PCA(n_components = 0.95, svd_solver = 'full')
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
pca2 = PCA(n_components = 0.95, svd_solver = 'full')
pca2.fit(X_test)
X_test_pca = pca.transform(X_test)
for score in scores:
    clf = GridSearchCV(SVC(kernel='rbf',class_weight = 'balanced'),param_grid = param_grid, cv=10,scoring = '%s_macro' % score)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    print(clf.score(X_test,y_test))
     # Pintamos la matriz de confusión
    nombres = ["PD","Control"]
    plot_confusion_matrix(y_test, pred, classes=nombres,normalize = False,title='Matriz de confusión')
    plt.show()

    # Pintamos la curva ROC
    print("Área bajo la curva ROC")
    fpr, tpr, threshold = metrics.roc_curve(y_test,pred)
    roc_auc = metrics.auc(fpr,tpr)
    plt.title('Curva ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print(classification_report(y_test,pred,target_names = nombres))
