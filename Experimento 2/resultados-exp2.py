import numpy as np
import matplotlib.pyplot as plt

accuracy_C1 = np.loadtxt("C13accY3.out",delimiter=',',dtype=np.float64)
accuracy_C2 = np.loadtxt("C23accY3.out",delimiter=',',dtype=np.float64)
accuracy_W = np.loadtxt("PD3accY3.out",delimiter=',',dtype=np.float64)

cC1 = np.loadtxt("componentsY-C13.txt", delimiter=',',dtype=np.int64)
cC2 = np.loadtxt("componentsY-C23.txt", delimiter=',',dtype=np.int64)
cPD = np.loadtxt("componentsY-PD3.txt", delimiter=',',dtype=np.int64)

dif = cC2 - cC1

plt.plot(range(0,len(dif)),dif.tolist())
plt.xlabel("Slice")
plt.ylabel("Diferencia de accuracy")
plt.title("Diferencia de accuracy para la diferencia C2-C1 en cada slice")
plt.plot(range(0,len(dif)),np.zeros(len(dif)))
plt.show()


mean_accuracy_C1 = np.mean(accuracy_C1)
mean_accuracy_C2 = np.mean(accuracy_C2)
mean_accuracy_W = np.mean(accuracy_W)

print(mean_accuracy_C1)
print(mean_accuracy_C2)
print(mean_accuracy_W)

print("Accuracy para X")

plt.plot(range(0,len(accuracy_C1)),accuracy_C1.tolist(), label = "C1")
plt.plot(range(0,len(accuracy_C2)),accuracy_C2.tolist(), label = "C2")
#plt.plot(range(0,len(accuracy_W)),accuracy_W.tolist(),label = "W")
plt.xlabel("Slice")
plt.ylabel("Accuracy")
plt.title("Accuracy-Slice C1-C2")
plt.legend()
plt.show()

print("Componentes principales")

plt.plot(range(0,len(cC1)),cC1.tolist())
plt.xlabel("Slice")
plt.ylabel("Número de componentes")
plt.title("Número de componentes principales por slice materia gris")
plt.show()

plt.plot(range(0,len(cC2)),cC2.tolist())
plt.xlabel("Slice")
plt.ylabel("Número de componentes")
plt.title("Número de componentes principales por slice materia blanca")
plt.show()

plt.plot(range(0,len(cPD)),cPD.tolist())
plt.xlabel("Slice")
plt.ylabel("Número de componentes")
plt.title("Número de componentes principales por slice cerebro completo")
plt.show()

