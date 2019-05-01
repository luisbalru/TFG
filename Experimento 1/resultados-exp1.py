import numpy as np

accuracy_X = np.loadtxt("accuracy_Xslices-ns.out",delimiter=',',dtype=np.float64)
accuracy_Y = np.loadtxt("accuracy_Yslices-ns.out",delimiter=',',dtype=np.float64)
accuracy_Z = np.loadtxt("accuracy_Zslices-ns.out",delimiter=',',dtype=np.float64)

mean_accuracy_X = np.mean(accuracy_X)
mean_accuracy_Y = np.mean(accuracy_Y)
mean_accuracy_Z = np.mean(accuracy_Z)

print(mean_accuracy_X)
print(mean_accuracy_Y)
print(mean_accuracy_Z)
