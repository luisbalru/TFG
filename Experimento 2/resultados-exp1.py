import numpy as np
import matplotlib.pyplot as plt

accuracy_X = np.loadtxt("accuracy_Xslices-d.out",delimiter=',',dtype=np.float64)
accuracy_Y = np.loadtxt("accuracy_Yslices-d.out",delimiter=',',dtype=np.float64)
accuracy_Z = np.loadtxt("accuracy_Zslices-d.out",delimiter=',',dtype=np.float64)

mean_accuracy_X = np.mean(accuracy_X)
mean_accuracy_Y = np.mean(accuracy_Y)
mean_accuracy_Z = np.mean(accuracy_Z)

print(mean_accuracy_X)
print(mean_accuracy_Y)
print(mean_accuracy_Z)

print("Accuracy para X")

plt.plot(range(0,len(accuracy_X)),accuracy_X.tolist())
plt.xlabel("Slice")
plt.ylabel("Accuracy")
plt.title("Accuracy-Slice en el plano axial")
plt.show()

print("Accuracy para Y")

plt.plot(range(0,len(accuracy_Y)),accuracy_Y.tolist())
plt.xlabel("Slice")
plt.ylabel("Accuracy")
plt.title("Accuracy-Slice en el plano coronal")
plt.show()

print("Accuracy para Z")

plt.plot(range(0,len(accuracy_Z)),accuracy_Z.tolist())
plt.xlabel("Slice")
plt.ylabel("Accuracy")
plt.title("Accuracy-Slice en el plano sagital")
plt.show()
