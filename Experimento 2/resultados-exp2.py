import numpy as np
import matplotlib.pyplot as plt

accuracy_C1 = np.loadtxt("C12acc2.out",delimiter=',',dtype=np.float64)
accuracy_C2 = np.loadtxt("C22acc2.out",delimiter=',',dtype=np.float64)
accuracy_W = np.loadtxt("PD2acc2.out",delimiter=',',dtype=np.float64)

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
plt.title("Accuracy-Slice C1-C2-W")
plt.legend()
plt.show()


