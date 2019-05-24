import numpy as np
import matplotlib.pyplot as plt

accuracy_X = np.loadtxt("accuracy_Xslices-d.out",delimiter=',',dtype=np.float64)
accuracy_Y = np.loadtxt("accuracy_Yslices-d.out",delimiter=',',dtype=np.float64)
accuracy_Z = np.loadtxt("accuracy_Zslices-d.out",delimiter=',',dtype=np.float64)

mean_accuracy_X = np.mean(accuracy_X)
mean_accuracy_Y = np.mean(accuracy_Y)
mean_accuracy_Z = np.mean(accuracy_Z)

print("Accuracy medio X: ", mean_accuracy_X)
print("Accuracy medio Y: ", mean_accuracy_Y)
print("Accuracy medio Z: ", mean_accuracy_Z)

buenosX = 0
buenosY = 0
buenosZ = 0
slices_bX = []
slices_bY = []
slices_bZ = []
positions_X = []
positions_Y = []
positions_Z = []

for i in range(len(accuracy_X)):
	if(accuracy_X[i] >= 0.75):
		buenosX = buenosX+1
		slices_bX.append(accuracy_X[i])
		positions_X.append(i)


for i in range(len(accuracy_Y)):
	if(accuracy_Y[i] >= 0.75):
		buenosY = buenosY+1
		slices_bY.append(accuracy_Y[i])
		positions_Y.append(i)

for i in range(len(accuracy_Z)):
	if(accuracy_Z[i] >= 0.75):
		buenosZ = buenosZ+1
		slices_bZ.append(accuracy_Z[i])
		positions_Z.append(i)

print("RESULTADOS")
print("PLANO X:")
print("Buenas: ", buenosX)
print(slices_bX) 
print(positions_X)

print("PLANO Y:")
print("Buenas: ", buenosY)
print(slices_bY) 
print(positions_Y)

print("PLANO Z:")
print("Buenas: ", buenosZ)
print(slices_bZ)
print(positions_Z) 

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
