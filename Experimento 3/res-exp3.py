import numpy as np
from numpy import loadtxt

lines = loadtxt("probs.txt",delimiter=",")
mayores = []
menores = []
for i in range(0,10):
    mayor = np.argmax(lines)
    menor = np.argmin(lines)
    mayores.append(mayor)
    menores.append(menor)
    lines[mayor] = 0
    lines[menor] = 0

print("Posiciones mayores")
print(mayores)
print("Posicion menores")
print(menores)


import nibabel as nib
import matplotlib.pyplot as plt

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        if i==1:
            for j in range(len(menores)):
                axes[i].plot([0,150],[mayores[j],mayores[j]],'C3')
        if i==0:
            for j in range(len(menores)):
                axes[i].plot([mayores[j],mayores[j]],[0,150],'C3')

img = nib.load('wSAG_FSPGR_3D_SAG_FSPGR_3D_20130502124054_6.nii')
img_data = img.get_fdata()
print(img_data.shape)

slice_0 = img_data[50,:,:]
slice_1 = img_data[:,:,50]
show_slices([slice_0,slice_1])
plt.show()
