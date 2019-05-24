# AUTOR: LUIS BALDERAS RUIZ
# TRABAJO FINAL DE GRADO

import nibabel as nib
import pywt

class Sujeto:
    def __init__(self, image_name,group):
        self.nombre_image = image_name
        self.grupo = group
        img = nib.nifti1.load(image_name)
        self.img_data = img.get_fdata()
        self.shape_image = self.img_data.shape


    def get_label(self):
        return self.grupo

    def get_shape(self,plano):
        return self.shape_image[plano]

    def get_slice(self,plano,slice):
        if plano == 0:
            return self.img_data[slice,:,:]
        elif plano == 1:
            return self.img_data[:,slice,:]
        elif plano == 2:
            return self.img_data[:,:,slice]
        else:
            print("Dio un plano que no existe")

    def get_wave2D(self,slice,transform,lev):
        cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(slice,transform,level=lev)
        coeffs = []
        for i in range(len(cA2)):
            for j in range(len(cA2[0])):
                coeffs.append(cA2[i][j])

        return coeffs
