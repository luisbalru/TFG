function [] = ppmi_normal(file)
addpath(genpath('NIfTI 20140122'));
addpath(genpath('spm12'));
% This function just normalizes the input file to MNI space, in a bounding
% box of 157x189x136 voxels, segments image into gray and white matter,
% and store the results in w*.nii wc1*.nii and wc2*.nii
tic
%Create the file paths to GM and WM
[path,name,ext] = fileparts(file);

out = strcat(path,'/out');
c1 = fullfile(path,[strcat('c1',name) ext]);
c2 = fullfile(path,[strcat('c2',name) ext]);
% Path to mats:
filename = file(1:end-4);
toc

tic
%Image segmentation
start = strcat(file);
disp(start)
results = spm_preproc(start);
toc


tic
% Generation of the normalization parameters
[po, pin] = spm_prep2sn(results);
% Inverse parameters storage
VG = pin.VG;
VF = pin.VF;
Tr = pin.Tr;
Affine = pin.Affine;

flags = pin.flags;
save(fullfile(strcat(filename,'_seg_inv_sn.mat')), '-V6','VG','VF','Tr','Affine','flags');
toc
% Forward parameters storage
VG = po.VG;
VF = po.VF;
Tr = po.Tr;
Affine = po.Affine;
flags = po.flags;
transformationParameters = strcat(filename,'_seg_sn.mat');
fnam = fullfile(transformationParameters);
save(fnam,'-V6','VG','VF','Tr','Affine','flags');

%Segmentation with output parameters
spm_preproc_write(po);
%Setup of the spatial normalization parameters
defaults.normalise.write.preserve = reshape(double(0), [1,1]);
defaults.normalise.write.bb = reshape(double([-78 78 -112 76 -50 85]),[2,3]);
defaults.normalise.write.vox = reshape(double([1 1 1]), [1,3]);
defaults.normalise.write.interp = reshape(double([1]), [1,1]);
defaults.normalise.write.wrap = reshape(double([0 0 0]), [1,3]);
tic

% Image normalization
% Normal image
spm_write_sn(file, transformationParameters,defaults.normalise.write);
% Grey matter image
spm_write_sn(c1,transformationParameters, defaults.normalise.write);
% White matter image
spm_write_sn(c2, transformationParameters, defaults.normalise.write);
toc

end
