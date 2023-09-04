
%	Script for "Model-Resolution based Deconvolution for Improved Quantitative Susceptibility Mapping"
%   NMR in Biomedicine (2023) [Accepted for Publication].
%   Data Credits:
%   [1] Lai et al., Learned Proximal Networks for Quantitative Susceptibility Mapping, MICCAI, 2020.
%   [https://github.com/Sulam-Group/LPCNN]
%   Metric Credits:
%   [2] Langkammer et al., Quantitative susceptibility mapping: report from the 2016 reconstruction challenge, 
%   Magnetic Resonance in Medicine, 2018.
%   linear dipole inversion
%   [3]Milovic C et al., Fast nonlinear susceptibility inversion with variational regularization. 
%   Magnetic resonance in medicine. 2018.

clc; 
clear all; close all; addpath('./'); format short;

DF   =@(x) 1/((x.*conj(x) + 1e-10).^(1/2)); % TV diffusivity function
drv3 =@(x) cat(4,x([2:end,end],:,:)-x,x(:,[2:end,end],:)-x,x(:,:,[2:end,end])-x);% deriv


%%-------------------------------------------------------------------------
% create dipole kernel
%%-------------------------------------------------------------------------
 
N =[224 224 126];

% Define Dipole kernel in Fourier Domain

center = N/2 + 1;
spatial_res = [1 1 1];
[ky,kx,kz] = meshgrid(-N(1)/2:N(1)/2-1, -N(2)/2:N(2)/2-1, -N(3)/2:N(3)/2-1);
kx = (kx / max(abs(kx(:)))) / spatial_res(1);
ky = (ky / max(abs(ky(:)))) / spatial_res(2);
kz = (kz / max(abs(kz(:)))) / spatial_res(3);

% Compute magnitude of kernel and perform fftshift
k2 = kx.^2 + ky.^2 + kz.^2;
kernel = 1/3 - (kz.^2 ./ (k2 + eps)); % Z is the B0-direction
kernel = fftshift(kernel);


lambda = 0.001;

%--------------------------------------------------------------------------      
load data


% msk = single(msk);
step_size = 1;      % gradient descent step size    

%%-------------------------------------------------------------------------
% TKD recon
%%-------------------------------------------------------------------------

kthre =0.22;
kernel_inv = ones(N)/kthre; 
kernel_inv( abs(kernel) > kthre ) = 1 ./ kernel(abs(kernel) > kthre);

chi_tkd1 = real( ifftn( kernel_inv.* fftn(phs) ) ) .* msk; 

tkd_metrics =compute_rmse(chi_tkd1.*msk,cos.*msk);

imagesc3d2(chi_tkd1, N/2, 1, [90,90,-90], [-0.10,0.14], [], 'TKD ')
%%-------------------------------------------------------------------------
% MR-TKD recon
%%-------------------------------------------------------------------------

M = kernel.*kernel_inv;

chi_mrtkd = real(ifftn (M.*fftn(chi_tkd1))) .* single(msk);
mrtkd_metrics =compute_rmse(chi_mrtkd.*msk,cos.*msk);

imagesc3d2(chi_mrtkd, N/2, 2, [90,90,-90], [-0.10,0.14], [], 'MR-TKD ')

disp('Metrics TKD   Metrics MR-TKD ')
closedform_metrics = [tkd_metrics mrtkd_metrics]

%%-------------------------------------------------------------------------
% LDI
%%-------------------------------------------------------------------------

params = [];
params.maxOuterIter = 8;
params.N = N;
params.K = kernel;
params.input = phs.*msk;
params.spatial_res = spatial_res;
params.weight = ones(size(msk)).*msk;
params.mask_use = msk;
params.phs_scale=1;
params.chi_cosmos=cos;    

outl = ldi(params); 
    
chi_ldi = real(outl.x) .*msk;

ldi_metrics =compute_rmse(chi_ldi.*msk,cos.*msk);

imagesc3d2(chi_ldi, N/2, 3, [90,90,-90], [-0.10,0.14], [], 'DI ')
%%-------------------------------------------------------------------------
% MR-TKD-iterative
%%-------------------------------------------------------------------------


x = zeros(size(chi_tkd1));

alp=.2;
for k=1:8
    
    x=x-alp*M.*(M.*x-fftn(chi_tkd1));
    chi_mrtkd1 = real(ifftn(x).*single(msk));
    
end

mrtkd1_metrics =compute_rmse(chi_mrtkd1.*msk,cos.*msk);

imagesc3d2(chi_mrtkd1, N/2, 4, [90,90,-90], [-0.10,0.14], [], 'MR-iterative ')

disp('Metrics DI   Metrics MR-iterative ')
iterative_metrics = [ldi_metrics mrtkd1_metrics]

%%-------------------------------------------------------------------------
% LDI-TV
%%-------------------------------------------------------------------------

params = [];
params.maxOuterIter = 9;
params.N = N;
params.K = kernel;
params.input = phs.*msk;
params.spatial_res = spatial_res;
params.weight = ones(size(msk)).*msk;
params.mask_use = msk;
params.phs_scale=1;
params.chi_cosmos=cos;    


outlt = ldi_tv(params); 

chi_ldit = real(outlt.x) .*msk;

ldit_metrics =compute_rmse(chi_ldit.*msk,cos.*msk);
 
imagesc3d2(chi_ldit, N/2, 5, [90,90,-90], [-0.10,0.14], [], 'DI-TV ')

gamma = 0.0001;

%%-------------------------------------------------------------------------
% MR-TKD-Sparsity
%%-------------------------------------------------------------------------

 x = zeros(size(chi_tkd1));
 grad_prev=0;
 alp=0.2;
for k=1:9
     x=x-alp*M.*(M.*x-fftn(chi_tkd1));
     chi_mrtkds = real(ifftn(x).*single(msk));
     % TV denoising ---------------------------------------
     dC       = drv3(chi_mrtkds);
     update2  = gamma*(div3(dC.*DF(dC)));
     chi_mrtkds  = real(chi_mrtkds - update2).*single(msk);      
     x = fftn(chi_mrtkds);
end

mrtkds_metrics =compute_rmse(chi_mrtkds.*msk,cos.*msk);

imagesc3d2(chi_mrtkds, N/2, 6, [90,90,-90], [-0.10,0.14], [], 'MR-Sparsity ')

disp('Metrics DI-TV   Metrics MR-sparsity ')
Sparsity_metrics = [ldit_metrics mrtkds_metrics]

%------------ calling function ---------------
function [Out] = div3(x)
dx   = x(:,:,:,1);
dy  = x(:,:,:,2);
dz  = x(:,:,:,3);
ddx = dx([1,1:end-1],:,:) - dx;
ddy = dy(:,[1,1:end-1],:) - dy;
ddz = dz(:,:,[1,1:end-1]) - dz;
ddx(1,:,:) = -dx(1,:,:); ddx(end,:,:) = dx(end-1,:,:);
ddy(:,1,:) = -dy(:,1,:); ddy(:,end,:) = dy(:,end-1,:);
ddz(:,:,1) = -dz(:,:,1); ddz(:,:,end) = dz(:,:,end-1);
Out        = ddx + ddy + ddz;
end