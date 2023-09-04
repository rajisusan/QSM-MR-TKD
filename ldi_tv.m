function out = ldi_tv(params)
% Linear Dipole Inversion with TV regularization

%
% Parameters: params - structure with 
% Required fields:
% params.input: local field map, in radians
% params.K: dipole kernel in the frequency space
% Optional fields:
% params.alpha: regularization weight (use small values for stability)
% params.maxOuterIter: maximum number of iterations
% params.weight: data fidelity spatially variable weight (recommended = magnitude_data).
% params.tau: gradient descent rate
% params.precond: preconditionate solution (for stability)
% params.isShowIters: show intermediate results for each iteration.
% params.isGPU: activate GPU acceleration (default = true).
%
% Output: out - structure with the following fields:
% out.x: calculated susceptibility map, in radians
% out.time: total elapsed time (including pre-calculations)
%
% Last modified by Carlos Milovic in 2021.10.12

% Modified by Raji Susan Mathew in 2023-09-04.


    if isfield(params,'alpha')
         alpha = params.alpha;
    else
        alpha = 1E-6;
    end
    
    if isfield(params,'tau')
         tau = params.tau;
    else
        tau = 1.0;
    end
    
    if isfield(params,'N')
         N = params.N;
    else
        N = size(params.input);
    end

    if isfield(params,'maxOuterIter')
        num_iter = params.maxOuterIter;
    else
        num_iter = 500;
    end
    
    if isfield(params,'weight')
        weight = params.weight;
    else
        weight = ones(N);
    end
    weight = weight.*weight;
    
    

    if isfield(params,'isShowIters')
        isShowIters = params.isShowIters;
    else
        isShowIters = false;
    end
    
    if isfield(params,'isPrecond')
        isPrecond = params.isPrecond;
    else
        isPrecond = false;
    end
    
    if ~isfield(params,'isGPU')
        isGPU = true;
    else
        isGPU = params.isGPU;
    end
    
    if isfield(params,'GT')
        isGT = true;
        GT = params.GT;
        if isfield(params,'mask')
            mask = params.mask;
        else
            mask = single(weight > 0);
        end
        if isfield(params,'scale')
            phs_scale = params.scale;
        else
            phs_scale = 1.0;
        end
        
    else
        isGT = false;
    end
    
    
    if isPrecond   
        x =params.weight.*params.input;
    else
        x = zeros(N, 'single');
    end

    kernel = params.K;
    phase = params.input;

try
if isGPU 
    disp('GPU enabled');
    phase = gpuArray(phase);
    kernel = gpuArray(kernel);
    x = gpuArray(x);

    weight = gpuArray(weight);
    alpha = gpuArray(alpha);
    tau = gpuArray(tau);
    num_iter = gpuArray(num_iter);
    
    if isGT
        GT = gpuArray(GT);
        mask = gpuArray(mask);
        phs_scale = gpuArray(phs_scale);
    end
end
catch
    disp('WARNING: GPU disabled');
end


DF   =@(x) 1/((x.*conj(x) + 1e-10).^(1/2)); % TV diffusivity function
drv3 =@(x) cat(4,x([2:end,end],:,:)-x,x(:,[2:end,end],:)-x,x(:,:,[2:end,end])-x);% deriv

gamma = 0.0001; %0.0001  


for t = 1:num_iter
%     t
    % update x : susceptibility estimate
    x_prev = x;
    phix = susc2field(kernel,x);
    x = x_prev - tau*susc2field( conj(kernel),weight.*(phix-phase ) ) ;

    % diffusion denoising ----------------------------------------
    
    dC       = drv3(x);
    update2  = gamma*(div3(dC.*DF(dC)));
    x  = real(x - update2);

    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    
    out.update(t) = gather(x_update);
end


if isGPU
    out.x = gather(x);
    out.iter = gather(t);
else
    out.x = x;
    out.iter = t;
end



end
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