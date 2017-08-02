% Divide and Update Algorithm implementation from:
% J. Hagemann and T. Salditt: 
% "Divide and Update: Towards Single-Shot Object and Probe Retrieval
%  for Near-Field Holography"
% published in Optics Express
% 
% MIT License
% 
% Copyright (c) 2017 Johannes Hagemann
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

function [result_P, result_O, EW, errors, gaps] =...
    divide_and_update(inputEW, inputP, guess_O, guess_P, iterations, F, p)
% inputEW, inputO:  measured amplitudes
% guess_O, guess_P: complex valued guesses for O and P
% iterations:       number of iterations
% F:                Fresnel number(s)
% p:                parameter object

h = waitbar(0, 'progress');

% Propagator objects
if(isfield(p,'oversample') == 0)
    warning('using default oversampling factor = 1.');
    oversample = 1;
else
    oversample = p.oversample;
end

%prepare algorithm
waitbar(0, h, ...
    sprintf('Preparing Propagators...'));
num_planes = numel(F);

%PropagatorGPU's
if num_planes > 1 % for multi plane reconstructions cf Apendix
    for ii =1:num_planes
        props{ii} = PropagatorGPU(F(ii), F(ii), size(guess_O,2), size(guess_O,1), oversample);
    end
    
    for ii = 1:num_planes
        inv_props{ii} = PropagatorGPU(-F(ii), -F(ii), size(guess_O,2), size(guess_O,1), oversample);
    end
else 
    for ii = 1:2
        props{ii} = PropagatorGPU(F(1), F(1), size(guess_O,2), size(guess_O,1), oversample);
    end
    
    for ii = 1:2
        inv_props{ii} = PropagatorGPU(-F(1), -F(1), size(guess_O,2), size(guess_O,1), oversample);
    end
    
    
end

% errors
errors = (zeros(iterations, 2));

% errors
gaps = (zeros(iterations, 2));

% use guesses as start points
O = gpuArray(guess_O);
P = gpuArray(guess_P);
inputEW = gpuArray(inputEW);
inputP = gpuArray(inputP);
EW = P .* O;

% count intermediates
intermediate_result_counter = 1;

if p.do_wavelet_filter == 1
    num_scales = 4;
    disp('generatiang shearlet system')
    tic
    sys = SLgetShearletSystem2D(1, p.rec_height, p.rec_width, num_scales);
    toc
end

for ii = 1:iterations
    waitbar(ii / iterations, h, ...
        sprintf('%d / %d',ii, iterations));
    
    EW_old = EW;
    P_old = P;
    O_old = O;
    
    EW = props{2}.propTF(EW);
    P = props{1}.propTF(P);
    
    if(isfield(p,'do_errors') == 1)
        if p.do_errors == 1
            if ii > 1
                tmp = mid(inputEW, p) - abs(mid(EW, p));
                errors(ii, 1) = gather(sum(abs(tmp(:)).^2)./ (p.height*p.width));
                
                tmp = mid(inputP, p) - abs(mid(P, p));
                errors(ii, 2) = gather(sum(abs(tmp(:)).^2)./ (p.height*p.width));
            else
                errors(ii, 1) = nan;
                errors(ii, 2) = nan;
            end
        end
    end
    
    %magnitude constraint, input is expected as ampl!
    if(p.cropping)
        %modified magnitude projection for cropped holograms
        EW(p.detector_roi > 1) = inputEW(p.detector_roi > 1) .* exp(1i .* angle(EW(p.detector_roi > 1)));
        P(p.detector_roi > 1) = inputP(p.detector_roi > 1) .* exp(1i .* angle(P(p.detector_roi > 1)));
    else
        EW = inputEW .* exp(1i .* angle(EW));
        P = inputP .* exp(1i .* angle(P));
    end
    
    % back at sample plane
    P_M_EW = inv_props{2}.propTF(EW);
    P_M_P= inv_props{1}.propTF(P);
    
    %separate fields
    O = (P_M_EW ./ P_M_P);
    
    if(isfield(p,'use_support') == 1)
        if p.use_support == 1
            if p.do_wavelet_filter == 1
                % removed shearlets
                base_fct_removed = [3 8 15 30 31 33 38];
                
                for base_idx = base_fct_removed
                    coeffsO(:,:, base_idx) = 0.8 .* coeffsO(:,:,base_idx);
                end
                % -> pure phase object
                O =  exp(1i * SLshearrec2D(coeffsO, sys)); 
                              
            end
            O = exp(1i .* angle(O)); %pure phase object
            O(angle(O) > 0 | ~p.supp) = exp(1i .* 0); %negative phase & support

        else
            O = exp(1i .* angle(O)); %pure phase object
            O(angle(O) > 0 | ~p.supp) = exp(1i .* 0); %negative phase & support
        end
    end
    
    
    if(p.probe_filter)
       P = (P_M_EW ./ O + P_M_P)/2;
        amp = imgaussfilt(abs(P), p.gauss_filt_probe/(2*sqrt(2*log(2))));
        pha = imgaussfilt(angle(P), p.gauss_filt_probe/(2*sqrt(2*log(2))));
        P = amp .* exp(1i * pha);
    else
        P = (P_M_EW ./ O + P_M_P)/2;
    end
    
    % measure |x_(n-1) - x_n|²
    if(isfield(p,'do_convergence_hist') == 1)
        if p.do_convergence_hist == 1
            if ii > 1
                tmp = mid(O_old, p) - (mid(O, p));
                gaps(ii, 1) = gather(sum(abs(tmp(:)).^2)./ (p.height*p.width));
                
                tmp = mid(P_old, p) - (mid(P, p));
                gaps(ii, 2) = gather(sum(abs(tmp(:)).^2)./ (p.height*p.width));
            else
                gaps(ii, 1) = nan;
                gaps(ii, 2) = nan;
            end
        end
    end
    
    % new exit wave
    EW = (P .* O);
    
    % measure |P_S(P_M(x_n)) - P_M(x_n)|²
    if(isfield(p,'do_gaps') == 1)
        if p.do_gaps== 1
            if ii > 1
                tmp = mid(EW, p) - (mid(P_M_EW, p));
                gaps(ii, 1) = gather(sum(abs(tmp(:)).^2)./ (p.height*p.width));
                
                tmp = mid(P, p) - (mid(P_M_P, p));
                gaps(ii, 2) = gather(sum(abs(tmp(:)).^2)./ (p.height*p.width));
            else
                gaps(ii, 1) = nan;
                gaps(ii, 2) = nan;
            end
        end
    end
    
    if(isfield(p,'do_intermediate_results') == 1)
        if p.do_intermediate_results == 1
            if mod(ii, p.do_intermediate_results_interval) == 0
                result_P{intermediate_result_counter} = gather(P);
                result_O{intermediate_result_counter} = gather(O);
                intermediate_result_counter = intermediate_result_counter + 1;
                
                figure
                tmp = mid(O,p);
                imagesc(angle(tmp))
                title(sprintf('phase rec iteration %i', ii))
                drawnow
                
            end
        end
    end    
end % end iterations



if(isfield(p,'do_last_projection') == 1)
    if p.do_last_projection == 1
        EW = props{2}.propTF(EW);
        P = props{1}.propTF(P);
        EW = inputEW .* exp(1i .* angle(EW));
        P = inputP .* exp(1i .* angle(P));
        
        % back at sample plane
        P_M_EW = inv_props{2}.propTF(EW);
        P_M_P= inv_props{1}.propTF(P);
        
        O = (P_M_EW ./ P_M_P);
        result_O{intermediate_result_counter} = gather(O);
        result_P{intermediate_result_counter} = gather(P);
        EW = gather(EW);
    else
        result_O{intermediate_result_counter} = gather(O);
        result_P{intermediate_result_counter} = gather(P_M_P);
        EW = gather(P_M_EW);
    end
    
else
    % default behavior
    EW = props{2}.propTF(EW);
    P = props{1}.propTF(P);
    EW = inputEW .* exp(1i .* angle(EW));
    P = inputP .* exp(1i .* angle(P));
    
    % back at sample plane
    EW = inv_props{2}.propTF(EW);
    P= inv_props{1}.propTF(P);
    
    O = (P_M_EW ./ P_M_P);
    result_O{intermediate_result_counter} = gather(O);
    result_P{intermediate_result_counter} = gather(P);
    EW = gather(EW);
end

close(h);
end