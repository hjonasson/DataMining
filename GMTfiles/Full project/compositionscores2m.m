%
% Calculates the composition scores for qmodemain2m %
function [CFS, scalefactors] = compositionscores2m(FS,Kbar,Xmax,Xmin)
[nv nf] = size(FS);
Xmax = Xmax(:); % forces Xmax to be column vector FS = FS .* (Xmax * ones(1,nf));
scalefactors = sum(FS,1);
scalefactors = Kbar ./ scalefactors;
CFS = FS .* (ones(nv,1) * scalefactors);
CFS = CFS + (Xmin' * ones(1,nf));