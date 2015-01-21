%
% Calculates goodness of fit statistics for qmode factor % analysis using Meisch (1976) extensions.
%
function [x] = nodosgoodns(X,B,CFS,scalefactors,Xmax,Xmin,Ks, Kbar,vectorlengths,lpfid,variable)
[n,nf] = size(B); [n,nv] = size(X); variance = var(X); Ascale = ones(nf-1,nf); for i = 1: nf-1
for j = i+2: nf Ascale(i,j) = 0.0;
end end
sumr = zeros(nv, nf-1); sumsr = zeros(nv, nf-1); for i =1: n
asum = cumsum(B(i,:) ./ scalefactors); asum = asum(2:nf);
A = ones(nf-1,1)*B(i,:);
A = A ./ (ones(nf-1,1) * scalefactors); A = A * Ks(i)/Kbar;
A = A ./ (asum' * ones(1,nf)); A = A .* Ascale;
xest = A * CFS';
resid = xest - ones(nf-1,1) * X(i,:);
sumr = sumr + resid';
sumsr = sumsr + (resid .* resid)'; end
sumr = sumr/n;
sumsr = sumsr/n - (sumr .* sumr);
goodness = ((variance' * ones(1,nf-1)) - sumsr) ./(variance' * ones(1,nf-1));
sumsr = sqrt(sumsr);
gmatrixprint(lpfid,['Residual Means for 2 to ' num2str(nf) ' Factors'],sumr,variable); gmatrixprint(lpfid,['Residual STD. DEVS. for 2 to ' num2str(nf) ' Factors'],sumsr,variable);
gmatrixprint(lpfid,['Coef. of Determination for 2 to ' num2str(nf) ' Factors'],goodness,variable);
x = goodness;
%
% Calculates the composition scores for qmodemain2m %
function [CFS, scalefactors] = compositionscores2m(FS,Kbar,Xmax,Xmin)
[nv nf] = size(FS);
Xmax = Xmax(:); % forces Xmax to be column vector FS = FS .* (Xmax * ones(1,nf));
scalefactors = sum(FS,1);
scalefactors = Kbar ./ scalefactors;
CFS = FS .* (ones(nv,1) * scalefactors);
CFS = CFS + (Xmin' * ones(1,nf));