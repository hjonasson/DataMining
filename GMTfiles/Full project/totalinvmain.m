clear;
lp_file = input('Enter file name for saving output: ','s');
disp(' ');
lpfid = fopen(lp_file,'w');
runlabel = input('Enter title: ','s');
runlabel = ['TOTAL INVERSION ' runlabel ' ' datestr(now,0)]; fprintf(lpfid,[runlabel '\n']);
% loads nazca plate surface sediment chemical composition data fid = -1;
while fid == -1
	disp(' ');
infile = input('Enter file name of input data: ','s'); [fid,message]=fopen(infile,'r');
end
fclose(fid);
X1 = load (infile);
nv = size(X1,2);
X = X1(:,[2:nv]);
X = X/1000.; %inserted for the nazca plate data variance = var(X);
[N nv] = size(X);
numvar = [1: 1: nv]';
sampnums = X1(:,1);
fid = -1;
while fid == -1
disp(' ');
labelfile = input('Enter file name with variable labels: ','s'); [fid,message]=fopen(labelfile,'r');
end
for i = 1: nv
variable(i).names = fscanf(fid,'%s',1); end
fid = -1;
while fid == -1
disp(' ');
infile = input('Enter file name with end-member compositions: ','s'); [fid,message]=fopen(infile,'r');
end
fclose(fid);
d = load (infile); % d matrix is a linear matrix with one number per line.
% composition of end member 1 first, then 2 etc.
% we know the number of variables. The size of d use the number of % end-members
nf = size(d(:),1)/nv; d_matrix = reshape(d,nf,nv)'; coef = d_matrix;
matrixprint(lpfid,'Composition Matrix',coef);
% total number of elements is ntotal = nv+nf+(nf*nv)
nde = nv * nf;
ntotal = nv + nf + nde;
% covariance initially set to be diagonal, with zeros off diagonal and
% diagonal elements errors first of the nv precisions, for the end members % and then errors for the composition matrix
fid = -1;
while fid == -1
disp(' ');
infile = input('Enter file name with variances: ','s'); [fid,message]=fopen(infile,'r');
end
fclose(fid);
comatrix = load (infile); % d matrix is a linear matrix with one number per line.
% composition of end member 1 first, then 2 etc.
matrixprint(lpfid,'Variance Vector',comatrix);
temp = comatrix;
co = zeros(ntotal,ntotal); % zero covariance matrix of variables for i = 1: ntotal
co(i,i) = temp(i); end
% define the initial guess vector x0. First part is data from a sample (nv) % then 1/nf for nf values and then coef array.
x0 = zeros(ntotal,1); x0(nv+1:nv+nf,1) = 1/nf;
temp = coef'; x0(nv+nf+1:ntotal,1) = temp(:); x0 = x0';
% define the fk matrix diagonal matrix for nv x nv, then all ones
fk = [eye(nv,nv) ones(nv,ntotal-nv)];
for i = 1: nv
for j = nv+nf+1: ntotal
itop = nv+nf+1 + (nf * i); ibottom = nv+nf+1 + nf * (i-1); if j >= itop || j < ibottom
fk(i,j) = 0; end
end end
% save initial matricies savex0 = x0;
savexk = x0;
saveco = co;
savefk = fk;
%
% zero vectors for goodness of fit calculation
% set lstep and eps number of iterations and accuracy lstep = 1000;
eps = .001;
% zero vectors before starting loop
sumr = zeros(nv, 1); sumsr = zeros(nv, 1); compmean = zeros(nv,nf); compvar = zeros(nv,nf); B = zeros(N,nf);
% Start main loop
for i = 1: N
x0 = savex0; xk = savexk; co = saveco; fk = savefk;
raw = X(i,:); x0(1,1:nv) = raw;
fk = fkfind(nv,nf,xk,fk);
% disp('done with initial entrance to fkfind') [fxk afxk] = fxkevl(nv, nf, xk);
% disp('done with initial call to fxkevl')
% iteration loop here for it = 1: lstep
x = x0' + co * fk' * inv(fk * co * fk') * (fk * ( xk' - x0') - fxk); xk = x';
for izerocheck = nv+1: nv+nf if xk(izerocheck) < 0
xk(izerocheck) = 0; x0(izerocheck) = 0; co(izerocheck,izerocheck) = 0;
end end
[fxk afxk] = fxkevl(nv, nf, xk); if sum(abs(afxk)) <= eps
disp ('Sample Done') Estimate1 = xk(1:nv)';
coef = xk(nv+nf+1:nv * nf + nv + nf);
coef = reshape(coef,nf,nv)';
compmean = compmean + coef;
compvar = compvar + coef .* coef;
B(i,:) = xk(nv+1:nv+nf);
EndMembers = ones(nv,1) * B(i,:) .* coef(1:nv,:); Estimate = B(i,:) * coef(1:nv,:)';
Estimate = Estimate'; % calculated for debug purposes
residual = raw(:) - Estimate1;
matrixprint(lpfid,['End Member Weights for Sample ' num2str(i)], B(i,:)); l1matrixprint(lpfid,['Sample ' num2str(i)], [EndMembers raw(:) Estimate1
residual],variable);
% sum residuals
sumr = sumr + residual;
sumsr = sumsr + (residual .* residual);
if i < N break
else
compmean = compmean/N;
compvar = compvar/N - compmean .*compmean matrixprint(lpfid,'Mean Composition ', compmean); matrixprint(lpfid,'Composition Variance ',compvar);
sumr = sumr/N;
sumsr = sumsr/N - (sumr .* sumr);
goodness = (variance' - sumsr) ./(variance');
goodness = 100.*goodness;
sumsr = sqrt(sumsr);
matrixprint(lpfid,'Residual Means ',sumr'); matrixprint(lpfid,'Residual STD. DEVS. ',sumsr'); matrixprint(lpfid,'Coef. of Determination ',goodness'); break
end
end
fk = fkfind(nv,nf,xk,fk); end
end
aoutfid = fopen('A_totalinv.doc','w');
fprintf(aoutfid,'%8.3f %8.3f %8.3f %8.3f %8.3f\n',B'); fclose(aoutfid);
fclose(lpfid);