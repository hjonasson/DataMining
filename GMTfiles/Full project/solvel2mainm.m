%
% Program fits linear partitioning model using the % constrained least squares routine.
%
clear;
lp_file = input('Enter file name for saving output: ','s');
disp(' ');
lpfid = fopen(lp_file,'w');
runlabel = input('Enter title: ','s');
runlabel = ['SOLVE Least Squares ' runlabel ' ' datestr(now,0)]; fprintf(lpfid,[runlabel '\n']);
% loads nazca plate surface sediment chemical composition data fid = -1;
while fid == -1
disp(' ');
infile = input('Enter file name of input data: ','s'); [fid,message]=fopen(infile,'r');
end
fclose(fid);
X1 = load (infile);
nv = size(X1,2);
X = X1(:,[2:nv]);
%X = X/1000.; %inserted for the nazca plate data [N nv] = size(X);
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
infile = input('Enter file name with end-member compositions: ','s');
[fid,message]=fopen(infile,'r'); end
fclose(fid);
d = load (infile);
% d matrix is a linear matrix with one number per line.
% composition of end member 1 first, then 2 etc.
% We know the number of variables. size of d gives use the % number of end-members.
nf = size(d(:),1)/nv;
d_matrix = reshape(d,nf,nv)';
coef = d_matrix; % coef now has end-member compositions
fprintf(lpfid,'Cost set at 1/sqrt(observed(i)) \n'); matrixprint(lpfid,'Composition Matrix',coef);
% add constraint that sum of coeff. add to 1.0 coef(nv+1,:) = 1.0/nf;
% zero vectors for goodness of fit calculation sumr = zeros(nv, 1);
sumsr = zeros(nv, 1);
% set options for lsqnonneg
options = optimset('Diagnostics','off'); % Start main loop
for i = 1: N
rhs = [X(i,:) 1]; scale = sqrt(rhs); for j = 1: nv
% rhs right hand side of equations % scale by right hand side
if scale(j) ~= 0 scale(j) = 1/scale(j);
end end
scale(nv+1) = 1.0; % scale for the sum of coefficients A = (scale'*ones(1,nf)) .* coef;
% apply scales and then make column vector
rhs = rhs .* scale;
rhs = rhs(:);
xbasic = lsqnonneg(A,rhs,[],options); % we did use nnls B(i,:) = xbasic';
% save end-member weights for later and make rows rhs = rhs ./scale';
% in B. Remove scaling
EndMembers = ones(nv,1) * B(i,:) .* coef(1:nv,:);
% calculate contribution of each end-member in sample
Estimate = B(i,:) * coef(1:nv,:)';
Estimate = Estimate'; % estimate sample composition residual = rhs(1:nv,:) - Estimate; % residual
matrixprint(lpfid,['End Member Weights for Sample ' num2str(i)], B(i,:)); l1matrixprint(lpfid,['Sample ' num2str(i)],...
[EndMembers rhs(1:nv,:) Estimate residual],variable); % sum residuals
sumr = sumr + residual;
sumsr = sumsr + (residual .* residual);
end
variance = var(X);
sumr = sumr/N;
sumsr = sumsr/N - (sumr .* sumr);
goodness = (variance' - sumsr) ./(variance');
goodness = 100.*goodness;
% goodness of fit is (var(data) - var(resid))/var(data) sumsr = sqrt(sumsr);
matrixprint(lpfid,'Residual Means ',sumr'); matrixprint(lpfid,'Residual STD. DEVS. ',sumsr'); matrixprint(lpfid,'Coef. of Determination ',goodness'); aoutfid = fopen('A_leastsq.doc','w');
fprintf(aoutfid,'%8.3f %8.3f %8.3f %8.3f %8.3f\n',B'); fclose(aoutfid);
fclose(lpfid);