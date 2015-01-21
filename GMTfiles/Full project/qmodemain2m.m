%
% qmodemain2m reads in multivariate data set and calculates
% a qmode factor analysis. Assumes first column of data
% set is sample number
% Last Revisions May 2005. Following items changed:
% 1. better labeling of matricies in functions
% fmatrixprint and bmatrixprint.
% 2. added variable names to label plots and output
% files.
% 3. checked B matrix to see if it''s all (or nearly all)
% zeros and multiples the B vector by -1 and the T varimax
% transformation matrix column by -1 as well.
% 4. This code has been slightly modified from the
% original qmodemain2 of Pisias. The modifications include:
% (a) range transformation is included
% (b) Pisias extra rotation to obtain positive Factor
% composition scores is also included. Modifications made
% by Nahysa Martinez (Boston University) August 2005. %

clear; %lpfid = 1;
lp_file = input('Enter file name for saving output: ','s'); disp(' ');
lpfid = fopen(lp_file,'w');
runlabel = input('Enter title: ','s');
runlabel = ['QMODE ' runlabel ' ' datestr(now,0)]; fprintf(lpfid,[runlabel '\n']);

% loads nazca plate surface sediment chemical composition data fid = -1;
while fid == -1
disp(' ');
infile = input('Enter file name of input data: ','s'); [fid,message]=fopen(infile,'r');
end
fclose(fid);

X1 = load (infile); nv = size(X1,2);
X = X1(:,[2:nv]);
[N nv] = size(X); numvar = [1: 1: nv]'; sampnums = X1(:,1);%'

fid = -1;
while fid == -1
	disp(' ');
	labelfile = input('Enter file name with variable labels: ','s'); [fid,message]=fopen(labelfile,'r');
end

for i = 1: nv
	variable(i).names = fscanf(fid,'%s',1);
end

%variable(10).names ='Ba';
% calculate means and variances and output
%X = X ./ 1000.;
xmean = mean(X)';
xvar = var(X)';
format = '%7i %20.4f %20.4f \n';
fprintf(lpfid,'Var. Num. Variable Name Means Variance \n'); for j = 1: nv
Sout = sprintf('%7i',numvar(j));
Sout = [Sout ' ',sprintf(' %s',variable(j).names)];
Sout = [Sout ' ',sprintf('%20.4f %20.4f \n',xmean(j),xvar(j))]; fprintf(lpfid,'%s',Sout);
end
% get input parameters If needed, scale data
%alpha = 100;
disp(' ');
alpha = input('alpha level, e.g. 95: ');
disp(' ');
disp('Enter 0 for No Pretreatment, 1 for Constant Mean,'); mode = input( '2 for percent max,3 for range, and 4 for log: '); if mode > 4 | mode < 0
mode = 0 end
vectorlengths = sqrt(diag(X * X'));%'
% transform data
[x, Xmin, Xmax, Ks, Kbar, constant] = transform2m(X,mode,lpfid);
% data now in and scaled to constant mean of 100 if needed. % go to qmode subroutine
[U,B,F,scF,Lambda,Pervar,Cumvar,count] = qmode2(x,alpha); Lambda = diag(Lambda);
% factor analysis completed. Results are in B and F. 
% output initial results
fprintf(lpfid,'\n');
nout = size(F,2);
nvcount =[1: 1:count];
fprintf(lpfid,' Eigen Values Per. Infor. Accumulative Information \n'); fprintf(lpfid,'%10i %10.4f %10.4f %10.4f \n',[nvcount' Lambda(1:count) Pervar(1:count) Cumvar(1:count)]');
fprintf(lpfid,'\n');
fmatrixprint(lpfid,'Principal Factor Scores Matrix',F,variable);
%output completed
% get number of factors wanted in analysis
message = ['Count = ', num2str(count)]; disp(message);
Pervar(1:count)
count = input('Enter NO. of Factors to keep: '); B = B(:,1:count);
comm = diag(B*B');
Info = diag(B'*B);
PerInfo = Info * 100/N; CumInfo = cumsum(PerInfo);
% now ouput factor loadings
bmatrixprint(lpfid,'Principal Factor Loadings Matrix', [sampnums comm B]); format = '';
for j = 1: count
format = [format,'%8.4f '];
end
format = [format,'\n']; fprintf(lpfid,['Percent Information fprintf(lpfid,['Cumulative Info. fprintf(lpfid,'\n');
' format],[PerInfo(1:count)']); ' format],[CumInfo(1:count)']);
% now output results.
% Now we have printed the B - Loadings matrix now % do factor rotations using varimax
[CFS, scalefactors] = compositionscores2m(F, Kbar, Xmax, Xmin); scalefactors = scalefactors(1:count);
CFS = CFS(:, 1:count);
fmatrixprint(lpfid,'Factor Composition Scores ',CFS,variable); goodness = nodosgoodns(X,B,CFS,scalefactors,Xmax,Xmin,Ks, Kbar,vectorlengths,lpfid,variable);
[B, T] = varimax(count,B); comm = diag(B*B');
Info = diag(B'*B);
PerInfo = Info * 100/N; CumInfo = cumsum(PerInfo);
% need to see if we need to multiply a factor by -1 because B's are all <0 % We can then adjust the T matrix (we might want this later) so that the L % matrix is correct sign.
for j = 1: count
positives = find(B(:,j)>0); if size(positives,1) < N/2
B(:,j) = -B(:,j);
T(:,j) = -T(:,j); end
end
F = F(:,1:count); L = F*T;
bmatrixprint(lpfid,'Varimax Factor Loadings Matrix',[sampnums comm B ]); fprintf(lpfid,['Percent Information ' format],[PerInfo(1:count)']); fprintf(lpfid,['Cumulative Info. ' format],[CumInfo(1:count)']); fprintf(lpfid,'\n');
% calculate Varimax Factor Scores using T matrix and output fmatrixprint(lpfid,'VARIMAX Factor Scores Matrix ',L,variable); % Scale the factor scores.
scale = ones(nv,1) * (ones(count,1) .* sqrt(nv))';
LS = L .* scale;
fmatrixprint(lpfid,'VARIMAX Scaled Factor Scores Matrix ',LS,variable); [CFS, scalefactors] = compositionscores2m(L,Kbar,Xmax,Xmin); fmatrixprint(lpfid,'VARIMAX Factor Composition Scores ',CFS,variable);
% Do the extra rotation to get scores in a positive space using Leinen & % Pisias, (1984).
[ERCS]=positiverot(X,CFS);
fmatrixprint(lpfid,'Final Factor Composition Scores ',ERCS,variable); fclose(lpfid);
% plot data xtic = [];
for i = 1: nv
xtic = [xtic;variable(i).names];
end
figure (1)
xcount = [1: 1: nv]'; for i = 1:count
subplot(2,4,i) bar(xcount,(L(:,i)),'r')
h = gca;
%get(h) set(h,'XTick',numvar); set(h,'XLim',[0 numvar(nv)]); set(h,'XTickLabel',xtic);
if i == 1 | i == 5
ylabel('Scores') else
ylabel('') end
h =title(['Factor ',num2str(i)]);
set (h,'FontSize',12); end