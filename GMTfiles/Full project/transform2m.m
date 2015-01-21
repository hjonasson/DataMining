%
% Routine used by qmodemain2m to scale data to constant means,
% percent range or range transformation
% Slightly modified from original code by N.G. Pisias to include the range % transformation (Nahysa Martinez August 2005).
%
function [x, xmin, xmax, ks, kbar, constant] = transform2m(X,transformtype,lpfid);
% calculate ks and kbar first ks = sum(X,2);
kbar = mean(ks);
xmin = min(X,[],1);
xmax = max(X,[],1);
xrange = xmax - xmin;
[n m] = size(X);
% n samples and m variables if transformtype == 0
fprintf(lpfid,'No pretreatement requested \n'); constant = kbar;
xmax = ones(1,m);
xmin = zeros(1,m);
x = X;
elseif transformtype == 1
scale = ones(n,1) * (ones(1,m).*100 ./(mean(X))); x = X .* scale;
constant = 100 * m;
xmax = mean(X) / 100.;
xmin = zeros(1,m);
fprintf(lpfid,'Constant Percent Transform \n'); elseif transformtype == 2
scale = ones(n,1) * (ones(1,m)./xmax); x = X .* scale;
constant = kbar;
xmin = zeros(1,m);
fprintf(lpfid,'Percent Maximum Transform \n'); elseif transformtype == 3
scale = X -(ones(n,1)* xmin);
x = scale./(ones(n,1)* xrange); fprintf(lpfid,'Range Transformed \n'); xmax = xrange;
kbar = kbar - sum(xmin);
xmin = min(X,[],1);
constant = kbar;
elseif transformtype == 4
x = log(X + ones(n,m));
fprintf(lpfid,'Log (X+1) Transformed \n'); ks = sum(x,2);
kbar = mean(ks); constant = kbar; xmax = ones(1,m); xmin = zeros(1,m);
end