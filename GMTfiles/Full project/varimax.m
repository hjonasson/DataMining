%
% Performs the varimax factor rotation and returns a
% transformation matrix (T).
% This procedure follows algorithms from Harman (1960).
% Modified by N.G. Pisias to fix the convergence criteria.
% We let the program do 10 iterations.
% Convergence is usually very fast. %
function [B, T] = varimax(nf,Lding)
B = Lding;
[n,nf]=size(Lding);
T = eye(nf);
hjsq = diag(Lding*Lding');
hj = sqrt(hjsq); % communalities calculated
%v0 = vfunct(Lding,hj);
for it = 1:10 % Never seems to need very many iterations, but add more if needed %
for icount = 1: nf-1 % Cycles through 2 factors at a time
jl = icount + 1; for j = jl: nf
xj = Lding(:,icount)./hj; yj = Lding(:,j)./hj;
uj = xj.*xj - yj.*yj;
vj = 2*xj.*yj;
A = sum(uj);
BB = sum(vj)';
C = uj'*uj - vj'*vj;
D = 2*uj'*vj;
num = D -2 * A * BB/n;
den = C - (A^2 - BB^2)/n;
tan4p = num/den;
phi = atan2(num,den)/4; % this finds the right quadrant angle = phi*180/pi;
if abs(phi) > .0000
Xj = cos(phi)*xj + sin(phi)*yj; Yj = -sin(phi)*xj + cos(phi)*yj; bj1 = Xj.*hj;
bj2 = Yj.*hj;
B(:,icount) = bj1;
B(:,j) = bj2;
Lding(:,icount) = B(:,icount);
Lding(:,j) = B(:,j); for k = 1:nf
tp = T(k,icount);
T(k,icount) = tp * cos(phi) + T(k,j) * sin(phi); T(k,j) = -tp * sin(phi) + T(k,j) * cos(phi);
end end
end end
Lding = B;
hjsq = diag(Lding*Lding');
hj = sqrt(hjsq); % communalities calculated %v = vfunct(Lding,hj);
% % % % % end;