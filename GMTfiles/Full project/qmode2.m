%
% Performs the Qmode Factor Analysis calculations where:
% X = input matrix
% B = Factor loading matrix
% F = Factor scores matrix
% scF = Factor scores matrix normalized to sqrt(p) %
function [U,B,F,scF,Lambda,Pervar,Cumvar,count] = qmode2(X,alpha)
disp ('Qmode using the psuedo-cosine theta matrix') [n,p]= size(X);
U = X./(sqrt(diag(X*X')*ones(1,p)));
% COSINE THETA MATRIX; EIGENVECTORS %Here we use the psuedo-cosine theta matrix
Ss = U'*U;
TV=trace(Ss);
[Q,Lambda]=eig(Ss);
sortvector = zeros(p);
eigenvalues = diag(Lambda); [eigenvalues,sortvector] = sort(eigenvalues); for i = 1: p
Lambda(i,i) = eigenvalues(p+1-i);
Eigenvectors (:,i) = Q(:,sortvector(p+1-i)); end
Q = Eigenvectors; % eigenvectors in V are columns and sorted by eigenvalues
%Lambda=rot90(Lambda,2);
%Q=fliplr(Q);
%Lambda
% PERCENT AND CUMULATIVE VARIANCE Pervar = (100*Lambda/TV);
cutoff=0; count=0; for i=1:p
if cutoff <= alpha; cutoff=cutoff + Pervar(i,i); count=count + 1;
end
end Pervar=diag(Pervar);
Cumvar = zeros(p,1); for i=1:p
Cumvar(i)= sum(Pervar(1:i,1)); end
% FACTOR LOADINGS (B); SCORES(F) B=U*Q;
F=U'*B*inv(Lambda);
scF = (sqrt(p))*F;
% GOODNESS OF FIT STATISTICS; COMMUNALITIES Comm=diag(B*B');
%Resmeans = ones(p,count);
% for j=count:-1:1
% Uest=B(:,1:j)*F(:,1:j)';
% Resmeans(:,j)=(mean(Uest)-mean(U))'; % end