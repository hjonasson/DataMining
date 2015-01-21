%
% to find partial derivatives of fk at xk %
function fka = fkfind(nelem,nend,xk,fk) nde = nelem * nend;
ntotal = nelem + nend + nde;
datam = xk(1,1:nelem);
para = xk(1,nelem+1:nelem+nend); datae = xk(1,nelem+nend+1:ntotal); max = nelem+nend;
index = 0;
for i = 1:nelem for j = 1:ntotal
if fk(i,j) ~= 0 if i == j
fk(i,j) = 1.;
elseif j <= max & i~=j
ll = j-nelem;
kk = ll+index;
fk(i,j) = -datae(kk);
else
kk = nelem+nend+index; ll = j-kk;
fk(i,j) = -para(ll);
end end
end
index = index+nend; end
fka = fk;