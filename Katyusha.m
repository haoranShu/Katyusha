function [hist, w, z, yy] = Katyusha(w, Xt, y, lambda, stepSize, tau, L, iters);
[nn, dd] = size(Xt);
%Xt   = sparse(Xt);
Xtt  = Xt';
hist = zeros(iters+1,1);
tau1 = 0.5 - tau;
tau2 = 1 + lambda*stepSize;

for ii = 1:iters
hist(ii) = comp_funr(w, Xt, y, lambda);
fullg    = full_gradient(w, Xt, Xtt, y, lambda, nn);
pp   = [randperm(nn),randperm(nn)]';
wold     = w;
z   = w;
yy  = w;
www = zeros(dd, 1);
bb  = 0;
for j = 1:2*nn
w  = z*tau + 0.5*wold + yy*tau1;
aa = 2*Xt(pp(j),:)*(w - wold);
wg = Xtt(:,pp(j))*aa + (w - wold)*lambda + fullg;
zold = z;
z  = z - wg*stepSize;
%y  = (L*w - wg)/(L + lambda);
yy  = w - wg/L;
www = www + tau2^(j-1)*yy;
bb  = bb + tau2^(j-1);
end
w = www/bb;
end
%hist(ii+1) = feval(@comp_funr, w, Xt, y, lambda);