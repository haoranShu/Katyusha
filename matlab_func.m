mex Katyusha.cpp -largeArrayDims

clear;
load('sparsedata.mat');

X = [ones(size(X,1),1) X];
[nn, dd] = size(X);
X = X';
L = 1000;
lambda = 1/nn;
tau = min(0.5, sqrt(2*nn*lambda/3/L));
stepSize = 0.0001;
iters = 1;
w = zeros(dd, 1);
tic;
histKatyusha = Katyusha(w, X, y, lambda, stepSize, tau, L, iters)
t = toc