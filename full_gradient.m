function fullg = full_gradient(w, Xt, Xtt, y, lambda,nn)

fullg = 2*Xtt*(Xt*w - y)./nn +lambda*w;

end