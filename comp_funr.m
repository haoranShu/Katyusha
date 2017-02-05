function x = comp_funr(w, Xt, y, lambda)

x = sum((Xt*w - y).^2)/(length(y))+(lambda/2)*sum(w.^2);

end