#include "mex.h"
#include <stdio.h>
#include <math.h>

//The function that updates w
void update_w(double *w, double *z, double *wold, double *yy, long long dd, double tau, double tau1){
    for(long k = 0; k < dd; k++){
        w[k] = tau*z[k] + 0.5*wold[k] + tau1*yy[k];
    }
}

//The function that updates z
void update_z(double *z, double *wg, long long dd, double stepSize){
    for (long k = 0; k < dd; k++){
        z[k] = z[k] - stepSize*wg[k];
    }
}

//The function that updates y
void update_y(double *yy, long long dd, double *w, double *wg, double L){
    for(long k = 0; k < dd; k++){
        yy[k] = w[k] - wg[k]/L;
    }
}

//The function that calculates wg (dense)
void calculate_wg_dense(double *wg, double *w, double *wold, double *pp, long j, double *fullg, double lambda, long long dd, double *Xt){
    double aa = 0;
    for (long k = 0; k < dd; k++){
        aa += 2*Xt[((long long)pp[j-1])*dd+k]*(w[k]-wold[k]);
    }
    for (long k = 0; k < dd; k++){
        wg[k] = aa*Xt[((long long)pp[j-1])*dd+k] + lambda*(w[k]-wold[k]) + fullg[k];
    }
}

//The function that calculates wg (sparse)
void calculate_wg_sparse(double *wg, double *w, double *wold, double *pp, long j, double *fullg, double lambda, long long dd, double *Xt, mwIndex *ir, mwIndex *jc){
    double aa = 0;
    long long l = (long long)pp[j-1];
    for (long k = jc[l]; k < jc[l+1]; k++){
        aa += 2*Xt[k]*(w[ir[k]]-wold[ir[k]]);
    }
    for (long k = 0; k < dd; k++){
        wg[k] = lambda*(w[k]-wold[k]) + fullg[k];
    }
    for (long k = jc[l]; k < jc[l]; k++){
        wg[ir[k]] += aa*Xt[k];
    }
}

//Compute objective function value (dense)
double compute_function_value_dense(long long nn, long long dd, double *w, double *Xt, double *y, double lambda){
    double value = 0;
    double tmp;
    for (long i = 0; i < nn; i++){
        tmp = 0;
        for (long j = 0; j < dd; j++){
            tmp += Xt[i*dd+j]*w[j];
        }
        value += (tmp-y[i])*(tmp-y[i]);
    }
    value = value/(double)nn;
    //Regularization
    for (long j = 0; j < dd; j++){
        value += (lambda/2)*w[j]*w[j];
    }
    return value;
}

//Compute objective function value (sparse)
double compute_function_value_sparse(long long nn, long long dd, double *w, double *Xt, double *y, double lambda, mwIndex *ir, mwIndex *jc){
    double value = 0;
    double tmp;
    for (long i = 0; i < nn; i++){
        tmp = 0;
        for (long j = jc[i]; j < jc[i+1]; j++){
            tmp += Xt[j]*w[ir[j]];
        }
        value += (tmp-y[i])*(tmp-y[i]);
    }
    value = value/(double)nn;
    //Regularization
    for (long j = 0; j < dd; j++){
        value += (lambda/2)*w[j]*w[j];
    }
    return value;
}

//Compute the full gradient (dense)
void compute_full_gradient_dense(double *fullg, long long nn, long long dd, double *Xt, double *y, double lambda, double *w){
    for (long k = 0; k < dd; k++){
        fullg[k] = 0;
    }
    double *tmp;
    mxArray *tmpM;
    tmpM = mxCreateDoubleMatrix(nn, 1, mxREAL);
    tmp = mxGetPr(tmpM);
    for (long i = 0; i < nn; i++){
        tmp[i] = 0;
    }
    for (long i = 0 ; i < nn; i++){
        for (long k = 0; k < dd; k++){
            tmp[i] += w[k]*Xt[i*dd+k];
        }
        tmp[i] -= y[i];
    }
    for (long k = 0; k < dd; k++){
        for (long i = 0; i < nn; i++){
            fullg[k] += 2*tmp[i]*Xt[i*dd+k];
        }
        fullg[k] = fullg[k]/(double)nn;
        fullg[k] += lambda*w[k];
    }
}

//Compute the full gradient (sparse)
void compute_full_gradient_sparse(double *fullg, long long nn, long long dd, double *Xt, double *y, double lambda, double *w, mwIndex *ir, mwIndex *jc){
    for (long k = 0; k < dd; k++){
        fullg[k] = 0;
    }

    double *tmp = new double[nn];
    for (long i = 0; i < nn; i++){
        tmp[i] = 0;
    }
    for (long i = 0 ; i < nn; i++){
        for (long k = jc[i]; k < jc[i+1]; k++){
            tmp[i] += w[ir[k]]*Xt[k];
        }
        tmp[i] -= y[i];
    }
    for (long i = 0; i < nn; i++){
        for (long k = jc[i]; k < jc[i+1]; k++){
            fullg[ir[k]] += 2*tmp[i]*Xt[k];
        }
    }
    for (long k = 0; k < dd; k++){
        fullg[k] = fullg[k]/(double)nn;
        fullg[k] += lambda*w[k];
    }
}
