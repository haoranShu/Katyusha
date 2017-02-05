//
//  MM.cpp
//  
//
//  Created by 舒 浩然 on 16/9/28.
//
//

#include <stdio.h>
#include "mex.h"
#include <math.h>
#include <string.h>
#include <algorithm>
#include <random>

#include "utilities.h"

//Katyusha

//For dense Xt
mxArray* Katyusha_dense(int nlhs, const mxArray *prhs[]){
    //Declare variables
    double *w, *Xt, *y;
    double lambda, stepSize, tau, L;
    long long iters;
    double *pp;
    double *hist;
    
    mxArray *plhs;
    
    double tau1, tau2;
    double bb;
    bool evalf = false;
    long long nn, dd;
    
    //Process input
    w = mxGetPr(prhs[0]);
    Xt = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    stepSize = mxGetScalar(prhs[4]);
    tau = mxGetScalar(prhs[5]);
    L = mxGetScalar(prhs[6]);
    iters = (long long)mxGetScalar(prhs[7]);
    
    if (nlhs == 1){
        evalf = true;
    }
    
    //Get problem related constants
    dd = mxGetM(prhs[1]);
    nn = mxGetN(prhs[1]);
    tau1 = 0.5 - tau;
    tau2 = 1 + lambda*stepSize;
    double *www;
    mxArray *wwwM;
    wwwM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    www = mxGetPr(wwwM);
    
    for (long k = 0; k < dd; k++){
        www[k] = 0;
    }
    
    //declare assisting variables
    double *wold, *z, *yy, *zold, *fullg, *wg;
    mxArray *woldM, *zM, *yyM, *zoldM, *fullgM, *wgM;
    woldM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    wold = mxGetPr(woldM);
    zM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    z = mxGetPr(zM);
    yyM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    yy = mxGetPr(yyM);
    zoldM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    zold = mxGetPr(zoldM);
    fullgM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    fullg = mxGetPr(fullgM);
    wgM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    wg = mxGetPr(wgM);
    
    mxArray *ppM;
    ppM = mxCreateDoubleMatrix(2*nn, 1, mxREAL);
    pp = mxGetPr(ppM);
    for (long long i = 0; i < nn; i++){
        pp[i] = i;
        pp[i+nn] = i;
    }
    
    if (evalf == true) {
        plhs = mxCreateDoubleMatrix((long)floor((double)iters / (nn)) + 1, 1, mxREAL);
        hist = mxGetPr(plhs);
    }
    
    //Interations
    for (long i = 1; i <= iters; i++){
        if (i%nn == 0){
            hist[(long)floor((double)i / (nn))] = compute_function_value_dense(nn, dd, w, Xt, y, lambda);
        }
        compute_full_gradient_dense(fullg, nn, dd, Xt, y, lambda, w);
        for (long k = 0; k < dd; k++){
            wold[k] = w[k];
            z[k] = w[k];
            yy[k] = w[k];
        }
        
        std::random_shuffle (&pp[0], &pp[nn]);
        std::random_shuffle (&pp[nn], &pp[2*nn]);
        
        for (long k = 0; k < dd; k++){
            www[k] = 0;
        }
        bb = 0;
        for (long j = 1; j <= 2*nn; j++){
            update_w(w, z, wold, yy, dd, tau, tau1);
            calculate_wg_dense(wg, w, wold, pp, j, fullg, lambda, dd, Xt);
            for(long k = 0; k < dd; k++){
                zold[k] = z[k];
            }
            update_z(z, wg, dd, stepSize);
            update_y(yy, dd, w, wg, L);
            for(long k = 0; k < dd; k++){
                www[k] = www[k] + pow(tau2, j-1)*yy[k];
            }
            bb += pow(tau2, j-1);
        }
        for (long k = 0; k < dd; k++){
            w[k] = www[k]/bb;
        }
    }
    hist[(long)floor((double)iters / (nn))] = compute_function_value_dense(nn, dd, w, Xt, y, lambda);
    if (evalf == true) { return plhs; }
    else { return 0; }
}


//For sparse Xt
mxArray* Katyusha_sparse(int nlhs, const mxArray *prhs[]){
    //Declare variables
    double *w, *Xt, *y;
    double lambda, stepSize, tau, L;
    long long iters;
    //double *pp;
    double *hist;
    
    mwIndex *ir, *jc;
    mxArray *plhs;
    
    double tau1, tau2;
    double bb;
    bool evalf = false;
    long long nn, dd;
    
    //Process input
    w = mxGetPr(prhs[0]);
    Xt = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    stepSize = mxGetScalar(prhs[4]);
    tau = mxGetScalar(prhs[5]);
    L = mxGetScalar(prhs[6]);
    iters = (long long)mxGetScalar(prhs[7]);
    
    if (nlhs == 1){
        evalf = true;
    }
    
    //Get problem related constants
    dd = mxGetM(prhs[1]);
    nn = mxGetN(prhs[1]);
    jc = mxGetJc(prhs[1]);
    ir = mxGetIr(prhs[1]);
    tau1 = 0.5 - tau;
    tau2 = 1 + lambda*stepSize;
    double *www;
    mxArray *wwwM;
    wwwM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    www = mxGetPr(wwwM);
    
    for (long k = 0; k < dd; k++){
        www[k] = 0;
    }
    
    //declare assisting variables
    //double *wold, *z, *yy, *zold, *fullg, *wg;
    //mxArray *woldM, *zM, *yyM, *zoldM, *fullgM, *wgM;
    double *wold = new double[dd];
    double *z = new double[dd];
    double *yy = new double[dd];
    double *zold = new double[dd];
    double *fullg = new double[dd];
    double *wg = new double[dd];
    /*
    woldM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    wold = mxGetPr(woldM);
    zM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    z = mxGetPr(zM);
    yyM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    yy = mxGetPr(yyM);
    zoldM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    zold = mxGetPr(zoldM);
    fullgM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    fullg = mxGetPr(fullgM);
    wgM = mxCreateDoubleMatrix(dd, 1, mxREAL);
    wg = mxGetPr(wgM);
    
    mxArray *ppM;
    ppM = mxCreateDoubleMatrix(2*nn, 1, mxREAL);
    pp = mxGetPr(ppM);
     */
    double *pp = new double[2*nn];
    
    for (long long i = 0; i < nn; i++){
        pp[i] = i;
        pp[i+nn] = i;
    }
    
    if (evalf == true) {
        plhs = mxCreateDoubleMatrix((long)floor((double)iters / (nn)) + 1, 1, mxREAL);
        hist = mxGetPr(plhs);
    }
    
    //Interations
    for (long i = 1; i <= iters; i++){
        if (i%nn == 0){
            hist[(long)floor((double)i / (nn))] = compute_function_value_sparse(nn, dd, w, Xt, y, lambda, ir, jc);
        }
        compute_full_gradient_sparse(fullg, nn, dd, Xt, y, lambda, w, ir, jc);
        for (long k = 0; k < dd; k++){
            wold[k] = w[k];
            z[k] = w[k];
            yy[k] = w[k];
        }
        
        std::random_shuffle (&pp[0], &pp[nn]);
        std::random_shuffle (&pp[nn], &pp[2*nn]);
        
        for (long k = 0; k < dd; k++){
            www[k] = 0;
        }
        bb = 0;
        double power = 1/tau2;
        for (long j = 1; j <= 2*nn; j++){
            update_w(w, z, wold, yy, dd, tau, tau1);
            calculate_wg_sparse(wg, w, wold, pp, j, fullg, lambda, dd, Xt, ir, jc);
            for(long k = 0; k < dd; k++){
                zold[k] = z[k];
            }
            update_z(z, wg, dd, stepSize);
            update_y(yy, dd, w, wg, L);
            power *= tau2;
            for(long k = 0; k < dd; k++){
                www[k] = www[k] + power*yy[k];
            }
            bb += pow(tau2, j-1);
        }
        for (long k = 0; k < dd; k++){
            w[k] = www[k]/bb;
        }
    }
    hist[(long)floor((double)iters / (nn))] = compute_function_value_sparse(nn, dd, w, Xt, y, lambda, ir, jc);
    if (evalf == true) { return plhs; }
    else { return 0; }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if (mxIsSparse(prhs[1])) {
        plhs[0] = Katyusha_sparse(nlhs, prhs);
    }
    else {
        plhs[0] = Katyusha_dense(nlhs, prhs);
    }
}