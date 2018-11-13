#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:36:19 2018

@author: charrington
"""

import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

data=pandas.read_csv("data.txt", sep=",", header=0)

#def custom likelihood function for quadratic
def nllikequadratic(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    
    expected=B0+B1*obs.x+B2*obs.x*obs.x
    nllquadratic=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nllquadratic

    #estimate parameters by minimizing neg log likelihood
    initialGuessquadratic=numpy.array([1,1,1,1])
    fitquadratic=minimize(nllikequadratic,initialGuessquadratic,method="Nelder-Mead",options={'disp':True},args=data)
    
    print (fitquadratic.x)
    
#def custom likelihood function for linear
def nllikelinear(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    
    expected=B0+B1*obs.x
    nlllinear=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nlllinear

    #estimate parameters by minimizing neg log likelihood
    initialGuesslinear=numpy.array([1,1,1])
    fitlinear=minimize(nllikelinear,initialGuesslinear,method="Nelder-Mead",options={'disp':True},args=data)
    
    print (fitlinear.x)

#which model is more appropriate
from scipy import stats
teststat=2.0*(fitquadratic.fun-fitlinear.fun)
df=(len(fitquadratic.x)-len(fitlinear.x))/1.0
1-stats.chi2.cdf(teststat,df)

#2
import scipy
import scipy.integrate as spint

def ddSim (y,t0,r,a):
    N1=y[0]
    N2=y[1]
    R1=r[0]
    R2=r[1]
    a11=a[0]
    a12=a[1]
    a21=a[2]
    a22=a[3]
    dN1dt=R1*(1-N1*a11-N2*a12)*N1
    dN2dt=R2*(1-N2*a22-N1*a21)*N2
    
    return [dN1dt,dN2dt]

#model simulation 1 a12<a11, a21<a22
params=([.05,.05],[.01,.005,.003,.02])
N0=[0.01,0.01]
times=range(0,600)

modelSim=spint.odeint(func=ddSim,y0=N0,t=times,args=params)
modelOutput=pandas.DataFrame({"t":times, "N1": modelSim2[:,0], "N2": modelSim2[:,1]})
ggplot(modelOutput)+geom_line(aes(x="t", y="N1"), color="red")+geom_line(aes(x="t", y="N2"), color="blue")

#model simulation 2 a12>a11, a21<a22
params=([.05,.05],[.1,3,.003,.02])
N0=[0.01,0.01]
times=range(0,600)

    
