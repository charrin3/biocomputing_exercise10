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
    
    expected=B0+B1*obs.x+B2**obs.x
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
teststat=2*(fitquadratic.fun-fitlinear.fun)
df=len(fitquadratic.x)-len(fitquadratic.x)
1-stats.chi2.cdf(teststat,df)
