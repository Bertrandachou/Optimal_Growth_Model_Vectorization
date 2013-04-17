from __future__ import division
import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:07:54 2013

@author: Bertrand Achou
"""

""" This is the deterministic optimal growth model"""



"""first we create a dictionary containing all our parameters of the model"""

p = {'alpha': 0.3, 'beta': 0.95, 'sigma': 1, 'delta': 1, 'A': 1 }


""" we then define the parameters of the grid in a list and the lists needed"""

kss = ( (1 - p['beta'] * (1 - p['delta']) ) / (p['alpha'] * p['beta']) )**( 1 / (p['alpha'] - 1))
sd  = 0.9



pgrid = { 'n': 100, 'kmin': kss*(1-sd), 'kmax': kss*(1+sd) }

kgrid    = np.array([np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n']) for i in range(pgrid['n'])])

kpgrid   = np.array( [np.array([kgrid[0,i] for j in range(pgrid['n'])]) for i in range(pgrid['n'])])

V0       = np.zeros(pgrid['n'])



""" we then define the functions of the model"""

def utility(c,p):
    # this is the utility function 
    # c is consumption
    # p is the dictionary of parameters
    if p['sigma'] == 1:
        return np.log(c)
    else:
        return ( c**(1 - p['sigma']) ) / ( 1 - p['sigma'] )
        

def production(k,p):
    # this is the production function of the model
    # k is capital
    # p is a dictionary of parameters
    return p['A'] * k**(p['alpha'])
    
    
def new_value(k,kp,V0,p,pgrid):
    # computes the new value function for k
    # given the matrix kp which is similar to k 
    # except that values are ordered in column (k is ordered in rows and is square)
    # knowing that the former guess on the value function was V0
    # and for a given dictionnary of parameters p
    
    # we use Boolean indexing checking the values kprime
    # that satisfy the constraint
    # when the resource constraint is not satisfied we impose a large penalty 
    # to the representative agent to ensure that he will never choose these 
    # values
    
    
    new_V             = np.zeros(pgrid['n'])    
    
    DV0               = np.array([[V0[i] for j in range(pgrid['n'])] for i in range(pgrid['n'])])
    
    budget_not        = ((production(k,p) + (1 - p['delta']) * k - kp) < 0)  # looks if the constraint is NOT satisfied
    
    DV0[budget_not]   = -999999999999  #gives a large penalty no to satisfy the constraint
    
    ctemp             = production(k,p) + (1 - p['delta']) * k - kp
    
    ctemp[budget_not] = 0.001          #gives a large penalty no to satisfy the constraint
    
    Vtemp             = utility(ctemp,p) + p['beta'] * DV0
    
    Vtemp2            = Vtemp.reshape(1,pgrid['n']*pgrid['n'])

    for i in range(pgrid['n']):
        
        new_V[i] = max(Vtemp2[0][i::pgrid['n']])
        
    return new_V
    




""" Now we run the algorithm """

crit    = 100
epsilon = 10**(-6)

kgrid2 = np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n'])

import pylab


while crit > epsilon:
    
    V = new_value(kgrid,kpgrid,V0,p,pgrid)
    crit = max(abs(V-V0))
    V0 = V
    
   
pylab.plot(kgrid2,V)