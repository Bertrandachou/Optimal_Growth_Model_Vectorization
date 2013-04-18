from __future__ import division

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:07:54 2013

@author: Bertrand Achou
"""

""" This is the deterministic optimal growth model"""

import numpy as np

"""first we create a dictionary containing all our parameters of the model"""

p = {'alpha': 0.3, 'beta': 0.95, 'sigma': 1, 'delta': 1 }


"""create a dictionary for the shocks"""

shock = { 'A': np.array([0.7,1.3]) , 'transit': np.array([[0.5,0.5],[0.5,0.5]]) } 

kss = ( (1 - p['beta'] * (1 - p['delta']) ) / (p['alpha'] * p['beta']) )**( 1 / (p['alpha'] - 1))
sd  = 0.9



pgrid = { 'n': 100, 'kmin': kss*(1-sd), 'kmax': kss*(1+sd) }

kgrid    = np.array([np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n']) for i in range(pgrid['n'])])

kpgrid   = np.array( [np.array([kgrid[0,i] for j in range(pgrid['n'])]) for i in range(pgrid['n'])])

V00       = np.zeros(pgrid['n'])          # value function for first state of productivity
V01       = np.zeros(pgrid['n'])          # value function for second state of productivity 


""" we then define the functions of the model"""

def utility(c,p):
    # this is the utility function 
    # c is consumption
    # p is the dictionary of parameters
    if p['sigma'] == 1:
        return np.log(c)
    else:
        return ( c**(1 - p['sigma']) ) / ( 1 - p['sigma'] )
        

def production(k,p,A):
    # this is the production function of the model
    # k is capital
    # p is a dictionary of parameters
    return A * k**(p['alpha'])
    
    
def new_value(k,kp,V00,V01,p,pgrid,shock):
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
    
    
    new_V00            = np.zeros(pgrid['n'])    
    new_V01            = np.zeros(pgrid['n']) 
    
    DV00               = np.array([[V00[i] for j in range(pgrid['n'])] for i in range(pgrid['n'])])
    DV01               = np.array([[V01[i] for j in range(pgrid['n'])] for i in range(pgrid['n'])])                
    
    budget_not0         = ((production(k,p,shock['A'][0]) + (1 - p['delta']) * k - kp) < 0)  # looks if the constraint is NOT satisfied
    budget_not1         = ((production(k,p,shock['A'][1]) + (1 - p['delta']) * k - kp) < 0)
    
    
    DV00[budget_not0]   = -999999999999  #gives a large penalty no to satisfy the constraint
    DV01[budget_not1]   = -999999999999
    
    
    ctemp0             = production(k,p,shock['A'][0]) + (1 - p['delta']) * k - kp
    ctemp1             = production(k,p,shock['A'][1]) + (1 - p['delta']) * k - kp
    
    ctemp0[budget_not0] = 0.001          #gives a large penalty no to satisfy the constraint
    ctemp1[budget_not1] = 0.001
    
    
    Vtemp0             = utility(ctemp0,p) + p['beta'] * ( shock['transit'][0,0]*DV00 + shock['transit'][0,1]*DV01 )
    Vtemp1             = utility(ctemp1,p) + p['beta'] * ( shock['transit'][1,0]*DV00 + shock['transit'][1,1]*DV01 )
    
    Vtemp02            = Vtemp0.reshape(1,pgrid['n']*pgrid['n'])
    Vtemp12            = Vtemp1.reshape(1,pgrid['n']*pgrid['n'])

    for i in range(pgrid['n']):
        
        new_V00[i] = max(Vtemp02[0][i::pgrid['n']])
        new_V01[i] = max(Vtemp12[0][i::pgrid['n']])
        
        
        
    return np.array([new_V00,new_V01])
    




""" Now we run the algorithm """

crit    = 100
epsilon = 10**(-6)

while crit > epsilon:
    
    grid_new = new_value(kgrid,kpgrid,V00,V01,p,pgrid,shock)
    V0       = grid_new[0]
    V1       = grid_new[1]
    crit = max([max(abs(V0-V00)),max(abs(V0-V01))])
    V00 = V0
    V01 = V0

    
import pylab

kgrid2 = np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n'])
   
pylab.plot(kgrid2,V00)
pylab.plot(kgrid2,V01)

