from __future__ import division

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:07:54 2013

@author: Bertrand Achou
"""

""" This is the deterministic optimal growth model"""

import numpy as np

"""first we create a dictionary containing all our parameters of the model"""

p = {'alpha': 0.3, 'beta': 0.95, 'sigma': 1, 'delta': 0.3 }


"""create a dictionary for the shocks"""

shock = { 'A': np.array([0.85,1.15]) , 'transit': np.array([[0.5,0.5],[0.5,0.5]]) } 



kss = ( (1 - p['beta'] * (1 - p['delta']) ) / (p['alpha'] * p['beta']) )**( 1 / (p['alpha'] - 1))
sd  = 0.9


pgrid = { 'n': 10, 'kmin': kss*(1-sd), 'kmax': kss*(1+sd) }

Agrid = np.array([[[shock['A'][i] for j in xrange(pgrid['n'])] for k in xrange(pgrid['n'])] for i in xrange(len(shock['A']))])

kgrid    = np.array([[np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n']) for i in xrange(pgrid['n'])] for k in xrange(len(shock['A']))])

kpgrid   = np.array( [ [np.array([kgrid[0,0,i] for j in xrange(pgrid['n'])]) for i in xrange(pgrid['n'])] for k in xrange(len(shock['A']))] )


# we build our guess value function
# first line of the matrix corresponds to the first level of productivity
# first column corresponds to the first level of capital 
    
V0 = np.array([ [ np.zeros(pgrid['n']) for k in xrange(pgrid['n']) ] for j in xrange(len(shock['A']))] )   

"""
V00       = np.zeros(pgrid['n'])          # value function for first state of productivity
V01       = np.zeros(pgrid['n'])          # value function for second state of productivity 
"""

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
    
    
def new_value(k,kp,V0,p,pgrid,A,shock):
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
    
    
    new_V0 = np.array([ [ np.zeros(pgrid['n']) for k in xrange(pgrid['n']) ] for j in xrange(len(shock['A']))] )   

    #new_V00            = np.zeros(pgrid['n'])    
    #new_V01            = np.zeros(pgrid['n']) 
    
    
    
    #DV00               = np.array([[V00[i] for j in range(pgrid['n'])] for i in range(pgrid['n'])])
    #DV01               = np.array([[V01[i] for j in range(pgrid['n'])] for i in range(pgrid['n'])])                
    
    budget_not           = ((production(k,p,A) + (1 - p['delta']) * k - kp) < 0)
    
    #budget_not0         = ((production(k,p,shock['A'][0]) + (1 - p['delta']) * k - kp) < 0)  # looks if the constraint is NOT satisfied
    #budget_not1         = ((production(k,p,shock['A'][1]) + (1 - p['delta']) * k - kp) < 0)
    

    
    #DV00[budget_not0]   = -999999999999  #gives a large penalty no to satisfy the constraint
    #DV01[budget_not1]   = -999999999999
    
    
    ctemp               = production(k,p,A) + (1 - p['delta']) * k - kp
    
    #ctemp0             = production(k,p,shock['A'][0]) + (1 - p['delta']) * k - kp
    #ctemp1             = production(k,p,shock['A'][1]) + (1 - p['delta']) * k - kp
    
    
    ctemp [budget_not]  = 0.01
    
    
    #ctemp0[budget_not0] = 0.001          #gives a large penalty no to satisfy the constraint
    #ctemp1[budget_not1] = 0.001
    
    
    utemp               = utility(ctemp,p)
    
    for i in xrange(len(shock['A'])):
        
        utemp[i] += sum( [shock['transit'][i,j]*V0[j] for j in range(len(shock['A'])) ] )
    
    
    #Vtemp0             = utility(ctemp0,p) + p['beta'] * ( shock['transit'][0,0]*DV00 + shock['transit'][0,1]*DV01 )
    #Vtemp1             = utility(ctemp1,p) + p['beta'] * ( shock['transit'][1,0]*DV00 + shock['transit'][1,1]*DV01 )
    
    
    utemp[budget_not] = -99999999
    
    Vtemp        = utemp.reshape(len(shock['A']),pgrid['n']*pgrid['n'])
    
    # Vtemp02            = Vtemp0.reshape(1,pgrid['n']*pgrid['n'])
    #Vtemp12            = Vtemp1.reshape(1,pgrid['n']*pgrid['n'])

    for i in xrange(len(shock['A'])):
        
        for j in xrange(pgrid['n']):
            
            new_V0[i,j,:] = max(Vtemp[i][j::pgrid['n']])
        
        #new_V00[i] = max(Vtemp02[0][i::pgrid['n']])
        #new_V01[i] = max(Vtemp12[0][i::pgrid['n']])
        
        
        
    return new_V0
    




""" Now we run the algorithm """

crit    = 100
epsilon = 10**(-6)

while crit > epsilon:
    
    grid_new  = new_value(kgrid,kpgrid,V0,p,pgrid,Agrid,shock)
    
    critmat      = abs(V0-grid_new).reshape(1,pgrid['n']*pgrid['n']*len(shock['A']))
    crit         = max(critmat[0])
    
    
    V0        = grid_new
    
    
    #grid_new = new_value(kgrid,kpgrid,V00,V01,p,pgrid,shock)
    #V0       = grid_new[0]
    #V1       = grid_new[1]
    #crit1    = max(abs(V0-V00))
    #crit2    = max(abs(V1-V01))
    #crit     = max([crit1,crit2])
    #V00 = V0
    #V01 = V1




    
import pylab

kgrid2 = np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n'])
   
pylab.plot(kgrid2,V0[0,:,0])
#pylab.plot(kgrid2,V01)

