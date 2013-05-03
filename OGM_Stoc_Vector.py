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

shock = { 'A': np.array([0.85,0.9,0.95,1,1.05,1.1,1.15]) , 'transit': np.array([[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.1]]) } 



kss = ( (1 - p['beta'] * (1 - p['delta']) ) / (p['alpha'] * p['beta']) )**( 1 / (p['alpha'] - 1))
sd  = 0.9


pgrid = { 'n': 100, 'kmin': kss*(1-sd), 'kmax': kss*(1+sd) }

Agrid = np.array([[[shock['A'][i] for j in xrange(pgrid['n'])] for k in xrange(pgrid['n'])] for i in xrange(len(shock['A']))])

kgrid    = np.array([[np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n']) for i in xrange(pgrid['n'])] for k in xrange(len(shock['A']))])

kpgrid   = np.array( [ [np.array([kgrid[0,0,i] for j in xrange(pgrid['n'])]) for i in xrange(pgrid['n'])] for k in xrange(len(shock['A']))] )


# we build our guess value function
# first line of the matrix corresponds to the first level of productivity
# first column corresponds to the first level of capital 
    

V0 = np.zeros( ( len(shock['A']), pgrid['n'] ) )


"""
V00       = np.zeros(pgrid['n'])          # value function for first state of productivity
V01       = np.zeros(pgrid['n'])          # value function for second state of productivity 
"""

""" we then define the functions of the model"""

import numexpr
def utility(c,p):
    # this is the utility function 
    # c is consumption
    # p is the dictionary of parameters

    ps = p['sigma']
    if ps == 1:
        return numexpr.evaluate( 'log(c)' )
    else:
        return numexpr.evaluate( 'c**(1 - ps) / ( 1 - ps )' )
        


def production(k,p,A):
    # this is the production function of the model
    # k is capital
    # p is a dictionary of parameters
    pp = p['alpha']
    return numexpr.evaluate('A * k**pp')
    
    
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
    
    
#    new_V0 = np.array([ [ np.zeros(pgrid['n']) for z in xrange(pgrid['n']) ] for j in xrange(len(shock['A']))] )   
#    new_V0 = np.zeros( ( len(shock['A']), pgrid['n'], pgrid['n'] ) )


    prod = production(k, p, A)
    ctemp               = prod + (1 - p['delta']) * k - kp
    budget_not           = ctemp < 0    
    ctemp[budget_not]  = 0.001  
   
    utemp               = utility(ctemp,p)
    utemp[budget_not] = -99999999

    P = shock['transit']
    VP0 = utemp + (p['beta'] * np.dot( P, V0 ))[:,:,None]

    new_V0 = VP0.max(axis=1)
            
             
    return new_V0
    


""" Now we run the algorithm """

def solve_value( kgrid,kpgrid,V0,p,pgrid,Agrid,shock ):

    import time

    crit    = 100
    epsilon = 10**(-6)
    maxit = 1000
    it = 0
    while crit > epsilon and it < maxit:
    
        it += 1
    
        t1 = time.time()
    
        TV  = new_value(kgrid,kpgrid,V0,p,pgrid,Agrid,shock)
        
        crit      = abs(V0-TV).max()
    
        t2 = time.time()
       
        print('Iteration {:5} . Error : {:6.3f}. Elapsed : {:6.3f}'.format(it, crit, t2-t1))
        
        V0        = TV

    return V0

#V0 = solve_value( kgrid,kpgrid,V0,p,pgrid,Agrid,shock )

#import pylab

#kgrid2 = np.linspace(pgrid['kmin'],pgrid['kmax'],pgrid['n'])
   
#pylab.plot(kgrid2,V0[0,0,:])
#pylab.plot(kgrid2,V0[1,0,:])
#pylab.plot(kgrid2,V0[2,0,:])
#pylab.plot(kgrid2,V0[3,0,:])
#pylab.plot(kgrid2,V0[4,0,:])
#pylab.plot(kgrid2,V0[5,0,:])
#pylab.plot(kgrid2,V0[6,0,:])
