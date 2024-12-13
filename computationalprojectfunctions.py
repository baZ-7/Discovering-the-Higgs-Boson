# -*- coding: utf-8 -*-
"""
Project 2: Discovering the Higgs Boson 
Functions 
by: Pit On (Andrew) Chim
"""

#importing relevant modules
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import special 

#%%

def Gaussian(x, mu, sigma):
    '''
    Parameters
    ----------
    x : the x value of the gaussian function
    mu : mean
    sigma : standard deviation

    Returns the value of the gaussian 
    -------
    None.
    '''
    normalval = 1 / (2*np.pi)**0.5 
    exponentval = np.exp(-(x - mu)**2/(2*sigma**2))
    return normalval*exponentval 

#x = np.linspace(-5,5,100)
#y = Gaussian(x, 0, 1)
#plt.plot(x,y)

#%%

def Integrate(x1, x2, f, f_param, param):
    if x2 >= 0.81:
        x = Simpsons2(x1, 100, f, f_param, param)[0] - Simpsons2(x2, 100, f, f_param, param)[0]
    else:
        x = AM4(x1, x2, 0, f, f_param, param)[0]
    
    return x

#%%

def Trapezoidal2(x1, x2, f, f_param, N):
    '''
    Parameters
    ----------
    x1 : starting x-point
    x2 : ending x-point
    f : function used
    f_param : function parameters
    N : number of iterations
    Returns
    -------
    Approximate Integration 

    '''
    h = (x2 - x1) / N
    S = 0
    vals = []
    for i in range(0,N):
        vals.append(S)
        fa = f(x1 + i*h, *f_param)
        fb = f(x1 + (i+1)*h, *f_param)
        S += ((1/2)*fa + (1/2)*fb)
    return S*h, vals
    
#%%

def Trapezoidal(x1, x2, f, f_param, epsilon):
    '''
    Parameters
    ----------
    x1 : left-bound of sample taken
    x2 : right-bound of sample taken
    f : function examined
    f_param : parameters for the function
    epsilon : convergence value 

    Returns : Integrand approximation between samples using Trapezoidal Rule
    -------
    None.
    '''
    iteration = 1 #iteration number
    y1 = f(x1, *f_param) 
    y2 = f(x2, *f_param) #evaluate inital points
    initial_stepsize = (x2 - x1) 
    initial_estimate = initial_stepsize * 0.5 * (y1 + y2)
    percent_diff = 1 #arbitrary initial value 
    vals = []
    
    while np.all(percent_diff > epsilon):
        new_samples = [] #empty array
        for i in range(1,2**(iteration-1)+1):
            new_samples.append(f(x1 + ((2*i - 1)*initial_stepsize)/2, *f_param))
        T_estimate = 0.5 * initial_estimate + 0.5 * initial_stepsize * np.sum(new_samples)
        initial_stepsize /= 2 #update stepsize
        percent_diff = np.abs((T_estimate - initial_estimate) / initial_estimate)
        initial_estimate = T_estimate
        vals.append(initial_estimate)
        iteration += 1 #number of iterations counter
    
    return initial_estimate, vals

#test1, test2 = Trapezoidal(-5,5,Gaussian,[0,1],10e-15)
#print('Trapezoid Rule:',test1)

#%%

def Simpsons2(x1, x2, f, f_param, N):
    '''
    Parameters
    ----------
    x1 : starting x-point
    x2 : ending x-point
    f : function used
    f_param : function parameters
    N : number of iterations
    Returns
    -------
    Approximate Integration 

    '''
    h = (x2 - x1) / N
    hh = int(N / 2)
    S = 0
    vals=[]
    for i in range(0,hh):
        vals.append(S*h)
        fa = f(x1 + i*2*h, *f_param)
        fb = f(x1 + (i*2+1)*h, *f_param)
        fc = f(x1 + (i*2+2)*h, *f_param)
        S += ((1/3)*fa + (4/3)*fb + (1/3)*fc)
    return S*h, vals

#print('Simp2',Simpsons2(0,5,Gaussian,[0,1],1e5))

#%%

def Simpsons(x1, x2, f, f_param, epsilon):
    '''
    Parameters
    ----------
    x1 : starting x-point 
    x2 : ending x-point 
    f : function examined
    f_param : parameters for the function
    epsilon : convergence value 

    Returns : Integrand approximation using Extended Simpson's Rule
    -------
    None.
    '''
    iteration = 1 #iteration number
    y1 = f(x1, *f_param) 
    y2 = f(x2, *f_param) #evaluate inital points
    initial_stepsize = (x2 - x1) 
    initial_estimate = initial_stepsize * 0.5 * (y1 + y2)
    percent_diff = 1 #arbitrary initial value 
    vals = []
    
    Sarr = [100]
    while np.all(percent_diff > epsilon):
        new_samples = [] #empty array
        for i in range(1,2**(iteration-1)+1):
            new_samples.append(f(x1 + ((2*i - 1)*initial_stepsize)/2, *f_param))
        T_estimate = 0.5 * initial_estimate + 0.5 * initial_stepsize * np.sum(new_samples)
        Sarr.append((4/3)*T_estimate - (1/3)*initial_estimate)
        initial_stepsize /= 2 #update stepsize
        percent_diff = np.abs((Sarr[-1] - Sarr[-2]) / Sarr[-2])
        initial_estimate = T_estimate
        vals.append(initial_estimate)
        iteration += 1 #number of iterations counter
    
    return Sarr[-1], vals

#print('Simpsons Rule:',Simpsons(-5,5,Gaussian,[0,1],10e-15))

#%%

def MonteCarlo(xmax, xmin, f, f_param, N):
    '''
    Parameters
    ----------
    xmax : x maximum bound
    xmin : x minimum bound
    f : function examined
    f_param : parameters for the function
    N : number of random samples

    Returns : Integrand approximation using Monte Carlo Integration 
    -------
    None.
    '''
    xarr = np.random.rand(N)*(xmax - xmin) + xmin
    yarr = []
    for i in range(N):
        y_curve = f(xarr[i], *f_param)
        yarr.append(y_curve) 
    average_y = np.sum(yarr) / N 
    prediction = average_y * (xmax - xmin)
    error = np.std(yarr) * (xmax - xmin) / np.sqrt(N)
    
    return prediction, error 

#monte1, monte2 = MonteCarlo(100,-100,Gaussian,[0,1],100000)
#print('Monte Carlo Method:', monte1, '+-', monte2)

#%%

def AB4(x1, x2, u0, f, f_param, N):
    '''
    Parameters
    ----------
    x1 : starting point in x
    x2 : end point in x
    f : function examined
    f_param : parameters for function
    Returns : Integrand approximation using fourth-order Adams-Bashforth method
    -------
    None.
    '''
    dt = (x2 - x1) / N
    u1 = u0 + f(x1,*f_param)*dt #Euler
    u2 = u1 + (3*f(x1 + dt, *f_param) - f(x1, *f_param))*dt/2 #AB2
    u3 = u2 + (23*f(x1 + 2*dt, *f_param) - 16*f(x1 + dt, *f_param) + 5*f(x1, *f_param))*dt/12 #AB3
    un = u3
    vals = []
    for i in range(3,N):
        fa = f(x1 + i*dt, *f_param)
        fb = f(x1 + (i-1)*dt, *f_param)
        fc = f(x1 + (i-2)*dt, *f_param)
        fd = f(x1 + (i-3)*dt, *f_param)
        un1 = un + (1/24) * (55*fa - 59*fb + 37*fc - 9*fd) * dt
        un = un1
        vals.append(un)

    return un, vals

#print('4th order Adams-Bashforth:',AB4(-5,5,0,Gaussian,[0,1],100000))

#%%

def AM4(x1, x2, u0, f, f_param, N):
    '''
    Parameters
    ----------
    x1 : starting point in x
    x2 : end point in x
    f : function examined
    f_param : parameters for function
    Returns : Integrand approximation using fourth-order Adams-Bashforth method
    -------
    None.
    '''
    dt = (x2 - x1) / N
    u1 = u0 + f(x1,*f_param)*dt #Euler
    u2 = u1 + (3*f(x1 + dt, *f_param) - f(x1, *f_param))*dt/2 #AB2
    un = u2
    vals = []
    for i in range(2,N):
        fa = f(x1 + (i+1)*dt, *f_param)
        fb = f(x1 + (i)*dt, *f_param)
        fc = f(x1 + (i-1)*dt, *f_param)
        fd = f(x1 + (i-2)*dt, *f_param)
        un1 = un + (1/24) * (9*fa + 19*fb - 5*fc + fd) * dt
        un = un1
        vals.append(un)

    return un, vals

#print('4th order Adams-Moulton:',AM4(-5,5,0,Gaussian,[0,1],100000)[0])

#%%

def RK4(x1, x2, u0, f, f_param, N):
    dt = (x2 - x1) / N
    un = u0
    vals = []
    x = x1
    while x <= x2:
        fa = f(x, *f_param)
        fbc = f(x + 0.5*dt, *f_param)
        fd = f(x + 1*dt, *f_param)
        un1 = un + (1/6) * (fa + 4*fbc + fd) * dt
        un = un1
        x += dt
        vals.append(un)

    return un , vals

#rk4_finalval, rk4_itervals = RK4(0,5,0,Gaussian,[0,1],100000)
#print('4th order Runge-Kutta:',rk4_finalval)
    
    #%%

def MassFunc(m, m_higgs):
    '''
    Parameters
    ----------
    m : mass input
    m_higgs : mass of higgs
    Returns : value of mass function
    -------
    None.
    '''
    A = 1500 #(GeV/c^2)^-1
    k = 20 #GeV/c^2
    exp_val = np.exp(- (m - m_higgs) / k)
    return A * exp_val

#x = np.linspace(0, 300, 100)
#y = MassFunc(x, 125.1)
#plt.plot(x,y)
#plt.show()

def Poisson(x, lamb):
    '''
    Parameters
    ----------
    x : input
    lamb : mean value / frequency
    Returns : value of Poisson distribution
    -------
    None.
    '''
    fac_val = special.factorial(x)
    exp_val = np.exp(-lamb)
    return lamb**x * exp_val / fac_val

#x = np.linspace(0, 150, 100)
#y = Poisson(x, 100)
#plt.plot(x,y)
#plt.show()

#%%

def Significance(ml, mu):
    '''
    Parameters
    ----------
    mu : mass cut upper bound
    ml : mass cut lower bound
    Returns : significance on the mass cuts
    -------
    None.
    '''
    Nb = -(20)*(MassFunc(mu, 125.1) - MassFunc(ml, 125.1)) #Analytically
    Nh = 470 * (1/1.4) * Integrate(ml, mu, Gaussian, [125.1, 1.4], 100000)
    #1/1.4 to normalize the Gaussian Function!
    #print(Nb, Nh)
    return Nh / np.sqrt(Nb)

#print(Significance(120,130))

#%%

def CentralDiff2D(x1, x2, step, f):
    before1 = f(x1 - step, x2)
    after1 = f(x1 + step, x2)
    before2 = f(x1, x2 - step)
    after2 = f(x1, x2 + step)
    grad = [(after1 - before1) / (2*step), (after2 - before2) / (2*step)]
    return np.array(grad)

#print(CentralDiff2D(120,130,1e-10,Significance))
    
def GradientDescent(start, alpha, h, f, max_iter):
    '''
    Parameters
    ----------
    start : starting coordinates
    alpha : learning rate
    h : step size
    f : function that is optimized
    f_param : parameters for the functions
    max_iter : maximum of iterations
    Returns
    -------
    None.
    '''
    change = 100
    num_iter = 0
    x = start
    locationx = []
    locationy = []
    
    while num_iter < max_iter and change > 10e-20:
        grad = CentralDiff2D(x[0], x[1], h, f)
        x_new = x + alpha * grad
        num_iter += 1
        change = f(x_new[0],x_new[1]) - f(x[0],x[1]) 
        x = x_new
        locationx.append(x[0])
        locationy.append(x[1])
        print('iter',num_iter,'x',x,'S',f(x[0],x[1]))
    
    return x, locationx, locationy

#print(GradientDescent([123.5,127.5], 1e-2, 10e-10, Significance, 1e10))
        
#%%       
   
def QuasiNewton(start, alpha, h, f, max_iter):
    '''
    Parameters
    ----------
    start : starting coordinates
    alpha : learning rate
    h : step size
    f : function that is optimized
    f_param : parameters for the functions
    max_iter : maximum of iterations
    Returns
    -------
    None.
    '''
    change = 100
    num_iter = 0
    locationx = []
    locationy = []
    x = np.array(start)
    G = np.mat([[1,0],[0,1]])
    
    
    while num_iter < max_iter and abs(change) > 10e-16:
        locationx.append(x[0])
        locationy.append(x[1])
        grad = np.mat(CentralDiff2D(x[0], x[1], h, f))
        x_new = np.array(np.mat(x).T + alpha * np.mat(G) * grad.T)
        xc = np.mat(x_new - np.mat(x).T)
        gc = np.mat(CentralDiff2D(x_new[0],x_new[1],h,f).T - CentralDiff2D(x[0],x[1],h,f)).T
        a1 = np.outer(xc,xc)
        a2 = gc.T*xc
        b1 = G*np.outer(gc,gc)*G
        b2 = gc.T*G*gc
        G = G + (a1)/(a2) - (b1)/(b2)
        num_iter += 1
        change = f(x_new[0],x_new[1]) - f(x[0],x[1]) 
        x = np.array(x_new.T)[0]
        print('iter',num_iter,'x',x,'S',f(x[0],x[1]))
    
    return x, locationx, locationy

#x, locationx, locationy = QuasiNewton([123.2,137.2], 1e-5, 1e-10, Significance, 1000)
#plt.plot(locationx,locationy)
#plt.show()
    
#%%

def Metropolis(delta_energy, temperature):
    kb = 1 #1.380649e-23
    if delta_energy <= 0:
        return np.exp(delta_energy / (kb * temperature))
    else:
        return 1.0
    
def MC_TA(start, alpha, temperature, iterations, cooling_rate):
    '''
    Parameters
    ----------
    start : starting coordinates
    alpha : step value
    temperature : temperature for thermal annealing 
    iterations : number of iterations taking
    cooling_rate : cooling rate for thermal annealing

    Returns
    -------
    current : optimal point
    current_energy : optimized value
    '''
    locationx = []
    locationy = []
    current = start
    current_energy = Significance(current[0],current[1])
    
    for i in range(iterations):
        new = [current[0]+np.random.uniform(-alpha,alpha),current[1]+np.random.uniform(-alpha,alpha)]
        new_energy = Significance(new[0],new[1])
        delta_energy = new_energy - current_energy
        
        if np.random.rand() <= Metropolis(delta_energy,temperature):
            current = new
            current_energy = new_energy
        locationx.append(current[0])
        locationy.append(current[1])
        temperature *= cooling_rate
        print('iter',i,'x',current,'S',current_energy)
    
    return current, current_energy, locationx, locationy


#test 1 [123.2375162063974, 127.15848370472592] 1e-9
#test 2 [123.23751632488785, 127.15848371533993] 1e-5 from [123,127]
#x, s, locationx, locationy = MC_TA([120,130],1e-3,100,5000,0.99)
#plt.plot(locationx,locationy)
#plt.show()
#print(Significance(123.2375162063974, 127.15848370472592))

#%%



    
