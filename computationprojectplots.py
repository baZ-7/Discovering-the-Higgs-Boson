# -*- coding: utf-8 -*-
"""
Project 2: Discovering the Higgs Boson 
Tests & Plots 
by: Pit On (Andrew) Chim
"""
#importing relevant modules
from computationalprojectfunctions import *
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import special 
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import LinearLocator
plt.rcParams.update({'font.size':12})

#%%

sns.set_style("whitegrid")

#%%
true_val = special.erf(5/np.sqrt(2))*0.5
Nvals = np.linspace(2,50,25)
simpNvals = []
trapNvals = []
am4Nvals = []
#mcNvals = []
for i in range(len(Nvals)):
    simpNvals.append(1/(abs(Simpsons2(0, 5, Gaussian, [0,1], int(Nvals[i]))[0]-true_val)))
    trapNvals.append(1/(abs(Trapezoidal2(0, 5, Gaussian, [0,1], int(Nvals[i]))[0]-true_val)))
    #am4Nvals.append(1/(abs(AM4(0, 5, 0, Gaussian, [0,1], int(Nvals[i]))[0]-true_val)))
    #mcNvals.append(1/(abs(MonteCarlo(5, 0, Gaussian, [0,1], int(Nvals[i]))[0]-true_val)))
    
#%%

fig, axs = plt.subplots(nrows=2, sharex=True)
fig.tight_layout(pad=1)
psimp = np.poly1d( np.polyfit(Nvals, simpNvals, 4) )
ptrap = np.poly1d( np.polyfit(Nvals, trapNvals, 2) )
axs[0].plot(Nvals, psimp(Nvals), 'k--', label='4th-order Polyfit')
axs[1].plot(Nvals, ptrap(Nvals), 'k--', label='2nd-order Polyfit')
axs[0].plot(Nvals, simpNvals, 'bx', label='Simpsons') 
axs[1].plot(Nvals, trapNvals, 'rx', label='Trapezoidal')
#axs[0].plot(Nvals, am4Nvals, 'gx', label='AM4')
axs[0].set_xlim([1.9,50.1])
axs[1].set_xlim([1.9,50.1])
axs[0].legend()
axs[1].legend()
axs[1].set_xlabel('Number of iteration $N$')
axs[0].set_ylabel('$Error^{-1}$')
axs[1].set_ylabel('$Error^{-1}$')
axs[0].set_title('Fig 1.a')
axs[1].set_title('Fig 1.b')
plt.savefig('Figure 0.5.jpg', dpi=200)
#plt.plot(Nvals, mcNvals, 'x')
plt.show()

#%%

'''
Investigating the different integration methods
'''

iter_num = 100000
N = 100 #intervals for the test
a_arr = np.linspace(0.0001,5,N)
#trap_arr = []
simp_arr = []
simpb_arr = []
#monte_arr = []
#ab4_arr = []
true_arr = []
am4_arr = []
am4b_arr = []
#rk4_arr = []
for i in range(N):
    #trap_arr.append(Trapezoidal(0,a_arr[i],Gaussian,[0,1],10e-14)[0])
    simp_arr.append(Simpsons2(0,a_arr[i],Gaussian,[0,1],iter_num)[0])
    simpb_arr.append(Simpsons2(0,100,Gaussian,[0,1],iter_num)[0] - Simpsons2(a_arr[i],100,Gaussian,[0,1],iter_num)[0])
    #monte_arr.append(MonteCarlo(a_arr[i],0,Gaussian,[0,1],100000)[0])
    #ab4_arr.append(AB4(0,a_arr[i],0,Gaussian,[0,1],iter_num)[0])
    am4_arr.append(AM4(0,a_arr[i],0,Gaussian,[0,1],iter_num)[0])
    am4b_arr.append(AM4(0,100,0,Gaussian,[0,1],iter_num)[0] - AM4(a_arr[i],100,0,Gaussian,[0,1],iter_num)[0])
    #rk4_arr.append(RK4(0,a_arr[i],0,Gaussian,[0,1],iter_num)[0])
    true_arr.append(special.erf(a_arr[i]/np.sqrt(2))*0.5)

#trap_arr = np.array(trap_arr)
simp_arr = np.array(simp_arr)
simpb_arr = np.array(simpb_arr)
#monte_arr = np.array(monte_arr)
#ab4_arr = np.array(ab4_arr)
true_arr = np.array(true_arr)
am4_arr = np.array(am4_arr)
am4b_arr = np.array(am4b_arr)
#rk4_arr = np.array(rk4_arr)


#%%

#plt.plot(a_arr, (trap_arr - true_arr) / true_arr, label='Trapezoidal')
plt.plot(a_arr, np.abs((simp_arr - true_arr) / true_arr), label='Simpsons - Direct')
plt.plot(a_arr, np.abs((simpb_arr - true_arr) / true_arr), label='Simpsons - Difference')
#plt.plot(a_arr, (monte_arr - true_arr) / true_arr, label='Monte Carlo')
#plt.plot(a_arr, (ab4_arr - true_arr) / true_arr, label='4th-order Adams-Bashforth')
plt.plot(a_arr, np.abs((am4_arr - true_arr) / true_arr), 'k--', label='AM4 - Direct')
plt.plot(a_arr, np.abs((am4b_arr - true_arr) / true_arr), label='AM4 - Difference')
cutoff = 0.81
plt.axvline(x=cutoff,color='red',linestyle='--',label='cut-off value')
plt.text(0.15, 4e-14, 'a = 0.81', color='red', fontsize=15)
#plt.plot(a_arr, (rk4_arr - true_arr) / true_arr, label='4th-order Runge-Kutta')
#plt.plot(a_arr, true_arr)
plt.legend(loc='upper center')
plt.xlim(0,5)
plt.ylim(-0.01e-14,5e-14)
plt.xlabel('a value')
plt.ylabel('Relative Error')
#plt.title('Relative Error Comparison')
plt.savefig('Figure 1', dpi=200)
plt.show()

#%%

iter_num = 100000
itersimp = int(iter_num/2)
a_val = 1
true_val = special.erf(a_val/np.sqrt(2))*0.5
true_gal = []
true_gal2a = []
true_gal2b = []
for i in range(itersimp):
    true_gal.append(special.erf((a_val*(i/itersimp))/np.sqrt(2))*0.5)
    true_gal2a.append(special.erf((100*((i)/itersimp))/np.sqrt(2))*0.5)
    true_gal2b.append(special.erf((a_val+((100 - a_val)*((i)/itersimp)))/np.sqrt(2))*0.5 - true_val)
simp1final, simp1vals = Simpsons2(0, a_val, Gaussian, [0,1], iter_num)
simp2afinal, simp2avals = Simpsons2(0, 100, Gaussian, [0,1], iter_num)
simp2bfinal, simp2bvals = Simpsons2(a_val, 100, Gaussian, [0,1], iter_num)
am41final, am41vals = AM4(0, a_val, 0, Gaussian, [0,1], iter_num+2)
am42afinal, am42avals = AM4(0, 100, 0, Gaussian, [0,1], iter_num+2)
am42bfinal, am42bvals = AM4(a_val, 100, 0, Gaussian, [0,1], iter_num+2)
simp1vals = np.array(simp1vals)
simp2avals = np.array(simp2avals)
simp2bvals = np.array(simp2bvals)
am41vals = np.array(am41vals)
am42avals = np.array(am42avals)
am42bvals = np.array(am42bvals)
true_gal = np.array(true_gal)
true_gal2 = np.array(true_gal2a) - np.array(true_gal2b)

#%%
simpx = np.linspace(1,iter_num,itersimp)
am4x = np.linspace(1,iter_num,iter_num)
plt.plot(simpx, np.log10(np.abs((simp1vals - true_gal)/true_gal)),label='Simpsons - Direct')
plt.plot(simpx, np.log10(np.abs((simp2avals - simp2bvals - true_gal2)/true_gal2)),label='Simpsons - Difference')
plt.plot(am4x, np.log10(np.abs((am41vals - true_val)/true_val)),label='AM4 - Direct')
plt.plot(am4x, np.log10(np.abs((am42avals - am42bvals - true_val)/true_val)),label='AM4 - Difference')
plt.legend()
#plt.ylim(-1e-9,1e-3)
plt.xlabel('Iteration Number')
plt.ylabel('Order of Relative Error')
#plt.xlim(0,iter_num)
plt.savefig('Figure 1.5.png', dpi=200)
plt.show()

#%%

'''
Investigate the effect of varying parameters for the different integration methods
'''
narr = [10,50,100,500,1000,5000,10000,50000,100000,500000,1000000]
orderarr = [1,1.5,2,2.5,3,3.5,4,4.5,5]
simparr1 = []
am4arr1 = []
simparr2 = []
am4arr2 = []
mc1arr = []
true_val = special.erf(5/np.sqrt(2))*0.5
for i in range(len(narr)):
    s1 = Simpsons2(0,5,Gaussian,[0,1],narr[i])[0]
    simparr1.append((s1 - true_val) / true_val)
    s2 = Simpsons2(0,10,Gaussian,[0,1],narr[i])[0] - Simpsons2(5,10,Gaussian,[0,1],narr[i])[0]
    simparr2.append((s2 - true_val) / true_val)
    a1 = AM4(0,5,0,Gaussian,[0,1],narr[i])[0]
    am4arr1.append((a1 - true_val) / true_val)
    a2 = AM4(0,10,0,Gaussian,[0,1],narr[i])[0] - AM4(5,10,0,Gaussian,[0,1],narr[i])[0]
    am4arr2.append((a2 - true_val) / true_val)
    m1 = MonteCarlo(5, 0, Gaussian, [0,1], narr[i])[0]
    mc1arr.append((m1 - true_val) / true_val)

#%%
orderarr = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
plt.plot(orderarr,np.log10(np.abs(simparr1)),label='Simpson - Direct')
plt.plot(orderarr,np.log10(np.abs(am4arr1)),label='AM4 - Direct')
plt.plot(orderarr,np.log10(np.abs(simparr2)),label='Simpson - Difference')
plt.plot(orderarr,np.log10(np.abs(am4arr2)),label='AM4 - Difference')
plt.plot(orderarr,np.log10(np.abs(mc1arr)),label='Monte Carlo')
plt.xlim(1,6)
plt.ylim(-16,0)
plt.xlabel('Order of N')
plt.ylabel('Order of Relative Error')
plt.legend()
plt.savefig('Figure 2',dpi=200)
plt.show()

#%%

'''
Plotting dependence of significance on the choice of mass cuts
'''
# defining all 3 axis
x = np.linspace(115, 125, 100)
y = np.linspace(125.01, 135, 100)
X, Y = np.meshgrid(x, y)
Z = []
for i in range(len(x)):
    z = []
    for j in range(len(x)):
        z.append(Significance(X[i][j],Y[i][j]))
    Z.append(z)
    
#%%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
Z = np.array(Z)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.bone ,linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_zlim(0, 5)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_xlabel('Lower Mass Cut $GeV/c^2$')
ax.set_ylabel('Upper Mass Cut $GeV/c^2$')
#ax.set_zlabel('Significance')
ax.view_init(30,300)

# Add a color bar which maps values to colors.
fig.colorbar(surf,shrink=0.5,aspect=10,pad=0.07,label='Significance')
plt.savefig('Figure 3',dpi=200)
plt.show()
#%%

'''
Finding Optimal Mass Cuts via Minimization 
'''
MCval, MCS_val, MCx, MCy = MC_TA([116,134],1e-2,100,7000,0.99)
GDval, GDx, GDy = GradientDescent([116,134], 1e-2, 10e-10, Significance, 7000)

#%%

#MCval, MCS_val, MCx, MCy = MC_TA([123.23751620206441, 127.15848353344296],1e-9,0.0001,7000,0.99)
#GDval, GDx, GDy = GradientDescent([123.2375147, 127.15848648], 1e-12, 10e-18, Significance, 7000)
#print(MCval,GDval)
actualS = Significance(123.237516, 127.158484)
plus1uS = actualS - Significance(123.237516, 127.158485)
minus1uS = actualS - Significance(123.237516, 127.158483)
plus1lS = actualS - Significance(123.237517, 127.158484)
minus1lS = actualS - Significance(123.237515, 127.158484)
print(plus1uS, minus1uS, plus1lS, minus1lS)


#%%
# defining all 3 axis
print(MCval,GDval)
#10e-2 7000 iterations run --> [123.2373035805146, 127.15846151070012] [123.23750175, 127.15850876]
x = np.linspace(114, 125, 25)
y = np.linspace(125.001, 136, 25)
X, Y = np.meshgrid(x, y)
Z = []
for i in range(len(x)):
    z = []
    for j in range(len(x)):
        z.append(Significance(X[i][j],Y[i][j]))
    Z.append(z)
    
#%%
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z, levels=20, cmap = 'bone')
fig.colorbar(cp, label='Significance') # Add a colorbar to a plot
#ax.set_title('Maximisation Pathways')
ax.set_xlabel('Lower Mass Cut $GeV/c^2$')
ax.set_ylabel('Upper Mass Cut $GeV/c^2$')
plt.plot(GDx[::500],GDy[::500],'o-',label='Gradient Descent')
plt.plot(MCx[::500],MCy[::500],'o--',label='Simulated Annealing')
plt.legend()
plt.savefig('Figure 4',dpi=200)
plt.show()


#%%

'''
Project Part 5
'''
#Nb with optimal Mass Cut
Nb = Integrate(123.237516, 127.158484, MassFunc, [125.1], 10000)
Nh = 470 * (1/1.4) * Integrate(123.237516, 127.158484, Gaussian, [125.1, 1.4], 10000)
print('Nh:',Nh,' Nb:',Nb)

'''
x = np.linspace(5500,6200,100)
y = Gaussian(x, Nb, np.sqrt(Nb))
plt.plot(x,y)
plt.show()
val = (Nh/5)**2
#print(val)
#print(Gaussian(val , Nb, np.sqrt(Nb)))
x = np.linspace(250,550,100)
y = Gaussian(x, Nh, np.sqrt(Nh))
plt.plot(x,y)
plt.show()
'''

'''
# defining all 3 axis
x = np.linspace(5500,6500,100)
y = np.linspace(300,500,100)
X, Y = np.meshgrid(x, y)
Z = []

A = (1 / (np.sqrt(Nb*Nh)))
for i in range(len(x)):
    z = []
    for j in range(len(x)):
        a = Gaussian(Y[i][j], Nh, np.sqrt(Nb))
        b = Gaussian(X[i][j], Nb, np.sqrt(Nb))
        #c = Y[i][j] / np.sqrt(X[i][j])
        z.append(a*b)
    Z.append(z)

fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z, levels=10)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Probability Contour Plot')
ax.set_xlabel('Nb')
ax.set_ylabel('Nh')
plt.show()
'''
'''
x = 10.49140691 #shift needed for 5 sigma 
A1 = Integrate(-1e5, Nb + x, Gaussian, [Nb, np.sqrt(Nb)], 10e-15)/np.sqrt(Nb)
A2 = Integrate(Nh - x, 1e5, Gaussian, [Nh, np.sqrt(Nh)], 10e-15)/np.sqrt(Nb)
print(A1) #probability of 5 sigma 
'''
N = Nb + Nh
x = np.linspace(5750,6750,100)
sigma5val = 5*np.sqrt(Nb) + Nb 
print(sigma5val)
sigval = np.sqrt(N)
y = (1/sigval)*Gaussian(x, N, sigval)

#%%

plt.axvline(x=sigma5val,color='red',linestyle='--',label='5-sigma value')
plt.xlim(5950,6550)
plt.ylim(-0.0001,0.0055)
plt.fill_between(x, y, where=x > sigma5val, color='grey', alpha=0.5, label='5-sigma region')
plt.xlabel('Number of Photon Pairs N')
plt.ylabel('Probability')
plt.text(6150, 0.0005, 'N = 6245', color='red', fontsize=15)
plt.text(6270,0.002, '55.45%', fontsize=15)
plt.plot(x,y,label='N Gaussian')
plt.legend()
plt.savefig('Figure 5',dpi=200)
plt.show()
#0.5544748735472437

#%%
print('Expected 5-sigma probability:',Integrate(sigma5val, 10e5, Gaussian, [N,sigval], 10e-10)/sigval)

#%%

'''
Project Part 6
'''
#1. Higgs mass is known to +- 0.2GeV/c^2

sig1 = np.sqrt(5862.08073547972)
mass_arr = np.linspace(125.1-0.2,125.1+0.2,100)
Nh_arr1 = []
for i in range(len(mass_arr)):
    b = 470 * (1/1.4) * Integrate(123.237516, 127.158484, Gaussian, [mass_arr[i], 1.4], 1000)
    Nh_arr1.append(b)
plt.plot(mass_arr, Nh_arr1)
plt.title('Nh dependance on mass fluctuations')
plt.axhline(y = 5*sig1, color = 'r', linestyle = '-') 
plt.show()

err1a = ((np.max(Nh_arr1) - np.mean(Nh_arr1))*100 / (393.6549536649624))
err1b = ((np.mean(Nh_arr1) - np.min(Nh_arr1))*100 / (393.6549536649624))

#%%

'''
x = np.linspace(120,130,100)
y_b = MassFunc(x, 125.1)
y_h = 470 * (1/1.4) * Gaussian(x, 125.1, 1.4)
y_hl = 470 * (1/1.4) * Gaussian(x, 125.1-0.2, 1.4)
y_hh = 470 * (1/1.4) * Gaussian(x, 125.1+0.2, 1.4)
y = y_b + y_h
yl = y_b + y_hl
yh = y_b + y_hh
plt.xlim(120,130)
plt.ylim(1200,1900)
plt.plot(x,y_b,':', label='Nb')
plt.plot(x,yl, '--', label='lower Nh + Nb')
plt.plot(x,yh, '--', label='upper Nh + Nb')
plt.plot(x,y, label='Nh + Nb')
plt.axvline(x=123.237516,color='red',linestyle='--',label='optimal lower mass cut')
plt.axvline(x=127.158484,color='purple',linestyle='--',label='optimal upper mass cut')
plt.legend()
plt.show()
'''

#2. Photon interacting with detector material and lose energy
aff_arr = np.linspace(0,0.04,100) #% of photons affected
Nh_arr2 = []
for i in range(len(aff_arr)):
    a = 470 * (1/2.6) * Integrate(123.237516, 127.158484, Gaussian, [124.5, 2.6], 1000)
    b = 470 * (1/1.4) * Integrate(123.237516, 127.158484, Gaussian, [125.1, 1.4], 1000)
    Nh_arr2.append(aff_arr[i]*a + (1-aff_arr[i])*b)
err2 = (np.max(Nh_arr2) - np.mean(Nh_arr2))*100 / 393.6549536649624
plt.plot(aff_arr, Nh_arr2)
plt.title('Nh dependance on Photon interaction fraction')
plt.show()

#3. Number of Higgs boson +-3% error
totalpluserr = np.sqrt(err1a**2 + 9)
totalminuserr = np.sqrt(err1b**2 + err2**2 + 9)
print('error1: + ',err1a,'% and - ',err1b,'%',' error2: - ',err2,'%',' error3: +- 3%')
print('total error + ',totalpluserr,'% - ',totalminuserr, '%')

#%%

N = Nb + Nh
x = np.linspace(5750,6750,100)
sigma5val = 5*np.sqrt(Nb) + Nb 
#print(sigma5val)
sigval = np.sqrt(N)
y = (1/sigval)*Gaussian(x, N, sigval)
N1 = Nb + 394.13733645093185
sigval1 = sigval + Nh*0.03
y1 = (1/sigval1)*Gaussian(x,N1, sigval1)
N2 = Nb + 389.6921798681271*0.96 +  387.9308592756789*0.04
sigval2 = sigval + Nh*(0.03 + 0.00727) 
y2 = (1/sigval2)*Gaussian(x,N2, sigval2)

#%%
'''
plt.axvline(x=sigma5val,color='red',linestyle='--',label='5-sigma value')
plt.xlim(6200,6500)
plt.ylim(0,0.0055)
plt.fill_between(x, y1, where=x > sigma5val, color='green', alpha=0.3, label='max 5-sigma region')
plt.fill_between(x, y2, where=x > sigma5val, color='red', alpha=0.3, label='min 5-sigma region')
plt.xlabel('Number of Photon Pairs N')
plt.ylabel('Probability')
#plt.text(6150, 0.0005, 'N = 6245', color='red', fontsize=15)
#plt.text(6270,0.002, '55.45%', fontsize=15)
plt.plot(x,y,label='original Gaussian')
plt.plot(x,y1, label='upper Gaussian', color='green')
plt.plot(x,y2, label='lower Gaussian', color='red')
plt.legend()
plt.savefig('Figure 6',dpi=200)
plt.show()
'''
#%%
err = 0.5*(totalpluserr + totalminuserr)/100
sigval3 = np.sqrt(N)+err*Nh 
p3 = Integrate(sigma5val, 10e5, Gaussian, [N,sigval3], 100000)/sigval3
print('Validation Probability = ',p3,' +- ',err*p3)

#%%
'''
p1 = Integrate(sigma5val, 10e5, Gaussian, [N1,sigval1], 100000)/sigval1
p2 = Integrate(sigma5val, 10e5, Gaussian, [N2,sigval2], 100000)/sigval2
print('Observed 5-sigma probability =', (p1+p2)/2)
'''
#%%

#Plotting 3D fit of errors
No=100
x = np.linspace(0,4,No)
y = np.linspace(124.9,125.3,No)
X, Y = np.meshgrid(x, y)
Z = []
for i in range(No):
    z = []
    for j in range(No):
        frac = X[i][j]/No
        val = Y[i][j]
        #print(frac,val)
        a = 470 * (1/2.6) * Integrate(123.237516, 127.158484, Gaussian, [124.5, 2.6], 10000)
        b = 470 * (1/1.4) * Integrate(123.237516, 127.158484, Gaussian, [val, 1.4], 10000)
        #print(a,b)
        N = frac * a + (1 - frac) * b + Nb
        #print(N)
        sigval = np.sqrt(N) + Nh*0.03
        x = Integrate(sigma5val, 10e5, Gaussian, [N,sigval], 10000)/sigval
        #print(x)
        z.append(x)
        #z.append(Gaussian(x, N, sigval)/sigval)
    Z.append(z)

#%%

print('Observed probability is:',np.mean(Z))
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z, levels=25, cmap = 'bone')
fig.colorbar(cp, label='Probability') # Add a colorbar to a plot
ax.set_xlabel('% photon pair degraded')
ax.set_ylabel('Higgs mass $GeV/c^2$')
plt.legend(['Average Probability = 53.09'])
plt.savefig('Figure 7',dpi=200)
plt.show()

#Expected 5-sigma probability upper: 0.5495350724779424
#Expected 5-sigma probability lower: 0.5289092559498556
#Observed 5-sigma probability = 0.539222164213899


