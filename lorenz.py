import numpy as np
from matplotlib import pyplot as plt
from scipy import linspace
from scipy.integrate import solve_ivp
#This script will generate coordinates over time
#for the Lorenz attractor

s = 10  #sigma
r = 28  #Rho
b = 8/3 #beta

duration       = 200 #how long to simulate for.
resolution     = 50000
start_position = [1,1,1]

t = np.linspace(0,duration,resolution) #time 


def lorenz_system(t,xyz):
    x,y,z = xyz
    return [ s*(y - x),
             r*x - x*z - y, 
             x*y - b*z ]

trajectory = solve_ivp(lorenz_system, [0,duration],start_position,t_eval=t)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
plt.plot(trajectory.y[0],trajectory.y[1],trajectory.y[2],linewidth = 0.6)
np.save("xyz_coordinates",(trajectory.y[0],
                           trajectory.y[1],
                           trajectory.y[2]))



    
