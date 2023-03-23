import numpy as np
from matplotlib import pyplot as plt
#This script will generate coordinates over time
#for the Lorenz attractor
s = 10  #sigma
r = 28  #Rho
b = 8/3 #beta

time_steps = 100000 #simulation length
dt = 0.008         #resolution of trajectory

#Initialize space
x = np.zeros((time_steps))
y = np.zeros((time_steps))
z = np.zeros((time_steps))

#initial values must be non-zero
x[0] = 1
y[0] = 1
z[0] = -2

#Run the dynamics:
for t in range(time_steps-1):
    x[t+1] = x[t] + ( s*(y[t] - x[t])       )*dt
    y[t+1] = y[t] + ( x[t]*(r - z[t]) - y[t])*dt
    z[t+1] = z[t] + ( x[t]*y[t] - b*z[t]   )*dt



fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
plt.plot(x,y,z,linewidth = 1)
plt.savefig("lorenz.pdf")
np.save("xyz_coordinates",(x,y,z))
