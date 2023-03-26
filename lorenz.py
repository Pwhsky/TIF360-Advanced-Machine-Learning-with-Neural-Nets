import numpy as np

from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
#This script will generate coordinates over time
#for the Lorenz attractor

s = 10  #sigma
r = 28  #Rho
b = 8/3 #beta


duration       = 40        #how long to simulate for. #40 for producing test data
res     = 40000     #100000 for training data or 400000
start_position1 = [-0.8,-0.2,0.5]
separation = 1e-9
cutoff = int(0.225*res)

start_position2 = [-0.8,-0.2,0.5+separation]

t = np.linspace(0,duration,res) #time 


def lorenz_system(t,xyz):
    x,y,z = xyz
    return [ s*(y - x),
             r*x - x*z - y, 
             x*y - b*z ]

#Generate a trajectory for the lorenz attractor:
trajectory1 = solve_ivp(lorenz_system, [0,duration],start_position1,t_eval=t)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
plt.plot(trajectory1.y[0],trajectory1.y[1],trajectory1.y[2],linewidth = 0.6)

x = trajectory1.y[0]
y = trajectory1.y[1]
z = trajectory1.y[2]
np.save("xyz_coordinates",(x[cutoff:],
                           y[cutoff:],
                           z[cutoff:]))

#To solve for lyaponov time, calculate a second trajectory 
#slightly offset and compute euclidian distance

#Compute a second trajectory slightly offset:
trajectory2 = solve_ivp(lorenz_system, [0,duration],start_position2,t_eval=t)

#compute euclidean distance between the trajectories
distance = np.sqrt((np.abs(trajectory1.y[0]-trajectory2.y[0]))**2 +
                   (np.abs(trajectory1.y[1]-trajectory2.y[1]))**2 +
                   (np.abs(trajectory1.y[2]-trajectory2.y[2]))**2 )
fig2 = plt.figure()

coefficent = (np.polyfit(t[cutoff:],distance[cutoff:],0))

plt.semilogy(t[cutoff:],distance[cutoff:],label="Distance between trajectories")
plt.semilogy(t[cutoff:],separation*np.exp(coefficent*t[cutoff:]), '--',label="linear fit")

plt.title("separation of trajectories for δ = 1e-9")
print("computed lyaponov exponent = "+ str(coefficent))
print("1 Lyaponov time = " + str(coefficent**-1))


    