import numpy as np
import reservoirpy
import scipy as sci
from  matplotlib import pyplot as plt
from reservoirpy.datasets import lorenz
from reservoirpy.nodes import Reservoir, Ridge, Input
data = np.load("xyz_coordinates.npy")
data = np.transpose(data)

#Parameters:
neurons=205
M = 3
p = 0.95

X = data[0:int(len(data)*p),:] #training data
Y = data[int(len(data)*p):-1,:] #Validation data


def generateWeights():
    w_res = np.random.randn(neurons,neurons)*(2/neurons)
    w_in  = np.random.randn  (neurons,M)*(1/neurons)
    return w_res,w_in


W_res,W_input = generateWeights()

#Reservoir neuron states (r)
states              = np.zeros((1,neurons))
states_over_time    = np.zeros((len(X),neurons))
states_over_time[0,:] = states

#append new states with: states_over_time[t,:] = states


for t in range(len(X)-1):
    
    #compute new states according to (1a):
    recurrent_term = np.squeeze(np.dot(W_res,np.transpose(states)))
    current_term = np.dot(W_input,X[t,:])
    new_states = np.tanh(np.transpose(np.add(recurrent_term,current_term)))
    
    #add the new states to our storage
    states_over_time[t+1,:] = new_states
    
    #update so that the new states can be used in the next iteration
    states = new_states

#Apply ridge regression to train the output matrix
kI = np.eye(neurons)*0.005
R = states_over_time
term1 = np.linalg.inv((np.dot(np.transpose(R),R)+kI))
term2 = np.dot(np.transpose(R),X)
W_out =  np.dot(term1,term2)

#predict the future:
O = np.dot(states,W_out)
predicted_coordinates = np.zeros((len(Y),3))

for i in range(len(predicted_coordinates)):
    O = np.dot(states,W_out)
    recurrent_term = np.squeeze(np.dot(W_res,np.transpose(states)))
    current_term = np.dot(W_input,O[:])
    new_states = np.tanh(np.transpose(np.add(recurrent_term,current_term)))
    states = new_states;
    predicted_coordinates[i,:] = O


x = predicted_coordinates[:,0]
y = predicted_coordinates[:,1]
z = predicted_coordinates[:,2]

fig2 = plt.figure()


plt.plot(np.arange(len(Y)),Y[:,1],label = "Ground Truth")
plt.plot(np.arange(len(Y)),y,label = "Predicted")
plt.legend()
#fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d')
#plt.plot(x,y,z,linewidth = 0.6)



