import numpy as np
import scipy as sci
import time 
from  matplotlib import pyplot as plt

tic = time.time()

#Data parameters
data = np.load("xyz_coordinates.npy")
data = np.transpose(data)
p      = 0.88      #training/validation ratio 
X = data[0:int(len(data)*p),:] #training data
Y = data[int(len(data)*p):-1,:] #Validation data


#Hyperparameters to play around with:
neurons                   = 600    #500 good
M                         = 3           #no. of coordinates
reservoir_sparsity        = 1       #0.9 good
reservoir_weight_variance = 0.00417  #Derived from parameter sweep, 2/neurons is also good
input_weight_variance     = 1/(neurons)
ridge_parameter           = 1e-5   #0.0005 gives good results
###################################################################


#Initialize the weights as gaussian distributed numbers.


def generateWeights(inputs,neuron_number):
    
    w_res = sci.sparse.random(neuron_number,neuron_number,density=reservoir_sparsity)*reservoir_weight_variance
    w_res = w_res.toarray()
    w_in  = np.random.randn(neuron_number,inputs)*input_weight_variance
    return w_res,w_in



#The idea is to compute a new set of neuron states in the
# reservoir for each timestep and store these states
# in a matrix to then perform ridge regression on.
    
#initialize current neuron states in reservoir, and weight matrices:
W_res,W_input         = generateWeights(M,neurons)
states                = np.zeros((1,neurons))
states_over_time      = np.zeros((len(X),neurons))

#add the initial states to the storage:
states_over_time[0,:] = states


#The main simulation loop, getting dimensions right is a hassle but it works now.
for t in range(len(X)-1):
    
    #compute new states according to equation (1a):
    recurrent_term = np.squeeze(np.dot(W_res,np.transpose(states)))
    current_term   = np.dot(W_input,X[t,:])
    new_states     = np.tanh(np.transpose(np.add(recurrent_term,current_term)))
    
    #add the new states to our storage
    states_over_time[t+1,:] = new_states
    
    #update so that the new states can be used in the next iteration
    states = new_states

#Apply ridge regression using our storage matrix to train the output matrix W_out.
kI    = np.eye(neurons)*ridge_parameter 
R     = states_over_time



term1 = np.linalg.inv((np.dot(np.transpose(R),R)+kI))  #(X'X + λI)^-1
term2 = np.dot(np.transpose(R),X)                      #X'y 

W_out =  np.dot(term1,term2)



#predict the future of the time series, and compare to validation data Y
    
predicted_coordinates = np.zeros((len(Y),M))
output = np.dot(states,W_out)
for i in range(len(predicted_coordinates)):
    output = np.dot(states,W_out)
    recurrent_term = np.squeeze(np.dot(W_res,np.transpose(states)))
    current_term   = np.dot(W_input,output[:])
    new_states     = np.tanh(np.transpose(np.add(recurrent_term,current_term)))
    states         = new_states;
    predicted_coordinates[i,:] = output




x = predicted_coordinates[:,0]
y = predicted_coordinates[:,1]
z = predicted_coordinates[:,2]

fig2 = plt.figure()

error = abs(Y[:,1]-y)


#1.1037 is theoretical lyaponov time for the lorenz system,
# adjust x-axis such that it displays Lyaponov times
lyaponov_times = np.arange(len(Y) )*(10)/(len(Y)*1.1037)

plt.plot(lyaponov_times, Y[:,1],'--',label = "Validation data")
plt.plot(lyaponov_times, y          ,label = "Predicted data")
plt.plot(lyaponov_times, error      ,label = "Error")
plt.legend()
plt.grid()
plt.xlabel(r"Lyaponov time $\lambda _1 t$")
plt.title("Reservoir trained on xyz")

toc = time.time()

print("Elapsed time: " + str(round(toc-tic)) + " seconds.")


