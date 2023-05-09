import numpy as np
import scipy as sci
import time 
from  matplotlib import pyplot as plt

tic = time.time()
size_of_sweep = 60
#Data parameters
data = np.load("xyz_coordinates.npy")
data = np.transpose(data)
p      = 0.88      #training/validation ratio
X = data[0:int(len(data)*p),:] #training data
Y = data[int(len(data)*p):-1,:] #Validation data

lyaponov_times=np.arange(len(Y) )*(10)/(len(Y)*1.1037) #x-axis in lyaponov times

#Hyperparameters to play around with:
neurons                   = 500    #500 good
M                         = 3           #no. of coordinates
reservoir_sparsity        = 1      #0.9 good
ridge_parameter           = 1e-5  #0.0005 gives good results
###################################################################
singular_value_history = np.zeros(size_of_sweep)
error_history = np.zeros(size_of_sweep)
def generateWeights(inputs,neuron_number):
    
    w_res = sci.sparse.random(neuron_number,neuron_number,density=reservoir_sparsity)*reservoir_weight_variance
    w_res = w_res.toarray()
    w_in  = np.random.randn(neuron_number,inputs)*input_weight_variance
    return w_res,w_in

for s in range(1,size_of_sweep):
    for k in range(5):
        reservoir_weight_variance = s/(2000)
        input_weight_variance     = 1/(neurons)
        
        #The idea is to compute a new set of neuron states in the
        # reservoir for each timestep and store these states
        # in a matrix to then perform ridge regression on.
            
        #initialize current neuron states in reservoir, and weight matrices:
        W_res,W_input         = generateWeights(M,neurons)
        states                = np.zeros((1,neurons))
        states_over_time      = np.zeros((len(X),neurons))
        
        #add the initial states to the storage:
        states_over_time[0,:] = states
        
        error = np.zeros(len(Y))
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
        W_out = np.dot(term1,term2)
        
        
        
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
        
        
        y = predicted_coordinates[:,1]  
        
        #Divide by 5 to get mean error
        error = error+ (Y[:,1]-y)/5
        error = abs(error)
        
    U, singular_values, V = np.linalg.svd(W_res)    
    singular_value_history[s] = singular_values[0]
    error_history[s] = np.mean(error[0:2000])
    #1.1037 is theoretical lyaponov time
   # print("Maximum singular value = " + str(singular_values[0]))
  #  print("Reservoir variance = " + str(round(reservoir_weight_variance,5)))
    print("Mean error: " + str(round(np.mean(error[0:2000]),5)))
   # print("Error variance: " + str(np.round(np.var(error[0:2000]),5)))
    print("-----------------------------")
    
fig = plt.figure()
# plt.semilogy(lyaponov_times[0:2000], error[0:2000] ,label  = "Max Sing. Value " + str(round(singular_matrix[0],3)))
plt.loglog(singular_value_history[1:],error_history[1:])
plt.legend()
plt.grid()
plt.xlabel(r"Maximum singular value")
plt.ylabel(r"$\delta$")
plt.title(r"Prediction error $\delta$")
toc = time.time()

print("Elapsed time: " + str(round(toc-tic)) + " seconds.")

