import time
import numpy as np
import scipy as sci

from  matplotlib import pyplot as plt
from sklearn import preprocessing
tic =  time.time()
#Data parameters
data = np.load("xyz_coordinates.npy")
data = np.transpose(data)
p      = 0.88      #training/validation ratio
X = data[0:int(len(data)*p),:] #training data
Y = data[int(len(data)*p):-1,:] #Validation data

lyaponov_times=np.arange(len(Y) )*(10)/(len(Y)*1.1037)
fig2 = plt.figure()
plt.grid()
plt.xlabel(r"Lyaponov time $\lambda _1 t$")
plt.ylabel(r"y")
plt.title("Reservoir trained on y")
size_of_sweep = 60
singular_value_history = np.zeros(size_of_sweep)
error_history = np.zeros(size_of_sweep)
def generateWeights(inputs,neuron_number):
 
   # w_res = sci.sparse.random(neuron_number,neuron_number,density=reservoir_sparsity)*reservoir_weight_variance
   # w_res = w_res.toarray()
    
    w_res = np.random.randn(neuron_number,neuron_number)*reservoir_weight_variance
    #w_res = w_res.toarray()
    w_in  = np.random.randn(neuron_number,inputs)*input_weight_variance
    return w_res,w_in

###################################################
#Train reservoir only on y:
####################################################


for s in range(1,size_of_sweep):
    for k in range(10):
        #Hyperparameters to play around with:
        neurons                   = 500   #500 good
        reservoir_sparsity        = 1  #0.9 good
        reservoir_weight_variance = s/(1000)
        input_weight_variance     = 1.1
        ridge_parameter           = 2e-5  #0.0005 gives good results

#initialize current neuron states in reservoir, and weight matrices:

        x_2 = X[:,1]

        M                     = 1
        W_res,W_input         = generateWeights(M,neurons)
        states                = np.zeros((neurons,1))
        states_over_time      = np.zeros((len(x_2),neurons))


#add the initial states to the storage:
        
        states_over_time[0,:] = states[:,0]
        error = np.zeros(len(Y))
        for t in range(len(x_2)-1):
                
                #compute new states according to equation (1a):
                recurrent_term = np.dot(W_res,states)
                current_term   = np.dot(W_input,x_2[t])
            
                new_states     = np.tanh(recurrent_term+current_term)
                
                #add the new states to our storage
                states_over_time[t+1,:] = new_states[:,0]
                
                #update so that the new states can be used in the next iteration
                states = new_states
    
        #Apply ridge regression using our storage matrix to train the output matrix W_out.
        kI    = np.eye(neurons)*ridge_parameter 
        R     = states_over_time
            
            
            
        term1 = np.linalg.inv((np.dot(np.transpose(R),R))  +kI )  #(X'X + λI)^-1
        term2 = np.dot(np.transpose(R),x_2)    #X'y 
                
        W_out =  np.expand_dims(np.dot(term1,term2),1)
            
        predicted_coordinates = np.zeros((len(Y)))
        output = np.squeeze(np.dot(np.transpose(W_out),states))
            
        for i in range(len(predicted_coordinates)):
                output = np.squeeze(np.dot(np.transpose(W_out),states))
                recurrent_term = np.dot(W_res,states)
                current_term   = W_input*output
                new_states     = np.tanh(recurrent_term+current_term)  
                states         = new_states;
                predicted_coordinates[i] = output
    

        
        y = predicted_coordinates
         #Divide by 5 to get mean error
       
        error = error+ (Y[:,1]-y)/10
        error = abs(error)
        
    U, singular_values, V = np.linalg.svd(W_res)  
    singular_value_history[s] = singular_values[0]
    error_history[s] = np.mean(error[0:1000])
    #1.1037 is theoretical lyaponov time
    print("Mean error: " + str(round(np.mean(error[0:2000]),5)))
    #print("Maximum singular value = " + str(singular_values[0]))
    #print("Reservoir variance = " + str(round(reservoir_weight_variance,5)))
    print("-----------------------------")
    
plt.loglog(singular_value_history[1:],error_history[1:])

#lyaponov coefficent = 0.906
#Lyaponov time= 0.906**-1 = 1.1037

#plt.plot(lyaponov_times, y,label  = "predicted")
#plt.plot(lyaponov_times,Y[:,1],'--',label = "Validation data")

plt.show
plt.legend()
plt.xlabel(r"Maximum singular value")
plt.ylabel(r"$\delta$")
plt.title(r"Prediction error $\delta$")
toc = time.time()
print("Elapsed time: " + str(round(toc-tic)) + " seconds.")