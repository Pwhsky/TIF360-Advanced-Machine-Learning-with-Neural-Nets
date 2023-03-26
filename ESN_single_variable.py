
import numpy as np
import scipy as sci

from  matplotlib import pyplot as plt
from sklearn import preprocessing

#Data parameters
data = np.load("xyz_coordinates.npy")
data = np.transpose(data)
p      = 0.9       #training/validation ratio
X = data[0:int(len(data)*p),:] #training data
Y = data[int(len(data)*p):-1,:] #Validation data

  
fig2 = plt.figure()
plt.grid()
plt.xlabel(r"Lyaponov time $\lambda _1 t$")
plt.title("Reservoir trained on y")
def generateWeights(inputs,neuron_number):
 
    w_res = sci.sparse.random(neuron_number,neuron_number,density=reservoir_sparsity)*reservoir_weight_variance
    w_res = w_res.toarray()
    w_in  = np.random.randn(neuron_number,inputs)*input_weight_variance
    return w_res,w_in

###################################################
#Train reservoir only on y:
####################################################
    

#Hyperparameters to play around with:
neurons                   = 500    #500 good
reservoir_sparsity        = 1  #0.9 good
reservoir_weight_variance = 2/neurons 
input_weight_variance     = 4.1
ridge_parameter           = 1e-1  #0.0005 gives good results

#initialize current neuron states in reservoir, and weight matrices:
M                     = 1
W_res,W_input         = generateWeights(M,neurons)
states                = np.zeros((1,neurons))
states_over_time      = np.zeros((len(X),neurons))

x_2 = X[:,1]



    #add the initial states to the storage:
states_over_time[0,:] = states



    #The main simulation loop, getting dimensions right is a hassle but it works now.
for t in range(len(x_2)-1):
        
        #compute new states according to equation (1a):
        recurrent_term = np.dot(W_res,np.transpose(states))
        current_term   = W_input*x_2[t]
        

        new_states     = np.tanh(np.transpose(np.add(recurrent_term,current_term)))
        
        #add the new states to our storage
        states_over_time[t+1,:] = new_states[:,0]
        
        #update so that the new states can be used in the next iteration
        states = new_states

#Apply ridge regression using our storage matrix to train the output matrix W_out.
kI    = np.eye(neurons)*ridge_parameter 
R     = states_over_time



term1 = np.linalg.inv((np.dot(np.transpose(R),R)+kI))    #(X'X + λI)^-1
term2 = np.expand_dims(np.dot(np.transpose(R),x_2),1)                      #X'y 

W_out =  np.dot(term1,term2)

        
predicted_coordinates = np.zeros((len(Y)))
output = np.dot(states,W_out)

for i in range(len(predicted_coordinates)):
        output = np.dot(states,W_out)
        recurrent_term = np.dot(W_res,np.transpose(states))
        current_term   = W_input*output
        new_states     = np.tanh(np.transpose(np.add(recurrent_term,current_term)))
        states         = new_states;
        predicted_coordinates[i] = output



y = predicted_coordinates

#lyaponov coefficent = 0.906
#Lyaponov time= 0.906**-1 = 1.1037

plt.plot(np.arange((len(Y)))*(10)/(len(Y)*1.1037), y,label  = "predicted")
plt.legend()


plt.show
plt.plot(np.arange(len(Y))*(10)/(len(Y)*1.1037),Y[:,1],'--',label = "Validation data")

    

