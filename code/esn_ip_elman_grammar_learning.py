
"""
Created on Mon May 23 13:26:46 2016
@author: Remy Portelas
https://github.com/rPortelas/ip_in_esn
---
Based on the minimalistic Echo State Networks by Mantas Lukosevicius 2012
enhanced by  Xavier Hinaut.
in "plain" scientific Python.
"""
import numpy as np
from matplotlib.pyplot import *
import scipy.linalg
import cPickle as pickle

#take an array of target words and 
#an array of prediction arrays to plot results
def plot_predictions(target_words, predictions):
    x = np.arange(class_nb)
    my_xticks = vocabulary
    
    for i in range(predictions.shape[0]):
        prediction = predictions[i,:]
        figure(i).clear()
        title('Prediction for ' + target_words[i])
        xticks(x,my_xticks, rotation = 45) 
        
        if scale_mode == 'scale entries' :
            #unshift data by +0.5 for ease of visualisation
            prediction += 0.5
        plot(x, prediction)

#display the spatio temporal distribution of 
#reservoir's neurons during 1000 timesteps
def plot_activity(x,epoch):
    figure(epoch+42).clear()
    hist(x.ravel(), bins = 200)
    xlim(-1,+1)
    xlabel('neurons outputs')       
    ylabel('number of timesteps')   
    #compute some caracteristics of the distribution
    mean = str(round(x.mean(), 2))
    med = str(round(np.median(x), 2))
    min = str(round(x.min(), 2))
    max = str(round(x.max(), 2))
    std_dev = str(round(x.std(), 2))
    title('Spatio-temporal distribution of the reservoir within 1000 timesteps'
    ' at epoch ' + str(epoch) + '\n mean = '+ mean + ' median = ' + med + \
    ' min = ' + min + ' max = ' + max + ' std_dev = ' + std_dev)

#display the activity of a neuron during 1000 timesteps
def plot_neuron_activity(x,epoch):
    figure(epoch+84).clear()
    hist(x.ravel(), bins = 200)
    xlim(-1,+1)
    xlabel('neuron outputs')       
    ylabel('number of timesteps')
    #compute some caracteristics of the distribution
    title('The output distribution of a single randomly chosen neuron' \
    'at epoch ' + str(epoch))
    
def set_seed(seed=None):
    """Making the seed (for random values) variable if None"""
    
    # Set the seed
    if seed is None:
        import time as t
        seed = int((t.time()*10**6) % 10**12)
        
    try:
        np.random.seed(seed) #np.random.seed(seed)
        print "Seed used for random values:", seed
    except:
        print "!!! WARNING !!!: Seed was not set correctly."
    return seed

#binary matrix refer to an array of 0 and 1
#turn the list of x integer into a x*class_nb matrix
#the idea: 3 becomes [0,0,0,1,0,...] (class_nb size)
def int_list_to_shifted_binary_matrix(int_list, class_nb):
    res = np.zeros((int_list.size,class_nb))  
    for i in range(int_list.size):
        res[i,int_list[i]] = 1    
        
    if scale_mode == 'scale entries' :
        #input and output are shifted by -0.5 
        #to scale well with the tanh activation function
        return res -0.5
    elif scale_mode == 'no scaling' :
        return res
    else :
        raise Exception, "ERROR: 'scale_mode' was not set correctly."

#given a vector of 0 and 1 (if no scaling) or -0.5 and 0.5 (if scaling) ,
#return the corresponding word according to the vocabulary      
def compute_word(bin_vec):
    if scale_mode == 'scale entries' :
        return vocabulary[np.where(bin_vec == 0.5)[0][0]]
    elif scale_mode == 'no scaling' :
        return vocabulary[bin_vec.nonzero()[0][0]]
    else :
        raise Exception, "ERROR: 'scale_mode' was not set correctly."
        
#given a network output vector,
#return 'singular' is the mean of all singular verbs probability 
#is higher than plural ones or return 'plural' if not.    
def compute_class(out):
    mean_singular = 0
    mean_plural = 0
    for v in singular_verbs:
        mean_singular += out[vocabulary.index(v)]
    mean_singular = mean_singular / singular_verbs.__len__()
    for v in plural_verbs:
        mean_plural += out[vocabulary.index(v)]
    mean_plural = mean_plural / plural_verbs.__len__()
    if mean_singular > mean_plural :
        return 'singular'
    return 'plural'

ip_mode = 'intrinsic plasticity on'
#ip_mode = 'intrinsic plasticity off'

# if "scale entries" is active, the input will be shifted by -0.5 to scale 
# with reservoir's tanh activation function
#scale_mode = 'scale entries'
scale_mode = 'no scaling'

 
#don't change this seed to generate the same dataset as in the report
data_shuffle_seed = 989898  
set_seed(data_shuffle_seed) 
  
#load the data
with open('../datasets/t5_train') as f:
    text_train =(' '.join(pickle.load(f))).split(' . ')
    np.random.shuffle(text_train)
    text_train = (' . '.join(text_train)).split(' ')
    
with open('../datasets/t5_test') as f:
    text_test =(' '.join(pickle.load(f))).split(' . ')
    np.random.shuffle(text_test)
    text_test = (' . '.join(text_test)).split(' ')
#the shuffle extract the last'.'
text_train.append('.') 

#display a samble of the testing set
print text_test[0:10]

#init the list of possible words 
vocabulary = ['boy', 'girl', 'cat', 'dog', 'boys', 'girls', 'cats',
              'dogs', 'John', 'Mary', 'hits', 'feeds', 'sees',
              'hears', 'walks', 'lives', 'hit', 'feed', 'see',
              'hear', 'walk', 'live', 'who', '.']
              
#instanciate classes of verbs
singular_verbs = ('sees','hears','hits','walks','lives','feeds')
plural_verbs = ('see','hear','hit','walk','live','feed')
verbs = singular_verbs + plural_verbs

#change words by they proper index
u_data_train = np.asarray([vocabulary.index(w) for w in text_train[:-1]])
y_data_train = np.asarray([vocabulary.index(w) for w in text_train[1:]])
u_data_test = np.asarray([vocabulary.index(w) for w in text_test[:-1]])
y_data_test = np.asarray([vocabulary.index(w) for w in text_test])

#create matrices to run the multi-classification network
class_nb = vocabulary.__len__()
u_train = int_list_to_shifted_binary_matrix(u_data_train,class_nb)
y_train = int_list_to_shifted_binary_matrix(y_data_train,class_nb)

u_test = int_list_to_shifted_binary_matrix(u_data_test,class_nb)
y_test = int_list_to_shifted_binary_matrix(y_data_test,class_nb)

#training and testing parameters
nb_network = 5 #nb_network will be trained and the performances 
               #will be averaged across them
nb_epoch = 2
trainLen = u_train.shape[0]
testLen = u_test.shape[0]

# generate the ESN reservoir
inSize = outSize = class_nb #input/output dimension
resSize = 50 #reservoir size
a = 1 # leaking rate, (a = 1) <==> no leaky neurons
spectral_radius = 0.98
input_scaling = 1.2
reg = 0.02 # regularization coefficient

if (ip_mode == 'intrinsic plasticity on'):
     #initialisation of IP parameters
     lr = 0.001 #IP learning rate
     m = 0.      #will store the mean of the reservoir activations
     sigma = 0.2 #standard deviation 0.2 gives best results
     var = np.square(sigma) #variance   
     #instanciate some matrix to store the evolution of IP's gain and bias
     ip_gain = np.ones((resSize, 1))
     record_ip_gain = np.zeros((5000 , 1))
     record_ip_bias = np.zeros((5000 , 1))
     ip_bias = np.zeros((resSize, 1))
    
#instanciate perfomances metrics
mse = 0
agreement_rate = 0
averaged_recorded_prediction = np.zeros((6,class_nb))
for network in range(nb_network) :
    #create a network with random weigths
    our_seed = None #42 * network 
    set_seed(our_seed) 
    Win = (np.random.rand(resSize,1+inSize)-0.5) * input_scaling
    W = np.random.rand(resSize,resSize)-0.5 
    # Option 1 - direct scaling (quick&dirty, reservoir-specific):
    #W *= 0.135 
    # Option 2 - normalizing and setting spectral radius (correct, slow):
    print 'Computing spectral radius...',
    rhoW = max(abs(np.linalg.eig(W)[0]))
    print 'done.'
    W *= spectral_radius / rhoW
    
    
    #to display the spatio-temporal activity of 1000 timesteps
    recorded_res_out = np.zeros((1000, resSize))

    #choose a random neuron in the res 
    #to create a histogram of its activations
    choosen_neuron = np.random.random_integers(resSize - 1)
    neuron_out_records = np.zeros(1000)
    
    #IP init
    ip_gain = 1
    ip_bias = 0
    
    # allocated memory for the design (collected states) matrix
    X = np.zeros((1+inSize+resSize,trainLen))
    
    # set the corresponding target matrix directly
    Yt = y_train.T
    
    # run the reservoir with the data and collect X
    x = np.zeros((resSize,1))
    for epoch in range(nb_epoch):
        for t in range(trainLen):
            #load the new input
            u = np.reshape(u_train[t],(u_train[t].shape[0],1))
            res_in = np.dot( Win, np.vstack((1, u))) + np.dot( W, x )
            #compute reservoir activation
            if ip_mode == 'intrinsic plasticity on':
                res_out = np.tanh( ip_gain * res_in + ip_bias )
                if (epoch !=0) & (t<5000) :
                    d_ip_bias = (-lr) * ((-(m / var)) + (res_out / var ) * \
                    ((2 * var) + 1 - np.square(res_out) + m * res_out))
                    #compute delta_bias and update IP's gain and bias
                    ip_bias += d_ip_bias
                    ip_gain += (lr / ip_gain) + (d_ip_bias * res_in)
                    #store the results to plot them
                    record_ip_bias[t,0] += ip_bias.mean()
                    record_ip_gain[t,0] += ip_gain.mean()
            else :
                res_out = np.tanh(res_in)
                
            x = (1-a) * x + a * res_out
            #the first epoch initialise the network,
            #we start storing results and training IP during the second epoch
            if(trainLen-t) <= 1000 :
                #store spatio-temporal activity of the reservoir
                recorded_res_out[trainLen-t-1] = res_out[:,0]  
                #accumulate values of a randomly choosen
                #reservoir's neuron activations
                neuron_out_records[trainLen-t-1] = round(res_out[choosen_neuron],2)
                
            #we perform linear regression after the last epoch of training
            #so we only store the activations of the last epoch
            if epoch == nb_epoch - 1 :
                X[:,t] = np.vstack((1,u,x))[:,0]
                
        #plot some signals to see if IP works
        plot_activity(recorded_res_out, epoch)
        plot_neuron_activity(neuron_out_records, epoch)
        
    # train the output
    X_T = X.T
    # use ridge regression (linear regression with regularization)
    Wout = np.dot( np.dot(Yt,X_T), np.linalg.inv(np.dot(X,X_T)) + \
        reg*np.eye(1+inSize+resSize) )
    #Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
    #    reg*eye(1+inSize+resSize) ) )
    # use pseudo inverse
    #Wout = dot( Yt, linalg.pinv(X) )
     
    #instanciate mean and std to z-score
    mean = np.zeros((outSize,1))
    std = np.ones((outSize,1))
    # run the trained ESN. no need to initialize here, 
    # because x is initialized with training data and we continue from there.
    Y = np.zeros((outSize,testLen))
    u = np.reshape(y_train[trainLen-1],(y_train[trainLen-1].shape[0],1))
    for t in range(testLen):
        #compute reservoir outputs
        if ip_mode == 'intrinsic plasticity on':
            x = (1-a)*x + a*np.tanh( ip_gain * (np.dot( Win, np.vstack((1,u)))\
            + np.dot( W, x )) + ip_bias )
        else :
            x = (1-a)*x + a*np.tanh(np.dot( Win, np.vstack((1,u)) ) \
            + np.dot( W, x ))
        #compute network outputs
        y = np.dot( Wout, np.vstack((1,u,x)))
        #store network activations
        Y[:,t] = y[:,0]
        u = np.reshape(u_test[t],(u_test[t].shape[0],1))
        
    #z-score network outputS
    mean[:,0] = Y.mean(axis=1)
    std[:,0] = Y.std(axis=1)
    Y = (Y - mean) / std
    
    #instanciate counters to compute verb agreement score
    nb_verbs = 0.
    nb_verbs_hit = 0.
    #instanciate arrays to store the first 6 word predictions
    recorded_target_word = []
    
    #compute verb agreement score and plot 6 word predictions
    for i in range(testLen):
        #compute the target word
        target_word = compute_word(y_test[i]) 
        if i<6:
            #collect 6 network activations to plot
            recorded_target_word.append(target_word)
            averaged_recorded_prediction[i,:] += Y[:,i]
        
        #if target word is a verb, store material to compute agreement score
        if verbs.__contains__(target_word) :
            nb_verbs += 1
            #compute the predicted verb class 
            #(singular or plural) given this network output
            verb_class_prediction = compute_class(Y[:,i])
           
            if singular_verbs.__contains__(target_word) \
               & (verb_class_prediction == 'singular'):
                #that's a hit, the network predicted a singular verb
                nb_verbs_hit += 1
            
            if plural_verbs.__contains__(target_word) \
               & (verb_class_prediction == 'plural'):
                #that's a hit, the network predicted a plural verb
                nb_verbs_hit += 1
                
    #compute MSE for the first errorLen time steps
    errorLen = 500
    mse += sum(sum( np.square( y_test[0:errorLen,:] - Y.T[0:errorLen,:] ))) \
           / errorLen
           
    #verb agreement rate
    agreement_rate += nb_verbs_hit / nb_verbs

#average the results over all networks
print 'MSE = ' + str( mse / nb_network )
print 'verb agreement rate :' + str(agreement_rate / nb_network)

#plot the 6 first word prediction during the test set
plot_predictions(recorded_target_word,
                 averaged_recorded_prediction / nb_network)

if ip_mode == 'intrinsic plasticity on':
    #plot the evolution of gain and bias to see if they converge
    figure(10).clear()
    plot( record_ip_gain / nb_network, label='gain' )
    plot( record_ip_bias / nb_network, label='bias' )
    legend()
    ylabel('mean value')
    xlabel('number of timesteps')
    title('Evolution of the mean of gain and bias '\
    'relative to intrinsic plasticity')

show()
