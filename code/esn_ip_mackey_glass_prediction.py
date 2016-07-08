"""
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" scientific Python.
by Mantas Lukosevicius 2012
http://minds.jacobs-university.de/mantas
---
Modified by Xavier Hinaut: 19 November 2015.
http://www.xavierhinaut.com

Modified by Remy Portelas: 30 May 2016
https://github.com/rPortelas/ip_in_esn
"""
from numpy import *
from matplotlib.pyplot import *
import scipy.linalg

def set_seed(seed=None):
    """Making the seed (for random values) variable if None"""

    # Set the seed
    if seed is None:
        import time
        seed = int((time.time()*10**6) % 10**12)
        
    try:
        random.seed(seed) #np.random.seed(seed)
        print "Seed used for random values:", seed
    except:
        print "!!! WARNING !!!: Seed was not set correctly."
    return seed

#reservoir's neurons activation function    
def sigmoid(x):
    if activation_function_mode == 'tanh':
        return tanh(x)
    elif activation_function_mode == 'fermi':
        return ( 1 / ( 1 + exp(-x)))
    else:
        raise Exception, "ERROR: 'activation_function_mode' was not " + \
        "set correctly."

#plot the activation of all neurons in the reservoir during 1 epoch
def plot_activity(x,epoch):
    figure(epoch+42).clear()
    if activation_function_mode == 'tanh':
         hist(x.ravel(), bins = 200)
         xlim(-1,+1)
    else :#fermi
        hist(x.ravel(), bins = 100)
        xlim(0,+1)
    xlabel('neurons outputs')       
    ylabel('number of neurons')   
    #compute some caracteristics of the distribution
    mean = str(round(x.mean(), 2))
    med = str(round(median(x), 2))
    min = str(round(x.min(), 2))
    max = str(round(x.max(), 2))
    std_dev = str(round(x.std(), 2))
    title('Spatio-temporal distribution of the reservoir at epoch ' + \
    str(epoch) + '\n mean = '+ mean + ' median = ' + med + ' min = ' + \
    min + ' max = ' + max + ' std_dev = ' + std_dev)
 
def plot_neuron_activity(x,epoch):
    figure(epoch+84).clear()
    if activation_function_mode == 'tanh':
         hist(x.ravel(), bins = 200)
         xlim(-1,+1)
    else :#fermi
        hist(x.ravel(), bins = 100)
        xlim(0,+1)
    xlabel('neuron outputs')       
    ylabel('number of timesteps')
    #compute some caracteristics of the distribution
    title('The output distribution of a single randomly chosen neuron ' + \
    'at epoch ' + str(epoch))

# load the data
trainLen = 2000
testLen = 2000

data = loadtxt('../datasets/MackeyGlass_t17.txt')

# plot some of it
figure(10).clear()
plot(data[0:1000])
title('A sample of data')

mode = 'prediction'  #given x try to predict x+1
#mode = 'generative' #compute x and use it as an input to compute x+1

activation_function_mode = 'tanh'
#activation_function_mode = 'fermi'

wout_mode = 'entries bias and resOut'
#wout_mode = 'resOut and bias only'

ip_mode = 'intrinsic plasticity on'
#ip_mode = 'intrinsic plasticity off'

#IP parameter
#ip_update_mode = 'leaky neurons treated'
ip_update_mode = 'leaky neurons ignored'

#Set the number of training's epochs,
if ip_mode == 'intrinsic plasticity on':
  if activation_function_mode == 'tanh':
      nb_epoch = 3 #IP with tanh neurons needs less time to converge
  else:
    nb_epoch = 41
else: #if no IP, then we don't need multiple epochs of training
  nb_epoch = 1

# generate the ESN reservoir
inSize = outSize = 1 #input/output dimension
resSize = 300 #reservoir size
a = 0.3 # leaking rate
if ip_mode == 'intrinsic plasticity on':
    spectral_radius = 1.
    reg = 0.02  #regularization coefficient
    #init Intrisic Plasticity (IP)
    lr = 0.001  #learning rate   
    if activation_function_mode == 'tanh':
        m = 0.      #mean
        sigma = 0.2 #standard deviation 0.2 gives best results
        var = square(sigma) #variance
    
    else : #fermi
        m = 0.2
    #instanciate some matrix to store the evolution of IP's gain and bias
    ip_gain = ones((resSize, 1))
    record_ip_gain = zeros(((nb_epoch-1) * trainLen , 1))
    record_ip_bias = zeros(((nb_epoch-1) * trainLen , 1))
    ip_bias = zeros((resSize, 1))
    
    
else :          #IP off
    spectral_radius = 1.25
    reg = 1e-8 # regularization coefficient
    
input_scaling = 1.

#change the seed, reservoir performances should be averaged accross at least 
#20 random instances (with the same set of parameters)
our_seed = None #Choose a seed or None
set_seed(our_seed) 

#generation of random weights
Win = (random.rand(resSize,1+inSize)-0.5) * input_scaling
W = random.rand(resSize,resSize)-0.5

# Option 1 - direct scaling (quick&dirty, reservoir-specific):
#W *= 0.135 
# Option 2 - normalizing and setting spectral radius (correct, slow):
print 'Computing spectral radius...',
rhoW = max(abs(linalg.eig(W)[0])) #maximal eigenvalue
print 'done.'
W *= spectral_radius / rhoW

# allocated memory for the design (collected states) matrix
if wout_mode == 'entries bias and resOut':
    X = zeros((1+inSize+resSize,trainLen))
    
elif wout_mode == 'resOut and bias only':
    X = zeros((1+resSize,trainLen))
else :
        raise Exception, "ERROR: 'wout_mode' was not set correctly."

#to display the spatio-temporal activity of an entire epoch
recorded_res_out = zeros((trainLen, resSize))

#choose a random neuron in the res to create a histogram of its activations
choosen_neuron = random.random_integers(resSize - 1)
neuron_out_records = np.zeros(trainLen)

# set the corresponding target matrix directly
Yt = data[None,1:trainLen+1] 

# run the reservoir with the data and collect X
x = zeros((resSize,1))
for epoch in range(nb_epoch):
    for t in range(trainLen):
        u = data[t]
        res_in = dot( Win, vstack((1,u))) + dot( W, x )
        #compute reservoir activations with or without IP
        if ip_mode == 'intrinsic plasticity on':
            res_out = sigmoid( ip_gain * res_in + ip_bias )
            x = (1-a) * x + a * res_out
            #compute delta_bias considering the activation function
            #we don't want to train our network during the first epoch
            if epoch != 0:
                if activation_function_mode == 'tanh':
                    if ip_update_mode == 'leaky neurons ignored':
                        d_ip_bias = (-lr) * ((-(m / var)) + (res_out / var) * \
                        ((2 * var) + 1 - square(res_out) + m * res_out))
                    elif ip_update_mode == 'leaky neurons treated':
                        d_ip_bias = (-lr) * ((-(m / var)) + (x / var ) * \
                        ((2 * var) + 1 - square(x) + m * x))
                    else:
                        raise Exception, "ERROR: 'ip_update_mode' was not " \
                        "set correctly." 
                else: #fermi
                    if ip_update_mode == 'leaky neurons ignored':   
                        d_ip_bias = lr * (1 - (2 + (1/m)) * res_out + \
                        (square(res_out) / m))
                    elif ip_update_mode == 'leaky neurons treated':
                        d_ip_bias = lr * (1 - (2 + (1/m)) * x + \
                        (square(x) / m))
                    else:
                        raise Exception, "ERROR: 'ip_update_mode' was not " \
                        "set correctly." 
                #compute delta_bias and update IP's gain and bias
                ip_bias += d_ip_bias
                ip_gain += (lr / ip_gain) + (d_ip_bias * res_in)
                #store the results to plot them
                record_ip_bias[t + (trainLen * (epoch-1)),0] = ip_bias.mean()
                record_ip_gain[t + (trainLen * (epoch-1)),0] = ip_gain.mean()
               
               
        elif ip_mode == 'intrinsic plasticity off':
            res_out = sigmoid(res_in)
            x = (1-a) * x + a * res_out 
           
        else:
           raise Exception, "ERROR: 'ip_mode' was not set correctly."
        #accumulate values of a randomly choosen reservoir's neuron activations
        neuron_out_records[t] = round(res_out[choosen_neuron],2)
        
        #we perform linear regression after the last epoch of training
        #so we only store the activations of the last epoch
        if epoch == nb_epoch - 1 :
            if wout_mode == 'entries bias and resOut':       
                X[:,t] = vstack((1,u,x))[:,0]        
      
            elif wout_mode == 'resOut and bias only':
                X[:,t] = vstack((1,x))[:,0]
            
            else :
                raise Exception, "ERROR: 'wout_mode' was not set correctly."
        #store spatio-temporal activity of the reservoir
        recorded_res_out[t] = res_out[:,0] 
        
    #plot some signals to see if IP works
    if activation_function_mode == 'tanh':    
        plot_activity(recorded_res_out, epoch)
        plot_neuron_activity(neuron_out_records, epoch)
    if activation_function_mode == 'fermi':
        if(epoch%20 == 0):
            plot_activity(recorded_res_out, epoch)
            plot_neuron_activity(neuron_out_records, epoch)
            
    # plot the evolution of gain and bias during training 
    if ip_mode == 'intrinsic plasticity on':
        figure(10).clear()
        plot( record_ip_gain, label='gain' )
        plot( record_ip_bias, label='bias' )
        legend()
        ylabel('mean value')
        xlabel('number of timesteps')
        title('Evolution of the mean of gain and bias ' \
        'relative to intrinsic plasticity')

# train the output 
X_T = X.T
# use ridge regression (linear regression with regularization)
if wout_mode == 'entries bias and resOut':       
    Wout = dot( dot(Yt,X_T), linalg.inv( dot(X,X_T) + \
    reg*eye(1+inSize+resSize)))
    
elif wout_mode == 'resOut and bias only':
    Wout = dot( dot(Yt,X_T), linalg.inv( dot(X,X_T) + \
    reg*eye(1+resSize)))
        
else :
    raise Exception, "ERROR: 'wout_mode' was not set correctly."
    
# use pseudo inverse
#Wout = dot( Yt, linalg.pinv(X) )

# run the trained ESN with the test set
Y = zeros((outSize,testLen))
u = data[trainLen]

for t in range(testLen):
    res_in = dot( Win, vstack((1,u)) ) + dot( W, x )
    if ip_mode == 'intrinsic plasticity on':        
        res_out = sigmoid(ip_gain * res_in + ip_bias )
        
    elif ip_mode == 'intrinsic plasticity off':
       res_out = sigmoid(res_in)
       
    else:
       raise Exception, "ERROR: 'ip_mode' was not set correctly." 
    x = (1-a) * x + a * res_out  
    
    if wout_mode == 'entries bias and resOut':       
        y = dot( Wout, vstack((1,u,x)) )
    
    elif wout_mode == 'resOut and bias only':
        y = dot( Wout, vstack((1,x)))
        
    else :
        raise Exception, "ERROR: 'wout_mode' was not set correctly."
      
    Y[:,t] = y
    if mode == 'generative':
        # generative mode:
        u = y
    elif mode == 'prediction':
        # predictive mode:
        u = data[trainLen+t+1] 
    else:
        raise Exception, "ERROR: 'mode' was not set correctly."

# compute MSE for the first errorLen time steps
errorLen = 500
mse_for_each_t = square( data[trainLen+1:trainLen+errorLen+1] - \
Y[0,0:errorLen] )
mse = sum( mse_for_each_t ) / errorLen
print 'MSE = ' + str( mse ) 
print 'compared to max default (Mantas) error 2.91524629066e-07'\
'(For prediction / 100 Neurons)'
print 'ratio compared to (Mantas) error ' + str(mse/2.91524629066e-07) + \
'  (For prediction / 100 Neurons)'
print "" 
print 'compared to max default (Mantas) error 4.06986845044e-06 '\
'(For generation / 1000 Neurons)' 
print 'compared to max default (Mantas) error 2.02529702465e-08 '\
'(For prediction / 1000 Neurons)' 
show()
