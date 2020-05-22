from torch import empty
from torch import set_grad_enabled
from torch import tanh, cosh, sinh, exp
import random
import math
import matplotlib.pyplot as plt

nb_samples = 1000

def generate_set(nb):
    """
    function to generate the data set
    Params: 
    nb - number of samples 
    Returns:
    input_ - matrix with (x,y) coordinates of each data sample
    target - vector with the class of each data sample
    """
    input_ = empty(nb,2).uniform_(0,1)
    target = empty(nb,).fill_(0.)
    r = 1/math.sqrt(2*math.pi)  #boundary
    c = empty((1,2)).fill_(0.5)  #center of the disc
    for i in range(nb):
        if ((input_[i,:] - c) ** 2).sum() ** (1/2) < r:
        #if torch.sqrt(torch.sum(torch.pow(input_[i,:] - c ,2))) < r:
            target[i] = 1. 
    return input_ , target

def visualize_data(nb_samples, dataset, labels):
    """
    function to plot the data set 
    Params:
    nb_samples - number of samples
    dataset - set of 2d data samples 
    labels - vector containing the ground truth classes
    Returns:
    None
    """
    for i in range(nb_samples):
        if labels[i] == 1:
            plt.plot(dataset[i,0], dataset[i,1], '*', color='blue')   #class 1 
        else:
            plt.plot(dataset[i,0], dataset[i,1], '*', color='red')    #class 0
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.legend(['Class 0', 'Class 1'], bbox_to_anchor=(0, 1), loc='lower right', ncol=1)
    
def one_hot_encode(bin_target):
    """
    function to get one hot encoding of a binary vector
    Params:
    bin_target - vector with 0's and 1's
    Returns:
    one_hot_encoded - vector with one hot encodings
    """
    one_hot_encoded = empty((bin_target.size()[0], 2)).fill_(0)
    one_hot_encoded.scatter_(1, bin_target.long().unsqueeze(1), 1)
    return one_hot_encoded


