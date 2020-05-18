from torch import empty
from torch import set_grad_enabled
from torch import tanh, cosh, sinh, exp
import random
import math
import matplotlib.pyplot as plt


class Module(object):
    """
    Superclass of all other modules
    """
    def forward(self, *x):
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []
    
    
class Linear(Module):
    """
    Linear module of a multi layer perceptron
    """
    def __init__(self, in_features, out_features, bias):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = empty(out_features, in_features).normal_(0, math.sqrt(2) * math.sqrt(2.0 / (in_features + out_features)))#.uniform_(-math.sqrt(1/in_features), math.sqrt(1/in_features))
        self.dweights = empty(out_features, in_features).fill_(0)
        self.input = None
        self.dinput = None
        if bias:
            self.bias = empty(out_features,).normal_(0, math.sqrt(2) * math.sqrt(2.0 / (in_features + out_features)))#.uniform_(-math.sqrt(1/in_features), math.sqrt(1/in_features))
            self.dbias = empty(out_features,).fill_(0)
        else:
            self.bias = None
            self.dbias = None
        self.nb_samples = 1
            
    def forward(self, x):
        self.input = x
        self.nb_samples = x.size()[0]
        if self.bias is None:
            return self.weights.mm(x)
        else:
            return x.mm(self.weights.t()) + self.bias.unsqueeze(0)
        
    def backward(self, gradwrtoutput):
        self.dweights.add_(gradwrtoutput.view(-1,self.nb_samples).mm(self.input.view(self.nb_samples,-1)))
        self.dinput = gradwrtoutput.mm(self.weights)
        if self.bias is not None:
            self.dbias.add_(gradwrtoutput.sum(0))
        return self.dinput
        
    def param(self):
        if self.bias is None:
            ret = [(self.weights, self.dweights)]
        else:
            ret = [(self.weights, self.dweights), (self.bias, self.dbias)]
        return ret
    
    
class Tanh(Module):
    """
    Tanh activation function module 
    """
    def __init__(self):
        super(Tanh, self).__init__()
        self.s = None
        self.ds = None
            
    def forward(self, x):
        self.s = tanh(x)
        return self.s
        
    def backward(self, gradwrtoutput):
        self.ds = (cosh(self.s) ** 2 -  sinh(self.s) ** 2) / cosh(self.s) ** 2
        return self.ds * gradwrtoutput
        
    def param(self):
        return None
    
    
class ReLU(Module):
    """
    ReLU activation function module 
    """
    def __init__(self):
        super(ReLU, self).__init__()
        self.s = None
        self.ds = None
            
    def forward(self, x):
        self.s = x.clone()
        self.s[self.s <= 0] = 0
        return self.s
        
    def backward(self, gradwrtoutput):
        self.ds = self.s.clone()
        self.ds[self.ds <= 0] = 0
        self.ds[self.ds > 0] = 1
        return self.ds * gradwrtoutput
        
    def param(self):
        return None

    
class LeakyReLU(Module):
    """
    Leaky ReLU activation function module 
    """
    def __init__(self):
        super(LeakyReLU, self).__init__()
        self.s = None
        self.ds = None
            
    def forward(self, x):
        self.s = x.clone()
        self.s[self.s <= 0] = 0.01*self.s
        return self.s
        
    def backward(self, gradwrtoutput):
        self.ds = self.s.clone()
        self.ds[self.ds <= 0] = 0.01
        self.ds[self.ds > 0] = 1
        return self.ds * gradwrtoutput
        
    def param(self):
        return None
    
    
class Sigmoid(Module):
    """
    Sigmoid activation function module 
    """
    def __init__(self):
        self.s = None
        self.ds = None
        
    def forward(self, x):
        self.s = 1 / (1 + exp(-x))
        return self.s

    def backward(self, gradwrtoutput):
        self.ds = (1 / (1 + exp(-self.s))) * (1 - (1 / (1 + exp(-self.s))))
        return self.ds * (gradwrtoutput)
        
    def param(self):
        return None
    
    
class MSE(Module):
    """
    MSE loss function module 
    """
    def __init__(self, output, target):
        super(MSE, self).__init__()
        self.output = output
        self.target = target
        self.s = None
        self.ds = None
            
    def forward(self):
        self.s = ((self.output - self.target)**2).sum(1)
        return self.s
        
    def backward(self):
        self.ds = 2 * (self.output - self.target)
        return self.ds
        
    def param(self):
        return None
    
    
class Sequential(Module):
    """
    Sequential container module 
    """
    def __init__(self, *args):
        super(Sequential, self).__init__()
        modules = list(args)
        self.layers = []
        for module in modules:
            self.layers.append(module)
    
    def forward(self, x):
        output = x
        for module in self.layers:
            output = module.forward(output)
        return output
    
    def backward(self, gradwrtoutput):
        grad = gradwrtoutput
        for i in range(1, len(self.layers)+1):
            grad = self.layers[-i].backward(grad)
        return grad
    
    def param(self):
        params = []
        for module in self.layers:
            if module.param() is not None:
                params += module.param()
        return params
                
        