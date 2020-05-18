from torch import empty
from torch import set_grad_enabled
from torch import tanh, cosh, sinh, exp
import random
import math
import matplotlib.pyplot as plt

class Optimizer(object):
    """
    Superclass of the optimizer modules
    """
    def step(self):
        raise NotImplementedError
    def zero_grad(self):
        raise NotImplementedError
        
# class SGD(Optimizer):
#     """
#     Vanilla stochastic gradient descent module
#     """
#     def __init__(self, params, lr):
#         super(SGD, self).__init__()
#         self.params = params
#         self.lr = lr
        
            
#     def step(self):
#         for param, gradient in self.params:
#             param = param.sub_(self.lr * gradient)
#         return self.params
    
#     def zero_grad(self):
#         for _, gradient in self.params:
#             gradient = gradient.fill_(0)
#         return self.params
    
class SGD(Optimizer):
    """
    Stochastic gradient descent module
    """
    def __init__(self, params, lr, momentum=0, lambda_=0, L1=False, L2=False):
        super(SGD, self).__init__()
        self.params = []
        self.lr = lr
        self.momentum = momentum
        self.lambda_ = lambda_
        self.L1 = L1
        self.L2 = L2
        for param, gradient in params:
            self.params.append((param.clone(), param, gradient))
            
    def step(self):
        for v, param, gradient in self.params:
            v = (v.mul_(self.momentum)).add_(gradient.mul_(self.lr))
            if self.L1:
                param = param.sub_(v).sub_(self.lambda_ * param.sign())
            elif self.L2:
                param = param.sub_(v).sub_(2 * self.lambda_ * param)
            else:
                param = param.sub_(v)
        return self.params
    
    def zero_grad(self):
        for _, _, gradient in self.params:
            gradient = gradient.fill_(0)
        return self.params
    
    
# class SGDMomentum(Optimizer):
#     #https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf
#     """
#     Stochastic gradient descent module with momentum 
#     """
#     def __init__(self, params, lr, alpha):  #gamma in slide 13 (5.2) is alpha*lr
#         super(SGDMomentum, self).__init__()
#         self.params = params
#         self.lr = lr
#         self.alpha = alpha
#         self.speed_w = empty([len(self.params[0][0]), len(self.params[0])]).fill_(0.)
#         self.speed_b = empty([len(self.params[1][0]),1]).fill_(0.)
#         self.temp_1 = empty([len(self.params[0][0]), len(self.params[0])]).fill_(0.)
#         self.temp_2 = empty([len(self.params[1][0]),1]).fill_(0.)
    
#     def step(self):
#         self.temp_1 = self.params[0][0] - (self.lr * self.params[0][1]) #+ self.alpha * self.speed_w)
#         self.temp_2 = self.params[1][0] - (self.lr * self.params[1][1]) #+ self.alpha * self.speed_b)
#         #update
#         #self.speed_w = self.lr * self.params[0][1]
#         #self.speed_b = self.lr * self.params[1][1]
#         self.params = ([self.temp_1, self.params[0][1]], [self.temp_2, self.params[1][1]])
        
#         return self.params
    
#     def zero_grad(self):
#         for _, gradient in self.params:
#             gradient = gradient.fill_(0)
#         return self.params
        