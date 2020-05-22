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
    