from torch import empty
from torch import set_grad_enabled
from torch import tanh, cosh, sinh, exp
import random
import math
import matplotlib.pyplot as plt
from data import *
from modules import *
from optimizer import *

def test(network, test_input, test_target, mini_batch_size=25):
    print('TESTING NETWORK')
    nb_errors = 0
    loss = 0
    for b in range(0, test_input.size(0), mini_batch_size):
        output = network.forward(test_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for i in range(mini_batch_size):
            if test_target[b + i] != predicted_classes[i]:
                nb_errors += 1
        mse = MSE(output, one_hot_encode(test_target.narrow(0, b, mini_batch_size)))
        error = mse.forward()
        loss += error.sum()
    accuracy = 100 - nb_errors*100/nb_samples
    test_mse = (loss/nb_samples)
    print('Test MSE: {}; Test Accuracy: {}'.format(test_mse, accuracy))
    print('TESTING ENDED')
    return accuracy, test_mse

def train(network, optimizer, train_input, train_target, epochs=25, mini_batch_size=1):
    print('STARTING TRAINING')
    losses = []
    accuracy = []
    for epoch in range(epochs):
        error_tmp = 0
        nb_errors = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            output = network.forward(train_input.narrow(0, b, mini_batch_size))
            _, predicted_classes = output.max(1)
            for i in range(mini_batch_size):
                if train_target[b + i] != predicted_classes[i]:
                    nb_errors += 1
            mse = MSE(output, one_hot_encode(train_target.narrow(0, b, mini_batch_size)))
            error = mse.forward()
            error_tmp += error.sum()
            grad = mse.backward()
            network.backward(grad)
            optimizer.step()
        print('Epoch {}: Train MSE: {}; Train Accuracy: {}'.format(epoch, error_tmp/nb_samples, 100 - nb_errors*100/nb_samples))
        print("="*70)
        accuracy.append(100 - nb_errors*100/nb_samples)
        losses.append(error_tmp/nb_samples)
    print('TRAINING ENDED')
    return network


def main():
    print('INITIALIZING INPUT AND TARGET')
    train_input, train_target = generate_set(nb_samples)
    test_input, test_target = generate_set(nb_samples)
    
    # normalization
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)
    
    print('INITIALIZED INPUT AND TARGET')
    print('INITIALIZING NETWORK')
    network = Sequential(Linear(2, 25, bias=True),
                  ReLU(),
                  Linear(25, 25, bias=True),
                  ReLU(),
                  Linear(25, 25, bias=True),
                  ReLU(),
                  Linear(25, 2, bias=True))

    optimizer = SGD(network.param(), 0.0005, 0.3)
    
    print('NETWORK INITIALIZED')
    netwotk = train(network, optimizer, train_input, train_target, epochs=25, mini_batch_size=1)
    accuracy, test_mse = test(network, test_input, test_target, mini_batch_size=1)
    



if __name__ == '__main__':
    main()