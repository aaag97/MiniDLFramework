from torch import empty
from torch import set_grad_enabled
from torch import tanh, cosh, sinh, exp
import random
import math
import matplotlib.pyplot as plt
from data import *
from modules import *
from optimizer import *
import matplotlib.pyplot as plt

def test(network, test_input, test_target, mini_batch_size=25):
    """
    function to test the performance of the trained net
    Params:
    network - trained model
    test_input - data samples for testing
    test_target - ground truth classes
    mini_batch_size - batch size for batch processing
    Returns: 
    accuracy - accuracy of the model 
    test_mse - test loss 
    """
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
    """
    function to train the model
    Params:
    network - model
    optimizer - training optimizer 
    train_input - data samples for training
    train_target - ground truth classes
    epochs - number of epochs
    mini_batch_size - batch size for batch processing
    Returns: 
    network - trained model
    """
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
    return network, accuracy, losses

def train_and_test(network, optimizer, train_input, train_target, test_input, test_target, epochs=25, mini_batch_size=1):
    """
    function to train the model and test model throughout the epochs
    Params:
    network - model
    optimizer - training optimizer 
    train_input - data samples for training
    train_target - ground truth classes
    epochs - number of epochs
    mini_batch_size - batch size for batch processing
    Returns: 
    network - trained model
    """
    print('STARTING TRAINING')
    train_losses = []
    train_accuracy = []
    test_losses = []
    test_accuracy = []
    
    for epoch in range(epochs):
        train_losses_tmp = 0
        train_errors = 0
        test_losses_tmp = 0
        test_errors = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = network.forward(test_input.narrow(0, b, mini_batch_size))
            _, predicted_classes = output.max(1)
            for i in range(mini_batch_size):
                if test_target[b + i] != predicted_classes[i]:
                    test_errors += 1
            mse = MSE(output, one_hot_encode(test_target.narrow(0, b, mini_batch_size)))
            error = mse.forward()
            test_losses_tmp += error.sum()
            
            optimizer.zero_grad()
            output = network.forward(train_input.narrow(0, b, mini_batch_size))
            _, predicted_classes = output.max(1)
            for i in range(mini_batch_size):
                if train_target[b + i] != predicted_classes[i]:
                    train_errors += 1
            mse = MSE(output, one_hot_encode(train_target.narrow(0, b, mini_batch_size)))
            error = mse.forward()
            train_losses_tmp += error.sum()
            grad = mse.backward()
            network.backward(grad)
            optimizer.step()
        train_accuracy_tmp = 100 - train_errors*100/nb_samples
        test_accuracy_tmp = 100 - test_errors*100/nb_samples
        print('Epoch {}: Train MSE: {}; Train Accuracy: {}\n        Test MSE: {}; Train Accuracy: {}'\
              .format(epoch,
                      train_losses_tmp,
                      train_accuracy_tmp,
                      test_losses_tmp, 
                      test_accuracy_tmp))
        print("="*70)
        train_accuracy.append(train_accuracy_tmp)
        train_losses.append(train_losses_tmp)
        test_accuracy.append(test_accuracy_tmp)
        test_losses.append(test_losses_tmp)        
    print('TRAINING ENDED')
    return network, train_accuracy, train_losses, test_accuracy, test_losses


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

    optimizer = SGD(network.param(), lr=0.05, momentum=0.3, lambda_=0.00001, L2=True)

    print('NETWORK INITIALIZED')
#     network, train_accuracies, train_mse, test_accuracies, test_mse = train_and_test(network, optimizer, train_input, train_target, test_input, test_target, epochs=25, mini_batch_size=25)
    network, train_accuracies, train_mse = train(network, optimizer, train_input, train_target, epochs=25, mini_batch_size=25)
    test_accuracy, test_mse = test(network, test_input, test_target, mini_batch_size=25)
    
    fig, axes = plt.subplots(1,2, figsize=(12, 4))
    
    axes[0].plot(train_mse, label='Train MSE')
#     axes[0].plot(test_mse, label='Test MSE')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title('Evolution of MSE loss through the epoch')
#     plt.legend()
    axes[1].plot(train_accuracies, label='Train Accuracy')
#     axes[1].plot(test_accuracies, label='Test Accuracy')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Evolution of accuracy through the epochs")
#     plt.legend()
    plt.savefig('accuracyandloss')
    
    



if __name__ == '__main__':
    main()