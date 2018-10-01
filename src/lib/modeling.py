from torch.autograd import Variable
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


def train_model(network, criterion, optimizer, trainLoader, valLoader, scheduler,lr_step=False, n_epochs=20, use_gpu=True):
    
    training_accuracy_values = []
    validation_accuracy_values = []
    
    training_loss_values = []
    validation_loss_values = []
    
    if use_gpu:
        network = network.cuda()
        criterion = criterion.cuda()
        
    # Training loop.
    for epoch in range(0, n_epochs):
        correct = 0.0
        cum_loss = 0.0
        counter = 0

        # Make a pass over the training data.
        t = tqdm(trainLoader, desc = 'Training epoch %d' % epoch)
        network.train()  # This is important to call before training!
        
        for (i, (inputs, labels)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            
            # Added
            # -----
            #if use_gpu:
                #loss = loss.cuda()

            # Backward pass:
            optimizer.zero_grad()
            # Loss is a variable, and calling backward on a Variable will
            # compute all the gradients that lead to that Variable taking on its
            # current value.
            loss.backward() 

            # Weight and bias updates.
            optimizer.step()

            # logging information.
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.max(1)
            correct += (max_labels == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)


        accuracy = 100 * correct / counter
        training_accuracy_values.append(float(accuracy))
        training_loss_values.append(float(cum_loss / (1 + i)))

        # Make a pass over the validation data.
        # -------------------------------------

        correct = 0.0
        cum_loss = 0.0
        counter = 0
        t = tqdm(valLoader, desc = 'Validation epoch %d' % epoch)
        network.eval()  # This is important to call before evaluating!
        for (i, (inputs, labels)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)

            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # logging information.
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.max(1)
            correct += (max_labels == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)

        accuracy = 100 * correct / counter
        validation_accuracy_values.append(float(accuracy))
        validation_loss_values.append(float(cum_loss / (1 + i)))

        # Scheduler
        # ---------
        if lr_step:
            scheduler.step()

    value_lists = [training_accuracy_values,validation_accuracy_values,training_loss_values,validation_loss_values]
    return value_lists