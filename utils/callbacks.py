import sys
import time
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from torch.optim.optimizer import Optimizer


###############################################################################
# TRAINING CALLBACKS
###############################################################################

class PlotLearning(object):
    def __init__(self, args, save_path, num_classes):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.train_acc = []
        self.val_acc = []
        self.save_path_lr = os.path.join(save_path, 'lr_plot.png')
        self.save_path_loss = os.path.join(save_path, 'loss_plot.png')
        self.save_path_accuracy = os.path.join(save_path, 'accuracy_plot.png')
        

    def plot(self, logs):
        self.train_losses.append(logs.get('train_loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('train_acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.learning_rates.append(logs.get('learning_rate'))
        

        plt.figure(1)
        plt.gca().cla()
        plt.plot(self.learning_rates, label='learning_rates')

        plt.title("Learning_rates")
        plt.savefig(self.save_path_lr)

        best_val_loss = min(self.val_losses)
        best_train_loss = min(self.train_losses)
        best_val_epoch = self.val_losses.index(best_val_loss)
        best_train_epoch = self.train_losses.index(best_train_loss)

        plt.figure(2)
        plt.gca().cla()
        plt.plot(self.train_losses, label='train')
        plt.plot(self.val_losses, label='valid')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_loss, best_train_epoch, best_train_loss))
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(self.save_path_loss)
        
        
        
        best_val_acc = max(self.val_acc)
        best_train_acc = max(self.train_acc)
        best_val_epoch = self.val_acc.index(best_val_acc)
        best_train_epoch = self.train_acc.index(best_train_acc)

        plt.figure(3)
        plt.gca().cla()
        plt.ylim(0, 1)
        plt.plot(self.train_acc, label='train')
        plt.plot(self.val_acc, label='valid')
        

        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_acc, best_train_epoch, best_train_acc))
        plt.legend()
        plt.savefig(self.save_path_accuracy)



# Taken from PyTorch's examples.imagenet.main
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
