import os
import sys
import signal
import importlib
import time
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


from utils.train import *
from utils.callbacks import (PlotLearning, AverageMeter)
from core.layers import ActionSimilarityNetwork
from data.loader import VideoEncodings
from core.losses import temp_loss, cont_loss, accuracy

import cv2
import imageio
import pickle
import json
from PIL import Image



import multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

def train(args, train_loader, act_sim_net, optimizer, epoch):
    total_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    torch.autograd.set_detect_anomaly(True)
    

    # switch to train mode
    act_sim_net.train()

    start = time.time()
    # Want random length trajectories if video enc
    
    len_dataloader =len(train_loader) 
    data_source_iter = iter(train_loader)
    i = 0
    while i < len_dataloader:
        # training model using source data (Human data here, has labeled tasks)
        data_source = data_source_iter.__next__()
        pos_data, anchor_data, neg_data  = data_source

        
        # Encoded videos
        pos_enc = pos_data.to(device)
        anchor_enc = anchor_data.to(device)
        neg_enc = neg_data.to(device)
    
        act_sim_net.zero_grad()
        
        # Forward pass and calculate loss
        pos_pair, neg_pair, pos, anchor, neg = act_sim_net.forward(pos_enc, anchor_enc, neg_enc)
           
            
        temp_reg_loss = temp_loss(pos, anchor, neg)
            
        
        Cont_loss = cont_loss(pos_pair, neg_pair) 
    
            
        loss = temp_reg_loss + Cont_loss
                
        acc =  accuracy(pos_pair, neg_pair)

        # record loss
        losses.update(loss.item(), 1)
        top1.update(acc, 1)

        # compute gradient and do SGD step for task classifier
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        
        grad_norm = []
        for param in act_sim_net.parameters():
    	    if param.grad is not None:
               grad_norm.append(torch.norm(param.grad).item() ** 2)  # Compute and store squared gradient norm.
               
        total_grad_norm = torch.sqrt(torch.tensor(grad_norm).sum())  # Compute the square root of the sum of squared norms.
        

        
        end = time.time()
        total_time.update(end - start)

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.4f} ({top1.avg:.4f})\t' .format(
                      epoch, i, len_dataloader, loss=losses, top1=top1))
            #print("sim_mat", torch.matmul(class_out, class_out.t()))
        i += 1
    return losses.avg, total_grad_norm, top1.avg, total_time.avg

def valid(args, val_loader, act_sim_net, epoch):
    total_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    act_sim_net.eval()

    
    start = time.time()
    with torch.no_grad():
        data_source_iter = iter(val_loader)
        i = 0
        while i < len(val_loader):
            data_source = data_source_iter.__next__()
            pos_data, anchor_data, neg_data  = data_source

            # Encoded videos
            pos_enc = pos_data.to(device)
            anchor_enc = anchor_data.to(device)
            neg_enc = neg_data.to(device)
           
           
            
            # Forward pass and calculate loss
            pos_pair, neg_pair, pos, anchor, neg = act_sim_net.forward(pos_enc, anchor_enc, neg_enc)
            
            
            temp_reg_loss = temp_loss(pos, anchor, neg)
            
            Cont_loss = cont_loss(pos_pair, neg_pair)
            
    
            
            loss = temp_reg_loss + Cont_loss #+ tcnloss
            
            acc =  accuracy(pos_pair, neg_pair)
            
            losses.update(loss.item(), 1)
            top1.update(acc, 1) #class_out.size(0)

            # measure elapsed time
            total_time.update(time.time() - start)
            
            i += 1
            
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))        
    return losses.avg, top1.avg, total_time.avg

def main():
    # load args
    args = load_args()


    # setup device - CPU or GPU
    device, device_ids = setup_cuda_devices(args)
    print(" > Using device: {}".format(device.type))
    print(" > Active GPU ids: {}".format(device_ids))
    
    best_loss = float('Inf')


    args.tasks = [int(i) for i in args.tasks]
    
    args.num_tasks = len(args.tasks)
    
    # set run output folder
    save_dir = args.log_dir + '_tasks' + str(args.num_tasks) + '_lr' + str(args.lr) + '_traj' + str(args.traj_length) + '_tasklist'
    
    for num in args.human_tasks:
        save_dir += str(num) 
    
    
    print(" > Output folder for this run -- {}".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'plots'))
        os.makedirs(os.path.join(save_dir, 'model'))
    with open(save_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.log_dir = save_dir

        
    act_sim_net = ActionSimilarityNetwork(args).to(device)
    act_sim_net.print_module_details()
                  

    train_enc = VideoEncodings(args,
                             root=args.human_data_dir,
                             json_file_input=args.json_data_train,
                             json_file_labels=args.json_file_labels,
                             clip_size=args.traj_length,
                             nclips=1,
                             step_size=1,
                             num_tasks=args.num_tasks,
                             is_val=False,
                             )

    print(" > Using {} processes for data loader.".format(2))

    train_loader = torch.utils.data.DataLoader(
        train_enc,
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        drop_last=True)

    val_enc = VideoEncodings(args, 
                           root=args.human_data_dir,
                           json_file_input=args.json_data_val,
                           json_file_labels=args.json_file_labels,
                           clip_size=args.traj_length,
                           nclips=1,
                           step_size=1,
                           num_tasks=args.num_tasks,
                           is_val=True,
                           )

    val_loader = torch.utils.data.DataLoader(
        val_enc,
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        drop_last=True)


    # define optimizer
    learning_rate = args.lr
    last_learning_rate = 1e-3
    
    params = list(act_sim_net.parameters())
    print("Number of learnable params", sum(p.numel() for p in act_sim_net.parameters() if p.requires_grad))
    optimizer = torch.optim.SGD(params, learning_rate, momentum=0.9, weight_decay=1e-5)

    # set callbacks
    plotter = PlotLearning(args, os.path.join(
        args.log_dir, "plots"), args.num_tasks)
        
    learning_rate_decayer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    val_loss = float('Inf')

    print(" > Training is getting started...")
    print(" > Training takes {} epochs.".format(args.num_epochs))
    start_epoch = args.resume if args.resume > 0 else 0
    
    report_losses = {}
    
    report_losses['train_loss'] = []
    report_losses['val_loss'] = []
    report_losses['total_grad_norm'] =[]
    report_losses['train_acc'] =[]
    report_losses['val_acc'] =[]
    


    for epoch in range(start_epoch, args.num_epochs):

        learning_rate = [params['learning_rate'] for params in optimizer.param_groups]
        print(" > Current Learning Rate -- {}".format(learning_rates))
        if np.max(learning_rate) < last_learning_rate and last_learning_rate > 0:
            print(" > Training is DONE by learning rate {}".format(last_learning_rate))
            sys.exit(1)


        train_loss, total_grad_norm, train_acc, time = train(args, train_loader, act_sim_net, optimizer, epoch)
        print('Time for {} epoch is {}' .format(epoch, time))
        
        # evaluate on validation set
        if epoch % args.log_freq == 0:
            print("Evaluating on epoch", epoch)
            val_loss, val_acc, time = valid(args, val_loader, act_sim_net, epoch)
            print("Validation loss:", val_loss, "Evaluation Time:", time)
                
            # set learning rate
            learning_rate_decayer.step(val_loss)

            #plot learning
            plotter_dict = {}
            plotter_dict['train_loss'] = train_loss
            plotter_dict['val_loss'] = 0 
            plotter_dict['learning_rate'] = learning_rate
            plotter_dict['val_loss'] = val_loss
            plotter_dict['train_acc'] = train_acc
            plotter_dict['val_acc'] = val_acc

            
            plotter.plot(plotter_dict)
            
            report_losses['train_loss'].append(train_loss)
            report_losses['val_loss'].append(val_loss)
            report_losses['train_acc'].append(train_acc)
            report_losses['val_acc'].append(val_acc)
            report_losses['total_grad_norm'].append(total_grad_norm)
            
            np.savetxt(args.log_dir + '/train_loss.txt', np.array(report_losses['train_loss']), fmt='%f')
            np.savetxt(args.log_dir + '/val_loss.txt', np.array(report_losses['val_loss']), fmt='%f')
            np.savetxt(args.log_dir + '/train_acc.txt', np.array(report_losses['train_acc']), fmt='%f')
            np.savetxt(args.log_dir + '/val_acc.txt', np.array(report_losses['val_acc']), fmt='%f')
            np.savetxt(args.log_dir + '/grad_norm.txt', np.array(report_losses['total_grad_norm']), fmt='%f')

                
            # remember best loss and save the checkpoint
            freq = 30
            if (epoch + 1) % freq == 0:
                is_best = val_loss < best_loss
                best_loss = min(val_loss, best_loss)

                #save act_sim_net
                save_path = os.path.join(args.log_dir, 'model', str(epoch+1) + 'act_sim_net.pth.tar')
                torch.save(act_sim_net.state_dict(), save_path)
            

if __name__ == '__main__':
    main()
