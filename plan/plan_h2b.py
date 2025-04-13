from core.layers import MultiColumn, ActionSimilarityNetwork
from utils.train import remove_module_from_checkpoint_state_dict
from utils.plan import get_args, get_obs
from transforms import ComposeMix

from sim_env.tabletop import Tabletop

import metaworld

import pickle
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import torch.optim as optim
from torch import nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import random
import time

from PIL import Image, ImageSequence
import imageio
import cv2
import gtimer as gt
import copy
import json
import importlib
import av
import copy
import colorednoise

import sys
from absl import flags, app



## CONSTANTS ##
TOP_K = 5 # uniformly choose from the top K trajectories
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS

flags.DEFINE_integer('action_dim', 5,'action_dim')
flags.DEFINE_integer('replay_buffer_size', 100,'replay_buffer_size')
flags.DEFINE_integer('hidden_size', 512,'hidden_size')
flags.DEFINE_integer('im_size', 120, 'im_size')
flags.DEFINE_integer('env_log_freq', 10000, 'env_log_freq')
flags.DEFINE_integer('verbose', 1, 'verbose')
flags.DEFINE_integer('num_epochs', 1,'num_epochs')
flags.DEFINE_integer('traj_length', 20,'traj_length')
flags.DEFINE_integer('num_traj_per_epoch', 3, 'num_traj_per_epoch')
flags.DEFINE_integer('batch_sz', 32, 'batch_sz')
flags.DEFINE_integer('elite_size', 5, 'elite set')
flags.DEFINE_float('random_act_prob', 0.01, 'random_act_prob')
flags.DEFINE_integer('grad_steps_per_update', 20,'grad_steps_per_update')
flags.DEFINE_bool('dynamics_var', False, 'dynamics_var')
flags.DEFINE_integer('logging', 2,'logging')
flags.DEFINE_string('root', '/home/yasir/yas_ws/catkin_ws/src/dvd/','root')
flags.DEFINE_integer('task_num', 5,'task_num')
flags.DEFINE_integer('num_tasks', 6,'num_tasks')
flags.DEFINE_integer('seed', 120, 'seed')
flags.DEFINE_string('xml', 'updated','xml')
flags.DEFINE_bool('pretrained', True, 'pretrained')
flags.DEFINE_bool('sanity_check', False, 'sanity_check')
flags.DEFINE_integer('sample_sz', 100,'sample_sz')
flags.DEFINE_integer('cem_iters', 0,'cem_iters')
flags.DEFINE_integer('num_demos', 3, 'num_demos')
flags.DEFINE_string('demo_path', 'demos/','demo_path')
flags.DEFINE_bool('similarity', False, 'similarity')
flags.DEFINE_integer('phorizon', 40, 'phorizon')
flags.DEFINE_integer('robot', 0, 'robot')
flags.DEFINE_float('noise_beta', 0, '1: for pink, 2 for red')
flags.DEFINE_bool('random', False, 'if planning with random actions')
flags.DEFINE_bool('robot_demo', False, 'whether to use robot demo or not')

def save_im(im, name):
    img = Image.fromarray(im.astype(np.uint8))
    img.save(name)
    

def gen_hd_traj(args, env, act_hd):
    actions_hd = act_hd
    
    eps_obs = []
    eps_next = []
    eps_act = []

    for i in range(actions_hd.shape[0]):
        actions = actions_hd[i]
        #actions = actions.squeeze(0)

        obs, env_info = env.reset_model()
        #print("actions after squeeze", actions)
        action_sample = []
        high_dim_sample = []
        high_dim_sample.append(obs)
        for action in actions:
            next_ob, reward, terminal, env_info = env.step(action)
            high_dim_sample.append(next_ob)
            action_sample.append(action)
                   
        
        imgs = np.array(high_dim_sample)
        actions = np.array(action_sample)
        obs = next_ob
        
        eps_obs.append(imgs[:-1])
        eps_next.append(imgs[1:])
        eps_act.append(actions)
            
    eps_obs = np.array(eps_obs)
    eps_next = np.array(eps_next)
    eps_act = np.array(eps_act)

    #print("eps_act shape", eps_act.shape)
    #print("eps_next shape", eps_next.shape)
    #print("eps_obs shape", eps_obs.shape)
    eps_obs = eps_obs.transpose((0, 1, 4, 2, 3))
    #print("eps_obs shape after transpose", eps_obs.shape)
    
    return eps_obs

def gen_meta_traj(args, env, act_hd):
    actions_hd = act_hd
    
    eps_obs = []
    eps_next = []
    eps_act = []

    for i in range(actions_hd.shape[0]):
        actions = actions_hd[i]
        #actions = actions.squeeze(0)

        obs = env.reset()
        obs = obs['pixels']
        #print("actions after squeeze", actions)
        action_sample = []
        high_dim_sample = []
        high_dim_sample.append(obs)
        for action in actions:
            next_ob, reward, done, env_info = env.step(action)
            next_ob = next_ob['pixels']
            high_dim_sample.append(next_ob)
            action_sample.append(action)
                   
        
        imgs = np.array(high_dim_sample)
        actions = np.array(action_sample)
        obs = next_ob
        
        eps_obs.append(imgs[:-1])
        eps_next.append(imgs[1:])
        eps_act.append(actions)
            
    eps_obs = np.array(eps_obs)
    eps_next = np.array(eps_next)
    eps_act = np.array(eps_act)

    #print("eps_act shape", eps_act.shape)
    #print("eps_next shape", eps_next.shape)
    #print("eps_obs shape", eps_obs.shape)
    eps_obs = eps_obs.transpose((0, 1, 4, 2, 3))
    #print("eps_obs shape after transpose", eps_obs.shape)
    
    return eps_obs   

def custom_slice_images(imgs, phorizon):
    total_imgs = len(imgs)
    if phorizon >= total_imgs:
        return imgs

    # Always include the first and last image
    indices = [0]
    interval = (total_imgs - 2) / (phorizon - 2)

    # Calculate intermediate indices, maintaining temporal order
    for i in range(1, phorizon - 1):
        indices.append(int(1 + i * interval))

    indices.append(total_imgs - 1)
    return [imgs[i] for i in indices]

class CEM(object):
    def __init__(self, args, savedir, phorizon,
               cem_samples, cem_iters):
        
        self.eps = 0
        self.savedir = savedir
        self.planstep = 0
        self.phorizon = phorizon
        self.cem_samples = cem_samples
        self.cem_iters = cem_iters
        self.verbose = False
        self.num_acts = args.action_dim
        self.factor_decrease_num = 1.25
        self.noise_beta = args.noise_beta
        #self.previous_mu1 = None

        


    def cem(self, args, curr, clusters, env, eps, planstep, verbose, trained_net,  act_sim_net, transform=None):
        """Runs Visual MPC between two images."""    
        horizon = self.phorizon

        mu1 = np.zeros(self.num_acts * horizon)
        sd1 = np.array([0.2]*(self.num_acts * horizon))
        
        _iter = 0
        nstep = 0
        rewards = []
        env_step = []
        sample_size = self.cem_samples 
        resample_size = args.elite_size
        #print("resample_size", resample_size)
        
        hz = horizon

        while _iter <= self.cem_iters:
            
            #print("_iter", _iter)
            # Decay of sample size
            if _iter > 0:  # Important improvement
                sample_size = max(resample_size * 2, int(sample_size / self.factor_decrease_num))
                #print("sample_size", sample_size)
                
            if _iter == 0:
                acts1 = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=[sample_size, hz, self.num_acts])
            
            else:
                #acts1 = np.random.normal(mu1, sd1, (sample_size, self.num_acts *  hz))
                acts1 = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size=[sample_size, self.num_acts*hz])
                mu1_reshaped = mu1.reshape((1, self.num_acts*hz))
                sd1_reshaped = sd1.reshape((1, self.num_acts*hz))

                acts1 = acts1 * sd1_reshaped + mu1_reshaped  # Scale and shift the noise
                acts1 = acts1.reshape((sample_size, hz, self.num_acts))
                acts1 = np.clip(acts1, env.action_space.low, env.action_space.high)

            #acts1 = acts1.reshape((sample_size, hz, self.num_acts))
            #print("All for {} iteration {}:".format(_iter, acts1))
            start = time.time()
            
            if args.task_num in [5, 41, 93]:
                forward_predictions = gen_hd_traj(args, env, acts1)
            else:
                forward_predictions = gen_meta_traj(args, env, acts1) 
            nstep += self.num_acts *  hz
            
            
            preds = torch.FloatTensor(forward_predictions).cuda()
            transform = ComposeMix([
                [torchvision.transforms.Normalize(
                           mean=[0.485, 0.456, 0.406],  # default values for imagenet
                           std=[0.229, 0.224, 0.225]), "img"]
                 ])

            if transform is not None:                    
                preds = transform(preds, online=True)
            preds = [preds.permute(0, 2, 1, 3, 4)] #B, C, T, W, H
            obs = []
            sub_len = 1
            rews = []
            demos = torch.tensor(clusters).cuda()
            #print("demos.shape", demos.shape)
            #demos = torch.cat(sub_len * [demos]).reshape(sub_len, -1, args.hidden_size)
            for p in range(sample_size // 1): # in order to fit on the gpu
                ob = trained_net.encode([preds[0][p * sub_len: (p+1)*sub_len]])
                outputs = []
                for i in range(demos.shape[0]):
                    demo_i = demos[i, :, :].unsqueeze(0)  # Shape: [1, 40, 512]
                    output = act_sim_net.forward(ob, demo_i)
                    output = output.mean()
                    outputs.append(output)
                    
                    
                outputs = torch.stack(outputs)
                outputs = torch.mean(outputs)
                rew = outputs.cpu().data.numpy()  
                rews.append(rew)
            
            rewards = np.stack(rews)
            #print("rewards after stack", rew)
                    
            best_actions = np.array([x for _, x in sorted(
          zip(rewards, acts1.tolist()), reverse=True)])
            best_costs = np.array([x for x, _ in sorted(
          zip(rewards, acts1.tolist()), reverse=True)])
            #print("All actions sorted", best_actions)
            #print("All costs sorted", best_costs)

            start = time.time()
            
            best_actions = best_actions[:resample_size]
            #print("best_actions after resample", best_actions)
            best_actions1 = best_actions.reshape(resample_size, -1)
            #print("best_actions after resample reshape", best_actions1[0])
            if _iter < self.cem_iters:
                best_costs = best_costs[:resample_size]
                #print("best_costs", best_costs)
                mu1 = np.mean(best_actions1, axis=0)
                #print("mu1", mu1)
                sd1 = np.std(best_actions1, axis=0)
                #print("sd1", sd1)
                
                _iter += 1
            else:
                break
          
        # Store mu1 and sd1 for the next planning step
        
        #self.previous_mu1 = mu1

        chosen = best_actions1[0]
        bestcost = best_costs[0]
        #print("chosen", chosen)
        #print("best cost", bestcost)
        return chosen, bestcost


def main(argv=None):
    '''Initialize replay buffer, models, and environment.'''
    
    # Get args in argparser form
    args = get_args(FLAGS)
    
    assert(torch.cuda.is_available())
    # Load in models and env
    print("----Load in models and env----")
    
        
    if args.task_num in [5, 41, 93]:
        env = Tabletop(log_freq=args.env_log_freq, filepath=args.log_dir + '/env', xml=args.xml, verbose=args.verbose)
        print('TableTop Environment Initialized')
    
    
    elif args.task_num == 1:
        task = 'pick-place-v2'
        scene = 'pick_place'
        env = metaworld_my.make(task=task, scene=scene, seed=args.seed)
        print(f'Metaworld Environment with the task {task} Initialized')        
    
    else:
        print('Task is not correct')
        
    sv2p = CEM(args, savedir=args.log_dir, phorizon=args.phorizon,
               cem_samples = args.sample_sz, cem_iters = args.cem_iters)
    
    print("----Done loading in models and env----")
    path = args.num_traj_per_epoch # num of trajs per episode
    hz = args.traj_length # traj_length
    full_low_dim = []
    
    ''' Initialize models '''
    model_dir = os.path.join(args.root + 'pretrained/6LS256f_act_sim_net.pth.tar')
    file_name = 'model3D_1'
    column_cnn_def = importlib.import_module("{}".format(file_name))
    
    trained_net = MultiColumn(args, args.num_tasks, column_cnn_def.Model, args.hidden_size)
    act_sim_net = None
    
    # checkpoint path to a trained model
    checkpoint_path = os.path.join("pretrained/sth_video_encoder/model_best.pth.tar")
    print("=> Checkpoint path --> {}".format(checkpoint_path))
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint['state_dict'] = remove_module_from_checkpoint_state_dict(
                                              checkpoint['state_dict'])
        print("Loading in pretrained model")
        trained_net.load_state_dict(checkpoint['state_dict'], strict=False)
        
        act_sim_net = ActionSimilarityNetwork(args).to(device)
        act_sim_net.load_state_dict(torch.load(model_dir, weights_only=False))
        print("act_sim_net loaded from", model_dir)
        act_sim_net.eval()
    else:
        print(" !#! No checkpoint found at '{}'".format(
            checkpoint_path))

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(checkpoint_path, checkpoint['epoch']))
    trained_net.eval()
    trained_net.cuda()
    
    total_params = sum(p.numel() for p in act_sim_net.parameters())
                    
    print(f"Total parameters in sim_discriminator: {total_params}")
    
    transform = ComposeMix([
        [Scale(args.im_size), "img"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(args.im_size), "img"],
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
         ])
    
    if not os.path.exists(args.log_dir + 'rankings/'):
        os.mkdir(args.log_dir + 'rankings/')
    
    # Get clusters
    clusters = None
    if not args.random:
        def get_cluster(demo, clusters, vids, save_demo=True):
            one_shot_demo = args.demo_path + str(vids[demo]) + '.webm'
            reader = av.open(one_shot_demo)
            imgs = []
            imgs = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
            print("imgs.shape", np.array(imgs).shape)
            
            if args.task_num in [5, 51, 93]:
                # Downsample imgs
                downsample = max(1, len(imgs) // 30) #if 60 frames downsample will be 2
                print("down sample", downsample)
                if args.robot_demo:
                    downsample = 1
                 
                if downsample > 2 and downsample < 4:
                    downsample = 2
                if downsample > 4:
                    downsample = 3
                print("down sample after fixing", downsample)
                imgs = imgs[1::downsample][:args.phorizon]  #slicing statring from 1(0 index) while downsample is step size and [:40] means at most 40(can be less than 40)

            else:
                imgs = custom_slice_images(imgs, args.phorizon)    
            
            print("number of frames after downsample", len(imgs)) 
            if save_demo:
                orig_imgs = np.array(imgs).copy()
                with imageio.get_writer(args.log_dir + '/demo' + str(vids[demo]) + '.gif', mode='I') as writer:
                    for k, frame in enumerate(orig_imgs):
                        img = Scale(args.im_size)(frame).astype('uint8')
                        writer.append_data(img)
            if transform:
                imgs = transform(imgs)
                #print("after transform", np.array(imgs).shape) 
            imgs = torch.stack(imgs).permute(1, 0, 2, 3).unsqueeze(0) # want 1, 3, traj_length, 84, 84
            input_var = [imgs.to(device)]
            features = trained_net.encode(input_var)
            features = features.cpu().data.numpy()
            clusters.append(features)
            

        clusters = []
        vids = [num for num in range(args.num_demos)]
        for d in range(args.num_demos):
            get_cluster(d, clusters, vids)
        clusters = np.concatenate(clusters, axis=0)            

    env.max_path_length = args.traj_length * args.num_traj_per_epoch
    report_losses = {}
    report_losses['dynamics_loss'] = []
    
    
    if args.task_num in [5, 41, 93]:
        env.initialize()
    total_good = 0
    results = []
    save_freq = 10 if not args.robot else 1
    
    succes_dir = args.log_dir + '/success'
    os.makedirs(succes_dir, exist_ok=True)
    for eps in range(args.num_epochs):
        eps_low_dim = []
        final_video = []
        start = time.time()

        if args.task_num in [5, 41, 93]:
            obs, env_info = env.reset_model()
            init_im = obs * 255 
        else:
            obs = env.reset()
            obs = obs['pixels']
            init_im = obs
        
        if eps == 0 and args.verbose:
            save_im(init_im, '{}/init.png'.format(args.log_dir))

        if args.robot: #for real robot
            for _ in range(4):
                obs, reward, terminal, action, succ = env.step([0, 0, 0, 0])
                if eps == 0 and args.verbose:
                    save_im(obs*255, '{}/init.png'.format(args.log_dir))
            
        step = 0
        if args.task_num in [5, 41, 93]:
            low_dim_state = get_obs(args, env_info)
            very_start = low_dim_state
            eps_low_dim.append(low_dim_state)
        else:
            final_video.append(obs)
        
        while step < path: # each episode is 3 x 20-step trajectories path=args.num_traj_per_epoch
            if step == 0:
                if args.task_num in [5, 41, 93]:
                    obs = obs * 255
                else:
                    obs = obs
            step_time = time.time()
            if args.random:
                chosen = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=[hz, args.action_dim])
                chosen = chosen.reshape(hz*args.action_dim, -1).squeeze()
                bestcost = 0
            else:
                chosen, bestcost = sv2p.cem(args, obs, clusters, env, eps, step, args.verbose, trained_net, act_sim_net, transform=transform)
            for h in range(hz): 
                if args.robot: #for robot
                    obs, reward, terminal, action, succ = env.step(chosen[args.action_dim*h:(args.action_dim)*(h+1)])
                    # if got error
                    if not succ:
                        print("Got " + str(eps) + " epochs")
                        assert(False)
                else:
                    obs, reward, terminal, env_info = env.step(chosen[args.action_dim*h:(args.action_dim)*(h+1)])
                
                if args.task_num in [5, 41, 93]:
                    obs = obs * 255
                
                else:
                    obs = obs['pixels']
                
                if args.verbose and eps % save_freq == 0:
                    save_im(obs, '{}step{}.png'.format(args.log_dir, step * hz + h))
                    
                if args.task_num in [5, 41, 93]:
                    low_dim_state = get_obs(args, env_info)
                    #print("step", step)
                    #print("low_dim_state", low_dim_state)
                    eps_low_dim.append(low_dim_state)
                    
                elif args.task_num in [1]:
                    final_video.append(obs)
            step += 1
            
        if args.verbose and eps % save_freq == 0:
            total_steps = args.traj_length * args.num_traj_per_epoch
            if args.robot:
                for p in range(4):
                    obs, reward, terminal, action, succ = env.step([0, 0, 0, 0])
                    save_im(obs*255, '{}step{}.png'.format(args.log_dir, args.traj_length * args.num_traj_per_epoch +p))
                total_steps += 4
            with imageio.get_writer('{}{}.gif'.format(args.log_dir, eps), mode='I', duration=125) as writer:
                for step in range(total_steps):
                    img_path = '{}step{}.png'.format(args.log_dir, step)
                    writer.append_data(imageio.imread(img_path))

        if args.task_num in [5, 41, 93]:
            full_low_dim.append(np.array(eps_low_dim))
        
        elif args.task_num in [1]:
            with imageio.get_writer('{}epoch_{}.gif'.format(succes_dir, eps), mode='I', duration=125) as writer:
                for frame in final_video:
                    img = np.array(frame)
                    writer.append_data(img.astype('uint8'))
                    
            print("video {} saved".format(eps))   
        
        else:
            criteria = input(f"Was task {args.task_num} completed?")
            if int(criteria) == 1:
                total_good += 1
                results.append(1)
            else:
                results.append(0)
        end = time.time()
        print("Time for 1 trial", end - start)
        print("-----------------EPS {} ENDED--------------------".format(eps))
        if args.robot:
            time.sleep(10)
    if args.task_num in [5, 41, 93]:
        import pickle
        pickle.dump(np.array(full_low_dim), open(args.log_dir + 'full_states.p', 'wb'))
    else:
        print("Total successes", total_good)
        print("Results", results)
        np.savetxt(args.log_dir + 'results.txt', np.array(results), fmt='%d')
            

if __name__ == "__main__":
    app.run(main)
        
    
