import av
import torch
import numpy as np
import os
import h5py
import pickle

from parser import Dataset

import torchvision

from collections import defaultdict, Counter
import json


FRAMERATE = 12  # default value


class VideoEncodings(torch.utils.data.Dataset):

    def __init__(self, args, root, json_file_input, json_file_labels, clip_size,
                 nclips, step_size, is_val, num_tasks=174,
                 augmentation_mappings_json=None, augmentation_types_todo=None,
                 is_test=False):
        self.num_tasks = num_tasks
        self.is_val = is_val
        self.dataset_object = Dataset(args, json_file_input, json_file_labels,
                                      root, num_tasks=self.num_tasks, is_test=is_test, is_val=is_val)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.im_size = args.im_size
        self.batch_size = args.batch_size

   
        
        classes = []
        for key in self.classes_dict.keys():
            if not isinstance(key, int):
                classes.append(key)
        self.classes = classes
        num_occur = defaultdict(int)
        for c in self.classes:
            for video in self.json_data:
                if video.label == c:
                    num_occur[c] += 1
        if not self.is_val:
            with open(args.log_dir + '/human_data_tasks.txt', 'w') as f:
                json.dump(num_occur, f, indent=2)
        else:
            with open(args.log_dir + '/val_human_data_tasks.txt', 'w') as f:
                json.dump(num_occur, f, indent=2)
                
        # Every sample in batch: anchor (randomly selected class A), positive (randomly selected class A), 
        # and negative (randomly selected class not A)
        # Make dictionary for similarity triplets
        self.json_dict = defaultdict(list)
        for data in self.json_data:
            self.json_dict[data.label].append(data)


        #print("self.total_robot", self.total_robot)
        print("Number of human videos: ", len(self.json_data), len(self.classes), "Total:", self.__len__())
        
        # Tasks used
        self.tasks = args.human_tasks

        assert(sum(num_occur.values()) == len(self.json_data))        
            


    def load_vodeo_encodings(self, item):
         # Open video file
        file_path = item.path
        try:
            loaded_object = np.load(file_path)
            return loaded_object
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except Exception as e:
            print(f"An error occurred while loading the pickle file: {e}")

        return None

    
    def __getitem__(self, index):
       
            
        # Need triplet for each sample
        item = random.choice(self.json_data) 
            
        # Get random anchor
        anchor = random.choice(self.json_dict[item.label])
            
        # Get negative 
        neg = random.choice(self.json_data)
        while neg.label == item.label:
            neg = random.choice(self.json_data)
                
            
        pos_data = self.load_vodeo_encodings(item)  
        anchor_data  = self.load_vodeo_encodings(anchor)
        neg_data = self.load_vodeo_encodings(neg)
        return (pos_data, anchor_data, neg_data)        
    
            

    def __len__(self):
        
        self.total_files = len(self.json_data)
        if not self.is_val and self.num_tasks <= 9:
            self.total_files = self.batch_size * 100
            
        if not self.is_val and self.num_tasks == 12:
            self.total_files = self.batch_size * 200
             
        if not self.is_val and self.num_tasks == 15:
            self.total_files = self.batch_size * 300
            
        if not self.is_val and self.num_tasks == 18:
            self.total_files = self.batch_size * 400
            
        return self.total_files
