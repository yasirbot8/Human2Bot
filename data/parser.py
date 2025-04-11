import os
import json
import numpy as np

from collections import namedtuple
from collections import defaultdict

ListData = namedtuple('ListData', ['id', 'label', 'path'])

"""
Adapted from [1]. A. S. Chen, S. Nair, and C. Finn, "Learning Generalizable Robotic Reward Functions 
                  from 'In-The-Wild' Human Videos," in \textit{Robotics: Science and Systems}, 2021. 
                  doi: 10.15607/RSS.2021.XVII.012.
"""

class DatasetBase(object):
    """
    To read json data and construct a list containing video sample `ids`,
    `label` and `path`
    """
    def __init__(self, args, json_path_input, json_path_labels, data_root,
                 extension, num_tasks, is_test=False, is_val=True):
        self.num_tasks = num_tasks
        self.json_path_input = json_path_input
        self.json_path_labels = json_path_labels
        self.data_root = data_root
        self.extension = extension
        self.is_test = is_test
        self.is_val = is_val
        self.sim_dir = args.sim_dir
        
        self.num_occur = defaultdict(int)
        
        self.tasks = args.human_tasks
        
        # preparing data and class dictionary
        self.classes = self.read_json_labels()
        #print('self.classes', self.classes)
        self.classes_dict = self.get_two_way_dict(self.classes)
        self.json_data = self.read_json_input()

        
        
    def read_json_input(self):
        json_data = []
        
        with open(self.json_path_input, 'rb') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                label = self.clean_template(elem['template'])
                if label not in self.classes_dict.keys(): # or label == 'Pushing something so that it slightly moves':
                    continue
                    if label not in self.classes:
                        raise ValueError("Label mismatch! Please correct")
                        
                    label_num = self.classes_dict[label]
                    item = ListData(elem['id'], label, os.path.join(self.data_root, elem['id'] + self.extension))
                    json_data.append(item)
                    self.num_occur[label] += 1

                        
        return json_data

    def read_json_labels(self):
        classes = []
        with open(self.json_path_labels, 'rb') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)

    def get_two_way_dict(self, classes):
        classes_dict = {} 
        tasks = self.tasks
        for i, item in enumerate(classes):
            if i not in tasks:
                continue
            classes_dict[item] = i
            classes_dict[i] = item
        print("Length of keys", len(classes_dict.keys()), classes_dict.keys())
        return classes_dict

    def clean_template(self, template):
        """ Replaces instances of `[something]` --> `something`"""
        template = template.replace("[", "")
        template = template.replace("]", "")
        return template


class Dataset(DatasetBase):
    def __init__(self, args, json_path_input, json_path_labels, data_root, num_tasks, 
                 is_test=False, is_val=False):
        EXTENSION = ".npy"
        super().__init__(args, json_path_input, json_path_labels, data_root,
                         EXTENSION, num_tasks, is_test, is_val)
