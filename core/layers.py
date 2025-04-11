import torch.nn as nn
import torch as th
from torch.autograd import Function
import torchvision.models as models
import numpy as np
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3)
        self.conv5 = nn.Conv1d(256, 1, kernel_size=1)


    def forward(self, x):

        batch_size, frame_frame_similarity = x.shape
        
        x = x.view((batch_size, frame_frame_similarity, 1))
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        
        x = th.tanh(x)
        
        x = x.permute(0, 2, 1)
        x = x.view((batch_size, -1))
        x = th.mean(x, dim=1)
        
        return x

class Feature_Reduction_Layer(nn.Module):
    def __init__(self, args):
        super(Feature_Embedding_Layer, self).__init__()  
        self.feature_reducer = nn.Linear(args.hidden_size, args.out_size)
            

    
    def forward(self, x):
        x_sq = x.squeeze(1)
        batch_size, num_frames, feats = x_sq.shape
        x = x.view(batch_size * num_frames, feats)  
        x = self.feature_reducer(x)  
        x = x.view(batch_size, num_frames, -1)  
        return x

class Attention_Layer_Local(nn.Module):
   
    def __init__(self, args):
        super(Attention_Layer_Local, self).__init__()
        
        d_model = args.out_size
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.scale = th.sqrt(th.tensor(d_model, dtype=th.float32))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear transformations
        q = self.q_linear(x)  
        k = self.k_linear(x)  
        v = self.v_linear(x)  
        
        # Scaled dot-product attention
        attn_weights = th.matmul(q, k.transpose(-2, -1)) / self.scale 
        attn_weights = F.softmax(attn_weights, dim=-1)  

        attn_output = th.matmul(attn_weights, v)  
        
        output = self.out_linear(attn_output)  
        
        return output
      
class Attention_Layer_Global(nn.Module):
    def __init__(self, args):
        super(Attention_Layer_Global, self).__init__()
        self.context_vector = nn.Parameter(th.randn(args.out_size, 1), requires_grad=True)

    def forward(self, logits):
        weights = th.matmul(logits, self.context_vector)
        norm_weights = F.softmax(weights, dim=1) # Normalize the features

        return logits * norm_weights
        

class VideoAttentionModel(nn.Module):
    def __init__(self, args):
        super(VideoAttentionModel, self).__init__()
        self.attention_local = Attention_Layer_Local(args)
        self.attention_global = Attention_Layer_Global(args)
        
    def forward(self, x):

        output_loc = self.attention_local(x)
        
        output = self.attention_global(output_loc)
        
        return output
       
class ActionSimilarityNetwork(nn.Module):
    def __init__(self, args):
        
        """
        - Example video encoder

        Args:
        - Input: Takes 3 video tensors, anchor, positive, and negative each of size
                 (batch_size, sequence_length, features)
        - Returns: 1. Attention-modulated video tensors anchor, positive, and negative each of size
                      (batch_size, sequence_length, features)
                   2. Similarity scores for (anchor, positive) and (anchor, negative) pairs of size
                      (batch_size, video_to_video_similarity_score)  
        
        """
        
        super(ActionSimilarityNetwork, self).__init__()
        
        self.feature_encoding = Feature_Embedding_Layer(args) 
        self.attention = VideoAttentionModel(args)
        
        self.conv_net = ConvNet().to(th.device('cuda'))
        
    def print_module_details(self):
        print("SimilarityDiscriminator Model Structure:") 
        for name, param in self.named_parameters():
            print(f"{name}: {param.size()}")
        

    
    def forward(self, x1, x2, x3):
    
        x1 = self.feature_encoding(x1)

        x2 = self.feature_encoding(x2)

        x3 = self.feature_encoding(x3)
        
        x1 = self.attention(x1)
        x2 = self.attention(x2)
        x3 = self.attention(x3)
        
        # calculate frame-to-frame cosine siilarity
        Sap = F.cosine_similarity(x2, x1, dim=2)
        San = F.cosine_similarity(x2, x3, dim=2)
        
        # calculate video-to-video siilarity
        anc_pos = self.conv_net(Sap)
        anc_neg = self.conv_net(San)
        
        
        return anc_pos, anc_neg, x1, x2, x3
            
            
class MultiColumn(nn.Module):

    def __init__(self, args, num_classes, conv_column, column_units,
                 clf_layers=None):
        """
        - Example video encoder

        Args:
        - Input: Takes in a list of tensors each of size
                 (batch_size, 3, sequence_length, W, H)
        - Returns: features of size (batch size, column_units)
        """
        super(MultiColumn, self).__init__()
        self.column_units = column_units
        self.conv_column = conv_column(column_units)
        
            
    def encode(self, inputs):
        outputs = []
        num_cols = len(inputs)
        #print("num_cols", num_cols)
        for idx in range(num_cols):
            x = inputs[idx]
            #print("x", x)
            x1 = self.conv_column(x)
            #print("x1.shape", x1.shape)
            outputs.append(x1)
            #print("outputs.shape", outputs)
        outputs = th.stack(outputs).permute(1, 0, 3, 2)
        outputs = th.squeeze(th.sum(outputs, 1), 1)
        avg_output = outputs / float(num_cols)
        return avg_output



if __name__ == "__main__":
    from backbone import Model
    num_classes = 174
    input_tensor = [th.autograd.Variable(th.rand(1, 3, 72, 84, 84))]
    model = MultiColumn(174, Model, 512)
    output = model(input_tensor)
    print(output.size())
