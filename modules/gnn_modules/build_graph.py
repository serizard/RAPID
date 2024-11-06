import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import itertools
import os
os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
from sklearn.preprocessing import MinMaxScaler
import configparser


class build_graph:

    def __init__(self,config,file_path):

        chunk_size = config['chunk_size']
        num_token = config['num_token']
        
        self.config = config
        def historic_feat(feat):
            next_ = np.append(feat[1:,:,:], np.expand_dims(feat[0,:,:],axis=0),axis=0)#.shape
            past_ = np.append(np.expand_dims(feat[-1,:,:],axis=0), feat[:-1,:,:], axis=0)#.shape

            historic = np.concatenate([feat,next_,past_],axis=2)
            
            return historic
        
        # audio, video feature 
        self.a_feat = np.load(file_path['feature_path']['a' + str(chunk_size)]) # (n_samples, sequence_length=n_nodes, audio_feature_dim)
        self.v_feat = np.load(file_path['feature_path']['v' + str(chunk_size)]) # (n_samples, sequence_length=n_nodes, video_feature_dim)
        
        self.a_feat = historic_feat(self.a_feat) # (n_samples, sequence_length, audio_feature_dim * 3)
        self.v_feat = historic_feat(self.v_feat) # (n_samples, sequence_length, video_feature_dim * 3)

        # adj matrix
        self.u_w = np.load(file_path['graph']['adj'+ str(chunk_size)])[:,:,:num_token] # (n_samples, n_nodes, n_disfluency)
        self.u_w = MinMaxScaler().fit_transform(self.u_w.reshape(self.u_w.shape[0], -1)).reshape(self.u_w.shape) # (n_samples, n_nodes, n_disfluency)
        
        self.k_feat = np.load(file_path['feature_path']['k'])[:num_token,:] # (n_disfluency, embedding_dim)
        
        
        print(self.u_w.shape, self.a_feat.shape, self.v_feat.shape, self.k_feat.shape)
        
    def data_load(self,device):
        graph_list = []
        
        for i, arr in tqdm(enumerate(self.u_w)):
            u_w_mat = torch.tensor(arr.nonzero()).to(f"cuda:{device}")

            #feat
            a_feat = torch.tensor(np.nan_to_num(self.a_feat[i]),dtype=torch.float).to(f"cuda:{device}")
            v_feat = torch.tensor(np.nan_to_num(self.v_feat[i]),dtype=torch.float).to(f"cuda:{device}")
            k_feat = torch.tensor(np.nan_to_num(self.k_feat),dtype=torch.float).to(f"cuda:{device}")       
            e_feat = torch.tensor([arr[u,v] for u,v in zip(u_w_mat[0], u_w_mat[1])],dtype=torch.float).to(f"cuda:{device}")
            # base node - target user
            data_dict = {} #edge
            num_nodes_dict = {} #num_node
            node_feat_dict = {} #node_feat
            edge_feat_dict = {} 
            
            if self.config['rel_type'] == 'v':
                data_dict[('v', 'vk', 'k')] = (u_w_mat[0],u_w_mat[1])
                data_dict[('k', 'kv', 'v')] = (u_w_mat[1],u_w_mat[0])
                num_nodes_dict['v'] = arr.shape[0]
                node_feat_dict['v'] = v_feat
                edge_feat_dict['vk'] = e_feat
                edge_feat_dict['kv'] = e_feat
            
            elif self.config['rel_type'] == 'a':
                data_dict[('a', 'ak', 'k')] = (u_w_mat[0],u_w_mat[1])
                data_dict[('k', 'ka', 'a')] = (u_w_mat[1],u_w_mat[0])
                num_nodes_dict['a'] = arr.shape[0]
                node_feat_dict['a'] = a_feat
                edge_feat_dict['ak'] = e_feat
                edge_feat_dict['ka'] = e_feat
                
            elif self.config['rel_type'] == 'va':

                # 비디오-키워드 관계용 인접행렬
                v_k_mat = torch.tensor(arr[:len(v_feat)].nonzero()).to(f"cuda:{device}")
                data_dict[('v', 'vk', 'k')] = (v_k_mat[0], v_k_mat[1])
                data_dict[('k', 'kv', 'v')] = (v_k_mat[1], v_k_mat[0])
                num_nodes_dict['v'] = len(v_feat)
                edge_feat_dict[('v', 'vk', 'k')] = torch.tensor([arr[u,v] for u,v in zip(v_k_mat[0], v_k_mat[1])], dtype=torch.float).to(f"cuda:{device}")
                edge_feat_dict[('k', 'kv', 'v')] = edge_feat_dict[('v', 'vk', 'k')]                


                # 오디오-키워드 관계용 인접행렬
                a_k_mat = torch.tensor(arr[len(v_feat):].nonzero()).to(f"cuda:{device}")
                data_dict[('a', 'ak', 'k')] = (a_k_mat[0], a_k_mat[1])
                data_dict[('k', 'ka', 'a')] = (a_k_mat[1], a_k_mat[0])
                num_nodes_dict['a'] = len(a_feat)
                edge_feat_dict[('a', 'ak', 'k')] = torch.tensor([arr[u,v] for u,v in zip(a_k_mat[0], a_k_mat[1])], dtype=torch.float).to(f"cuda:{device}")
                edge_feat_dict[('k', 'ka', 'a')] = edge_feat_dict[('a', 'ak', 'k')]
                # data_dict[('v', 'vk', 'k')] = (u_w_mat[0],u_w_mat[1])
                # data_dict[('k', 'kv', 'v')] = (u_w_mat[1],u_w_mat[0])
                # num_nodes_dict['v'] = arr.shape[0]
                # node_feat_dict['v'] = v_feat
                # edge_feat_dict[('v', 'vk', 'k')] = e_feat
                # edge_feat_dict[('k', 'kv', 'v')] = e_feat

                # data_dict[('a', 'ak', 'k')] = (u_w_mat[0],u_w_mat[1])
                # data_dict[('k', 'ka', 'a')] = (u_w_mat[1],u_w_mat[0])
                # num_nodes_dict['a'] = arr.shape[0]
                # node_feat_dict['a'] = a_feat
                # edge_feat_dict[('a', 'ak', 'k')] = e_feat
                # edge_feat_dict[('k', 'ka', 'a')] = e_feat
            
            
            num_nodes_dict['k'] = arr.shape[1]
            node_feat_dict['k'] = k_feat
            
            #graph
            g = dgl.heterograph(data_dict = data_dict,num_nodes_dict = num_nodes_dict).to(f"cuda:{device}")
            g.ndata['features'] = node_feat_dict
            
            if self.config['edge_weight']:
                g.edata['weights'] = edge_feat_dict
                
            graph_list.append(g)

        return graph_list