import matplotlib
matplotlib.use('Agg')

from process_input import FrameExtractor, Transcriptor, GestureExtractor, AudioExtractor, chunk_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from modules.gnn_modules.build_graph import *
from modules.gnn_modules.graphconv import SAGEConv,HeteroGraphConv
from modules.gnn_modules.self_att import Attention
from modules.mult_modules.mulT import MULTModel
import yaml
import json



class RAPIDModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.seed = self.config['random_seed']
        self.gpu = self.config['gpu'] if torch.cuda.is_available() else 'cpu'
        
        self.chunk_size = self.config['chunk_size']
        
        self.num_labels = self.config['num_labels']
        self.dropout = nn.Dropout(self.config['dropout'])
        self.loss_type = self.config['loss']
        ## col
        self.y_col = self.config['y_col']
        self.modal = self.config['modal']
        self.t, self.a, self.v  = self.modal.split('_') 
        
        self.att = self.config['att']
        self.tonly,self.aonly,self.vonly= self.att.split('_')
        self.output_dim = self.config['num_labels']
        self.txt_col = 'asr_body_pre'
        self.token_num = self.config['num_token']

        pretrained = 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained, do_lower_case=True)
        self.model = RobertaModel.from_pretrained(pretrained)
        self.keywords = json.load(open(self.config['keyword_path']))[:self.token_num]
        self._update_keywords()


        
        self.a_hidden = int(self.config['hidden'][self.a])
        self.t_hidden = int(self.config['hidden'][self.t])
        self.v_hidden = int(self.config['hidden'][self.v])
        
        ## dgl
        # Inter Relation define
        self.gnn_size = 768
        self.agg_type = self.config['agg_type']
        self.hetero_type = self.config['hetero_type']
        if self.config['rel_type'] == 'v':
            rel_names = {'vk': (int(self.v_hidden)*3, int(self.t_hidden)),
                         'kv': (int(self.t_hidden), int(self.v_hidden)*3)}
        elif self.config['rel_type'] == 'a':
            rel_names = {'ak': (int(self.a_hidden)*3, int(self.t_hidden)),
                         'ka': (int(self.t_hidden), int(self.a_hidden)*3),
                         'vk': (int(self.v_hidden)*3, int(self.t_hidden)),
                         'kv': (int(self.t_hidden), int(self.v_hidden)*3)}
        elif self.config['rel_type'] == 'va':
            rel_names = {'ak': (int(self.a_hidden)*3, int(self.t_hidden)),
                         'ka': (int(self.t_hidden), int(self.a_hidden)*3),
                         'vk': (int(self.v_hidden)*3, int(self.t_hidden)),
                         'kv': (int(self.t_hidden), int(self.v_hidden)*3)}
        
        # Model init
        mod_dict = {rel : SAGEConv((src_dim, dst_dim), self.gnn_size,
                           aggregator_type = self.agg_type)\
                          for rel,(src_dim, dst_dim) in rel_names.items()}

        self.conv = HeteroGraphConv(mod_dict, aggregate=self.hetero_type)
        
        self.v_lstm = nn.LSTM(input_size=self.v_hidden,
                             hidden_size=self.v_hidden,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True)
        
        self.a_lstm = nn.LSTM(input_size=self.a_hidden,
                             hidden_size=self.a_hidden,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True)
        
        self.v_atten = Attention(self.config['gpu'],int(self.v_hidden *2), batch_first=True)  # 2 is bidrectional
        self.a_atten = Attention(self.config['gpu'],int(self.a_hidden *2), batch_first=True)  # 2 is bidrectional
        self.dropout = nn.Dropout(self.config['dropout'])

        # model & hidden
        if self.config['graphuse']:
            self.MULTModel = MULTModel(self.config, use_origin=True)
        else:
            self.MULTModel = MULTModel(self.config, use_origin=False)
            
        self.fc1 = nn.Linear(self.t_hidden, int(self.t_hidden/2))
        self.fc2 = nn.Linear(int(self.t_hidden/2), self.output_dim)

        self.frame_extractor = FrameExtractor()
        self.transcriptor = Transcriptor(model_size='turbo', device=f'cuda:{self.gpu}' if self.gpu != 'cpu' else 'cpu')
        self.gesture_extractor = GestureExtractor()
        self.audio_extractor = AudioExtractor()

        self._load_model()


    def _update_keywords(self):
        self.tokenizer.add_tokens(self.keywords, special_tokens=False)
        self.model.resize_token_embeddings(len(self.tokenizer))
        keyword_tokens = [self.tokenizer.encode(keyword, add_special_tokens=False)[0] for keyword in self.keywords]
        self.keyword_token = torch.tensor(keyword_tokens)
        self.key_embed = self.model(input_ids=self.keyword_token.unsqueeze(0))[0][0]


    @torch.inference_mode()
    def forward_inference(self, txt, aud, vid, **kwargs):
        """
        txt: (# of chunks, seq_len)
        aud: (# of chunks, seq_len, audio_feature_dim)
        vid: (# of chunks, seq_len, video_feature_dim)
        """
        # 0. Input
        num_chunk, seq_ = txt.shape
        v_h,_ = self.v_lstm(vid)
        v_h, v_att_score = self.v_atten(v_h)
        
        a_h,_ = self.a_lstm(aud)
        a_h, a_att_score = self.a_atten(a_h)
        

        def historic_feat(feat):
            next_ = torch.cat([feat[1:,:,:], feat[0,:,:].unsqueeze(0)],axis=0)
            past_ = torch.cat([feat[-1,:,:].unsqueeze(0),feat[:-1,:,:]],axis=0)

            feat =torch.cat([feat,next_,past_],axis=2)
            return feat
        
        vid_h = historic_feat(v_h)
        aud_h = historic_feat(a_h)
        
        
        # 1. Speech-gesture graph encoder
        if self.config['graphuse']:
            bg = dgl.batch(self.g_list)#.to(device)    
            bg.ndata['features']['a'] = a_h.reshape(-1,int(self.a_hidden*2))
            bg.ndata['features']['v'] = v_h.reshape(-1,int(self.v_hidden*2))
            bg.ndata['features']['k'] = self.key_embed.repeat(num_chunk,1,1).reshape(-1,self.t_hidden)
            
            if self.config['edge_weight']:
                mod_args = bg.edata['weights'] #{'edge_weight': bg.edata['weights']}
            else:
                mod_args = None
            gnn_h = self.conv(g=bg, inputs=bg.ndata['features'], mod_args=mod_args)
        
        
        # 2. Gesture-aware embedding update
        if self.config['update']:
            key_embed = gnn_h['k'].reshape(num_chunk,-1,self.gnn_size) # 32 x 30 x 20

            with torch.no_grad():
                new_embedding = self.model.embeddings.word_embeddings.weight.data.clone()
                new_embedding[self.keyword_token] = key_embed[0] 
                self.model.embeddings.word_embeddings.weight.set_(new_embedding)

        if self.config['graphuse']:
            if self.config['rel_type'] == 'v':
                aud_h = gnn_h['v'].reshape(num_chunk,-1,self.gnn_size) # 32 x 30 x 20 [batch*node, hidden_dim]
                vid_h = aud_h
            elif self.config['rel_type'] == 'a':
                vid_h = gnn_h['a'].reshape(num_chunk,-1,self.gnn_size) # 32 x 30 x 20
                aud_h = vid_h 
                
            elif self.config['rel_type'] == 'va':
                aud_h = gnn_h['a'].reshape(num_chunk,-1,self.gnn_size) # 32 x 30 x 20 [batch*node, hidden_dim]
                vid_h = gnn_h['v'].reshape(num_chunk,-1,self.gnn_size) # 32 x 30 x 20
        txt_h = self.model(input_ids =txt)
        
        # 3. Multimodal Fusion Encoder
        with torch.cuda.amp.autocast():
            relation_h,_, att_vl =self.MULTModel(txt_h[0], aud_h, vid_h) # 32 x 20
        last_h_l = txt_h[1]+relation_h
        
        # 4. Aphasia Type Detection
        logits = self.fc2(F.relu(self.fc1(last_h_l)))

        return logits,att_vl,v_att_score

    def _load_model(self):
        state_dict = torch.load(self.config['checkpoint_path'], map_location='cpu')['state_dict']
        self.load_state_dict(state_dict, strict=False)
        self.eval()
    
    
    def extract_features(self, video_path):
        transcription, audio_path = self.transcriptor.transcribe(video_path)
        print(f'Transcription done')
        self.frame_extractor.extract_frames(video_path)
        print(f'Frames extracted')
        text_bind = self.frame_extractor.align_frames(transcription)
        print(f'Frames aligned')
        opensmile_bind = self.audio_extractor.extract_opensmile_bind(text_bind, audio_path)
        print(f'Opensmile features extracted')
        pose_bind = self.gesture_extractor.extract_pose_bind(text_bind)
        print(f'Pose features extracted')

        text_bind[self.txt_col] = text_bind[self.txt_col].map(lambda x: self.tokenizer.encode(
            str(x),
            padding = 'max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))

        return text_bind, opensmile_bind, pose_bind


    def predict(self, video_path):
        text_bind, opensmile_bind, pose_bind = self.extract_features(video_path)
        audio_feats, video_feats, audio_feats_with_additional_tokens, video_feats_with_additional_tokens, adj_matrices = chunk_dataset(text_bind, opensmile_bind, pose_bind)

        if self.config['graphuse']:
            self.g_list =build_graph(self.config, 
                                    audio_feats,
                                    video_feats, 
                                    adj_matrices).data_load(self.gpu)

        audio_feats_with_additional_tokens = np.nan_to_num(audio_feats_with_additional_tokens).astype('float64')
        video_feats_with_additional_tokens = np.nan_to_num(video_feats_with_additional_tokens).astype('float64')

        logits, att_vl, v_att_score = self.forward_inference(audio_feats_with_additional_tokens, video_feats_with_additional_tokens, text_bind[self.txt_col])
        
        mean_logits = logits.mean(0)
        pred = mean_logits.argmax().item()

        return pred, att_vl, v_att_score
    
# For debugging
if __name__ == '__main__':
    config = configparser.ConfigParser()
    with open('D:/aphasia/MMATD/src/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print('Starting...')
    model = RAPIDModel(config)
    print('Model loaded...')
    pred, _, _ = model.predict('D:/aphasia/MMATD/src/example.mp4')
    print(f'Prediction: {pred}')
    print('Prediction done...')