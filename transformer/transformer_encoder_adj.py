from pickle import NONE
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def multi_head_attention(q, k, v, mask=None):
    # q shape = (B, n_heads, n, key_dim)   : n can be either 1 or N
    # k,v shape = (B, n_heads, N, key_dim)
    # mask.shape = (B, group, N)

    B, n_heads, n, key_dim = q.shape

    # score.shape = (B, n_heads, n, N)
    score = th.matmul(q, k.transpose(2, 3)) / np.sqrt(q.size(-1))

    if mask is not None:
        score += mask[:, None, :, :].expand_as(score)

    shp = [q.size(0), q.size(-2), q.size(1) * q.size(-1)]
    attn = th.matmul(F.softmax(score, dim=3), v).transpose(1, 2)
    return attn.reshape(*shp)


def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(EncoderLayer, self).__init__()

        self.n_heads = n_heads

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim))
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.weight_init_()

    def weight_init_(self):
        nn.init.kaiming_normal_(self.feed_forward[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.feed_forward[2].weight, mode='fan_in', nonlinearity='relu')    
        
    def forward(self, x, mask=None):
        q = make_heads(self.Wq(x), self.n_heads)
        k = make_heads(self.Wk(x), self.n_heads)
        v = make_heads(self.Wv(x), self.n_heads)
        x = x + self.multi_head_combine(multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class TransformerEncoder(nn.Module):
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, obs_num, obs_dim, args):
        super(TransformerEncoder, self).__init__()
        self.hidden_dim = args.hid_size
        self.attend_heads = args.attend_heads
        assert (self.hidden_dim % self.attend_heads) == 0
        self.n_layers = args.n_layers
        self.attend_heads = args.attend_heads
        self.args = args
        self.tp = args.tp
        self.obs_num = obs_num
        self.obs_dim = obs_dim
        self.time_dims = args.time_dims
        self.obs_bus_dim = args.obs_bus_dim
        self.task_projection = nn.Sequential(
                                    nn.Linear(4, self.hidden_dim * 2),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim * 2,self.hidden_dim)
        )

        self.init_projection_layer = nn.Linear(obs_dim, self.hidden_dim)

        self.attn_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=self.hidden_dim * 2,
                         n_heads=self.attend_heads)
            for _ in range(self.n_layers)
        ])

        if args.layernorm:
            self.layernorm = nn.LayerNorm(self.hidden_dim)
        if args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

        self.time_bilstm = nn.LSTM(self.time_dims, self.hidden_dim, batch_first=True, bidirectional=True)
        self.lstm_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.attn_time_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=self.hidden_dim * 2,
                         n_heads=self.attend_heads)
            for _ in range(self.n_layers)
        ])

        self.relu = nn.ReLU()
        self.agent_num = self.args.agent_num
        self.weight_init_()

    def weight_init_(self):
        nn.init.kaiming_normal_(self.task_projection[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.task_projection[2].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.lstm_fc.weight, mode='fan_out', nonlinearity='relu')
        for name, param in self.time_bilstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=0.25)

    def forward(self, obs, obs_time, month, agent_index):
        # obs : (b*n, self.obs_num, self.obs_dim)
        # obs_time : (b*n, tp, self.time_dim)  -> (b*n, self.hidden_dim)
        # agent_index : (b*n, 1)

        b = obs.size(0)
        x = self.init_projection_layer(obs)
        task_emb = self.task_projection(self.month2task(month,b).to(0)).unsqueeze(1)

        #blstm
        time_hidden_state, _ = self.time_bilstm(obs_time)
        time_hidden_state = time_hidden_state[:,-1,:].unsqueeze(1)
        x_time = self.relu(self.lstm_fc(time_hidden_state))
        x = th.cat((x,x_time.expand_as(x)), dim=-1)
        
        for layer in self.attn_layers:
            x = layer(x)
        x = self.relu(self.fc(x))

        x = th.cat((x, task_emb.expand_as(x)), dim=-1)
        for layer in self.attn_time_layers:
            x = layer(x)
        emb = x

        index = agent_index.unsqueeze(dim=-1).expand(-1, 1, self.hidden_dim)
        output = x.gather(1, index).contiguous().squeeze(dim=1)  # (b*n, h)

        return output, None, emb
    
    def month2task(self,month,b):
        agent_num = self.agent_num
        task = th.empty((b,4))
        for i,m in enumerate(month):
            task[i*agent_num:(i+1)*agent_num,:] = self.s_month2season(m).repeat(agent_num,1)
        return task

    def s_month2season(self,month):
        #new season_emb
        if month >= 2 and month <= 4:  # Spring
            task = th.FloatTensor([0,0,0,1])
        elif month >= 5 and month <= 7 :  # Summer
            task = th.FloatTensor([0,0,1,0])
        elif month >= 8 and month <= 10:  # Fall
            task = th.FloatTensor([0,1,0,0])
        else:
            task = th.FloatTensor([1,0,0,0])
        return task
