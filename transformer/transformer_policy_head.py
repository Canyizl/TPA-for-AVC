import torch.nn as nn
import torch

# MLP and GRU after transformer encoder
class TransformerAgent(nn.Module):
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, args):
        super(TransformerAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.hid_size, args.hid_size)
        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.hid_size)
        self.rnn = nn.GRUCell(args.hid_size, args.hid_size)
        self.final_output_layer = nn.Linear(args.hid_size, args.action_dim)
        if self.args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif self.args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

        self.agent_num = self.args.agent_num
        self.epsilon = 1e-4
        self.Softmax = nn.Softmax(dim=-1)
        self.output_dim = 4 #season
        self.num_prototypes_per_class = 6 #jieqi
        self.prototype_shape = (self.output_dim * self.num_prototypes_per_class, args.hid_size)

        self.jieqi_info = torch.FloatTensor(args.jieqi_info).to(0)
        self.jieqi_lstm = nn.LSTM(3,args.hid_size,batch_first=True,bidirectional=True)
        self.jieqi_lstm_fc = nn.Linear(args.hid_size*2, args.hid_size)
        self.relu = nn.ReLU()
        self.proto_proj = nn.Linear(args.hid_size,args.hid_size,bias=False)

        self.num_prototypes = self.prototype_shape[0]
        self.last_layer = nn.Linear(self.num_prototypes, self.output_dim,
                                    bias=False)  # do not use bias
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.output_dim)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)
        self.conv_global = nn.Linear(args.hid_size*2, args.hid_size)
        self.bn_global = nn.BatchNorm1d(args.hid_size)
    
    def get_jieqi_prototype(self):
        time_hidden_state, _ = self.jieqi_lstm(self.jieqi_info)
        time_hidden_state = time_hidden_state[:,-1,:]
        prototype_vectors = self.relu(self.jieqi_lstm_fc(time_hidden_state))
        self.prototype_vectors = self.proto_proj(prototype_vectors)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
    
    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors))
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True))
        similarity = torch.log((distance+1) / (distance + self.epsilon))
        return similarity, distance

    def prototype_subgraph_distances(self, x, prototype):
        b = prototype.size(0)
        all_sim = torch.empty(b,6,1)
        x = x.reshape(b,-1,64)
        for j in range(6):
            p = prototype[:,j].unsqueeze(1)
            distance = torch.norm(x - p, p=2, dim=-1, keepdim=True) ** 2
            similarity = torch.mean(torch.log((distance + 1) / (distance + self.epsilon)), dim=1)
            all_sim[:,j,:] = similarity
        return all_sim, None
    
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.agent_num, self.args.hid_size).zero_()
        # return th.zeros(1, self.args.agent_num, self.hidden_dim).cuda()
  
    def forward(self, inputs, hidden_state, month):
        self.get_jieqi_prototype()
        seasons = torch.tensor(self.month2season(month)).to(0)
        prototype_activations, min_distances = self.prototype_distances(inputs)
        logits = self.last_layer(prototype_activations)
        select_proto = self.prototype_vectors.reshape(4,6,64).to(0)
        sub_prototype = torch.index_select(select_proto,dim=0,index=seasons.long()).to(0)
        similarity, _ = self.prototype_subgraph_distances(inputs,sub_prototype)
        chosen_idx = torch.argmax(similarity, dim=1, keepdim=True).to(0)
        chosen_idx = chosen_idx.expand(-1, 1, 64)
        chosen_proto = torch.gather(sub_prototype, index=chosen_idx.long(), dim=1)
        chosen_proto = chosen_proto.repeat(1,self.agent_num,1).reshape(-1,64)
        x_glb = torch.cat((inputs, chosen_proto), dim=-1)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x = self.fc1(x_glb)
        if self.args.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)
        h_in = hidden_state.reshape(-1, self.args.hid_size)
        h = self.rnn(x, h_in)
        a = self.final_output_layer(h)
        return a, None, h, logits, min_distances
    
    def month2season(self,month):
        task = list()
        for m in month:
            task.append(self.s_month2season(m))
        return task
    
    def agentsmonth(self,month):
        agent_num = self.agent_num
        task = torch.empty(agent_num*month.size(0))
        for i,m in enumerate(month):
            task[i*agent_num:(i+1)*agent_num] = torch.tensor(self.s_month2season(m)).repeat(agent_num)
        return task

    def s_month2season(self,month):
        if month >= 2 and month <= 4 :  # Spring
            task = 0
        elif month >= 5 and month <= 7:  # Summer
            task = 1
        elif month >= 8 and month <= 10:  #Fall
            task = 2
        else:
            task = 3
        return task