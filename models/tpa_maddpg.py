from numpy.core.fromnumeric import repeat
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities.util import select_action
from models.model import Model
from critics.mlp_critic import MLPCritic
from transformer.transformer_critic_ex import TransformerCritic
from transformer.transformer_encoder_adj import TransformerEncoder

class TPAMADDPG(Model):
    def __init__(self, args, target_net=None):
        super(TPAMADDPG, self).__init__(args)
        # for constraint
        self.cs_num = self.n_
        self.multiplier = th.nn.Parameter(th.tensor([args.init_lambda for _ in range(self.cs_num)],device=self.device))
        self.upper_bound = args.upper_bound

        self.tp = args.tp
        self.time_dims = args.time_dims

        # for observation transformer encoder
        self.obs_bus_dim = args.obs_bus_dim
        self.obs_bus_num = np.max(args.obs_bus_num)
        self.agent_index_in_obs = args.agent_index_in_obs
        self.obs_mask = th.zeros(self.n_, self.obs_bus_num).to(self.device)
        self.obs_flag = th.ones(self.n_, self.obs_bus_num).to(self.device)
        self.q_index = -1
        self.v_index = 2
        for i in range(self.n_):
            self.obs_mask[i,args.obs_bus_num[i]:] = -np.inf
            self.obs_flag[i,args.obs_bus_num[i]:] = 0.
        self.agent2region = args.agent2region
        self.region_num = np.max(self.agent2region) + 1
        self.adj_mask = th.zeros(self.n_, self.obs_bus_num, self.obs_bus_num).to(self.device)
        self.region_adj = self.args.region_adj
        for i in range(self.n_):
            region_id = self.agent2region[i]
            self.adj_mask[i] = 1 - th.tensor(self.region_adj[region_id])
            self.adj_mask[i] = self.adj_mask[i] - th.diag_embed(th.diag(self.adj_mask[i]))
        self.adj_mask = self.adj_mask.masked_fill(self.adj_mask == 1, -np.inf)
        self.encoder = TransformerEncoder(self.obs_bus_num, self.obs_bus_dim + self.region_num, args)

        # for transformer encoder pretrained, not useful
        if self.args.pretrained is not None:
            param = th.load(self.args.pretrained, map_location='cpu') if not args.cuda else th.load(self.args.pretrained)
            self.encoder.load_state_dict(param)

        self.construct_model()
        self.apply(self.init_weights)
        self.ce = nn.CrossEntropyLoss()

        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)

    def construct_policy_net(self):
        # transformer encoder + mlp head
        if self.args.agent_type == 'transformer':
            from transformer.transformer_policy_head import TransformerAgent
            Agent = TransformerAgent
        else:
            NotImplementedError() 
        if self.args.shared_params:
            self.policy_dicts = nn.ModuleList([ Agent(self.args) ])
        else:
            self.policy_dicts = nn.ModuleList([ Agent(self.args) for _ in range(self.n_) ])

    def construct_value_net(self):
        # (transformer encoder / raw obs) + transformer critic
        if self.args.critic_type == "transformer":
            if self.args.critic_encoder:
                input_shape = self.args.hid_size + self.act_dim
                output_shape = 1
                self.value_dicts = nn.ModuleList( [ TransformerCritic(input_shape, self.args) ] )
            else:
                input_shape = self.obs_dim + self.act_dim
                output_shape = 1
                self.value_dicts = nn.ModuleList( [ TransformerCritic(input_shape, self.args) ] )
        elif self.args.critic_type == "mlp":
            input_shape = ( self.obs_dim + self.act_dim ) * self.n_
            output_shape = 1
            self.value_dicts = nn.ModuleList([MLPCritic(input_shape, output_shape, self.args)])
        else:
            NotImplementedError()

    def construct_auxiliary_net(self):
        if self.args.auxiliary:
            input_shape = self.args.hid_size
            output_shape = 1
            from transformer.transformer_aux_head import TransformerCritic as MLPHead
            self.auxiliary_dicts = nn.ModuleList( [ MLPHead(input_shape, output_shape, self.args, self.args.use_date) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()
        self.construct_auxiliary_net()

    def update_target(self):
        for name, param in self.target_net.policy_dicts.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.policy_dicts.state_dict()[name]
            self.target_net.policy_dicts.state_dict()[name].copy_(update_params)
        for name, param in self.target_net.value_dicts.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.value_dicts.state_dict()[name]
            self.target_net.value_dicts.state_dict()[name].copy_(update_params)
        if self.args.mixer:
            for name, param in self.target_net.mixer.state_dict().items():
                update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.mixer.state_dict()[name]
                self.target_net.mixer.state_dict()[name].copy_(update_params)
        if self.args.encoder:
            for name, param in self.target_net.encoder.state_dict().items():
                update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.encoder.state_dict()[name]
                self.target_net.encoder.state_dict()[name].copy_(update_params)

    def encode(self, raw_obs, raw_obs_time, raw_month):
        batch_size = raw_obs.size(0)
        obs = raw_obs.view(batch_size*self.n_, self.obs_bus_num, self.obs_bus_dim).contiguous() # (b*n, self.obs_bus_num, self.obs_bus_dim)
        obs_time = raw_obs_time.view(batch_size*self.n_, self.tp, self.time_dims).contiguous()
        zone_id = F.one_hot(th.tensor(self.agent2region)).to(self.device).float()
        zone_id = zone_id[None,:,None,:].contiguous().repeat(batch_size, 1, self.obs_bus_num, 1).view(batch_size*self.n_, self.obs_bus_num, self.region_num)
        #mask = self.adj_mask[None,:,:,:].repeat(batch_size,1,1,1).view(batch_size*self.n_, self.obs_bus_num, self.obs_bus_num).contiguous()
        #final_mask = self.obs_mask[None,:,None,:].repeat(batch_size,1,1,1).view(batch_size*self.n_,1,-1).contiguous() # (b*n, 1, obs_bus_num)
        agent_index = th.tensor(self.agent_index_in_obs)[None,:,None].repeat(batch_size, 1, 1).view(batch_size*self.n_, 1).contiguous().to(self.device)
        obs = th.cat((obs,zone_id),dim=-1)
        emb_agent_glimpsed, _, emb = self.encoder(obs, obs_time, raw_month, agent_index)
        flag_mask = self.obs_flag[None,:,:, None].repeat(batch_size,1,1,1).view(batch_size*self.n_,self.obs_bus_num,1).contiguous() # (b*n, num, 1)
        mean_emb = (emb * flag_mask).sum(dim=1) / flag_mask.sum(dim=1)
        return emb_agent_glimpsed, None, mean_emb

    def policy(self, raw_obs, raw_obs_time, raw_month, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        # obs_shape = (b, n, o)
        batch_size = raw_obs.size(0)

        if self.args.shared_params:
            enc_obs, _, _ = self.encode(raw_obs, raw_obs_time, raw_month)
            # _, _, enc_obs = self.encode(raw_obs)
            agent_policy = self.policy_dicts[0]
            means, log_stds, hiddens, proplogits, min_distances = agent_policy(enc_obs, last_hid, raw_month)
            # hiddens = th.stack(hiddens, dim=1)
            means = means.contiguous().view(batch_size, self.n_, -1)
            hiddens = hiddens.contiguous().view(batch_size, self.n_, -1)
            if self.args.gaussian_policy:
                log_stds = log_stds.contiguous().view(batch_size, self.n_, -1)
            else:
                stds = th.ones_like(means).to(self.device) * self.args.fixed_policy_std
                log_stds = th.log(stds)
        else:
            NotImplementedError()
        
        return means, log_stds, hiddens

    def value(self, obs, act): 
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)
        if self.args.critic_encoder:
            if self.args.value_grad:
                emb_agent_glimpsed, _, emb = self.encode(obs)
            else:
                with th.no_grad():
                    emb_agent_glimpsed, _, emb = self.encode(obs)

            if self.args.use_emb == "glimpsed":
                obs = emb_agent_glimpsed
            elif self.args.use_emb == "mean":
                obs = emb
            else:
                NotImplementedError()

            obs_reshape = obs.view(batch_size, self.n_, -1).contiguous()
            act_reshape = act.contiguous()
            inputs = th.cat( (obs_reshape, act_reshape), dim=-1 )   # (b, n, h+a)

        else:
            obs_reshape = obs.contiguous()
            act_reshape = act.contiguous()
            inputs = th.cat( (obs_reshape, act_reshape), dim=-1 )   # (b, n, o+a)
            if self.args.critic_type == "mlp":
                inputs = inputs.view(batch_size, -1)

        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            values, costs = agent_value(inputs)
            values = values.contiguous().unsqueeze(dim=-1).repeat(1, self.n_, 1).view(batch_size, self.n_, 1)
            if self.args.critic_type == "mlp":
                costs = th.zeros_like(values)
            costs = costs.contiguous().view(batch_size, self.n_, 1)
        else:
            NotImplementedError()

        return values, costs

    def get_actions(self, state, state_time, month, status, exploration, actions_avail, target=False, last_hid=None):
        target_policy = self.target_net.policy if self.args.target else self.policy
        if self.args.continuous:
            means, log_stds, hiddens = self.policy(state, state_time, month, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
            if means.size(-1) > 1:
                means_ = means.sum(dim=1, keepdim=True)
                log_stds_ = log_stds.sum(dim=1, keepdim=True)
            else:
                means_ = means
                log_stds_ = log_stds
            actions, log_prob_a = select_action(self.args, means_, status=status, exploration=exploration, info={'log_std': log_stds_})
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out, hiddens

    def get_loss(self, batch): 
        batch_size = len(batch.state)
        state,state_time,month, actions, old_log_prob_a, old_values, old_next_values, rewards, cost, next_state,next_state_time,next_month, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state,state_time,month, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=last_hids)
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state,next_state_time,next_month,status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state,next_state_time,next_month,status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
        compose = self.value(state, actions_pol)
        values_pol, costs_pol = compose[0].contiguous().view(-1, self.n_), compose[1].contiguous().view(-1, self.n_)
        compose = self.value(state, actions)
        values, costs = compose[0].contiguous().view(-1, self.n_), compose[1].contiguous().view(-1, self.n_)
        compose = self.target_net.value(next_state, next_actions.detach())
        next_values, next_costs = compose[0].contiguous().view(-1, self.n_), compose[1].contiguous().view(-1, self.n_)
        returns, cost_returns = th.zeros((batch_size, self.n_), dtype=th.float).to(self.device), th.zeros((batch_size, self.n_), dtype=th.float).to(self.device)
        assert values_pol.size() == next_values.size()
        assert returns.size() == values.size()
        done = done.to(self.device)
        returns = rewards - (self.multiplier.detach() * cost).sum(dim=-1, keepdim=True)  + self.args.gamma * (1 - done) * next_values.detach()
        cost_returns = cost + self.args.cost_gamma * (1-done) * next_costs.detach()
        deltas, cost_deltas = returns - values, cost_returns - costs
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        policy_loss = - advantages
        policy_loss = policy_loss.mean()
        value_loss = deltas.pow(2).mean()
        # if self.args.cost_loss:
        #     value_loss += cost_deltas.pow(2).mean()
        lambda_loss = - ((cost_returns.detach() - self.upper_bound) * self.multiplier).mean(dim=0).sum()
        return policy_loss, value_loss, action_out, lambda_loss

    def get_prop_loss(self, batch):
        batch_size = len(batch.state)
        state,state_time,month, actions, old_log_prob_a, old_values, old_next_values, rewards, cost, next_state,next_state_time,next_month, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        obs = state.view(batch_size, self.n_, self.obs_bus_num, self.obs_bus_dim).contiguous() # (b*n, self.obs_bus_num, self.obs_bus_dim)
        enc_obs, _, _ = self.encode(obs, state_time, month)
        # _, _, enc_obs = self.encode(raw_obs)
        agent_policy = self.policy_dicts[0]
        means, log_stds, hiddens, proplogits, min_distances = agent_policy(enc_obs, last_hids, month)
        label_month = agent_policy.agentsmonth(month).to(self.device)
        label_month = label_month.long()
        loss = self.ce(proplogits, label_month)

        batch_loss = 0
        for i in range(batch_size):
            #cluster loss
            prototypes_of_correct_class = th.t(agent_policy.prototype_class_identity[:, label_month[i*agent_policy.agent_num]].bool()).to(self.device)
            cluster_cost = th.mean(th.min(min_distances[i, prototypes_of_correct_class].reshape(-1, agent_policy.num_prototypes_per_class), dim=1)[0])

            #seperation loss
            separation_cost = -th.mean(th.min(min_distances[i, ~prototypes_of_correct_class].reshape(-1, (agent_policy.output_dim-1)*agent_policy.num_prototypes_per_class), dim=1)[0])

            #diversity loss
            ld = 0
            for k in range(agent_policy.output_dim):
                p = agent_policy.prototype_vectors[k*agent_policy.num_prototypes_per_class: (k+1)*agent_policy.num_prototypes_per_class]
                p = F.normalize(p, p=2, dim=1)
                matrix1 = th.mm(p, th.t(p)) - th.eye(p.shape[0]).to(self.device) - 0.3
                matrix2 = th.zeros(matrix1.shape).to(self.device)
                ld += th.sum(th.where(matrix1 > 0, matrix1, matrix2))

            batch_loss = batch_loss + 0.10*cluster_cost + 0.05*separation_cost  + 0.001*ld
        loss = loss + batch_loss / batch_size  
        return loss * 0.8

    def reset_multiplier(self):
        for i in range(self.cs_num):
            if self.multiplier[i] < 0:
                with th.no_grad():
                    self.multiplier[i] = 0.

    def get_auxiliary_loss(self, batch):
        batch_size = len(batch.state)
        state,state_time,month, actions, old_log_prob_a, old_values, old_next_values, rewards, cost, next_state,next_state_time,next_month, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)

        obs = state.view(batch_size, self.n_, self.obs_bus_num, self.obs_bus_dim).contiguous() # (b*n, self.obs_bus_num, self.obs_bus_dim)
        with th.no_grad():
            label = self._cal_out_of_control(obs.view(batch_size*self.n_, self.obs_bus_num, self.obs_bus_dim))
        enc_obs, _, _ = self.encode(obs, state_time, month)
        pred, _ = self.auxiliary_dicts[0](enc_obs, None)
        loss = nn.MSELoss()(pred, label)
        return loss

    def _cal_out_of_control(self, obs):
        batch_size = obs.shape[0] // self.n_
        mask = self.obs_flag[None, : ,:].repeat(batch_size, 1, 1).view(batch_size*self.n_, -1)
        v = obs[:,:,self.v_index]
        out_of_control = th.logical_or(v<0.95,v>1.05).float()
        percentage_out_of_control = (out_of_control * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
        return percentage_out_of_control
