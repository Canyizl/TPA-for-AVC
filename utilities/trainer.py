import numpy as np
import torch as th
import torch.nn as nn
from torch import optim
from utilities.util import multinomial_entropy, get_grad_norm, normal_entropy
from utilities.replay_buffer import TransReplayBuffer, EpisodeReplayBuffer
#import wandb


class PGTrainer(object):
    def __init__(self, args, model, env, logger, constraint_model=None):
        self.best_CR = 0
        self.args = args
        self.tpa = args.tpa
        self.device = th.device(
            "cuda" if th.cuda.is_available() and self.args.cuda else "cpu")
        self.logger = logger
        self.episodic = self.args.episodic
        self.constraint_model = constraint_model
        if self.args.target:
            target_net = model(self.args).to(self.device)
            if constraint_model is not None:
                self.behaviour_net = model(
                    self.args, target_net, constraint_model=constraint_model).to(self.device)
            else:
                self.behaviour_net = model(
                    self.args, target_net).to(self.device)
        else:
            if constraint_model is not None:
                self.behaviour_net = model(
                    self.args, constraint_model=constraint_model).to(self.device)
            else:
                self.behaviour_net = model(self.args).to(self.device)
        if self.args.replay:
            if not self.episodic:
                self.replay_buffer = TransReplayBuffer(
                    int(self.args.replay_buffer_size))
            else:
                self.replay_buffer = EpisodeReplayBuffer(
                    int(self.args.replay_buffer_size))
        self.env = env
        # policy optim
        params = []
        params.append(
            {'params': self.behaviour_net.policy_dicts.parameters(), 'lr': args.policy_lrate})
        if self.args.encoder:
            params.append(
                {'params': self.behaviour_net.encoder.parameters(), 'lr': args.encoder_lrate})
        self.policy_optimizer = optim.RMSprop(params, alpha=0.99, eps=1e-5)
        # value optim
        params = []
        if hasattr(self.behaviour_net.value_dicts[0], 'cost_head'):
            cost_head_params = list(
                map(id, self.behaviour_net.value_dicts[0].cost_head.parameters()))
            other_params = filter(lambda p: id(
                p) not in cost_head_params, self.behaviour_net.value_dicts.parameters())
            params.append({'params': other_params, 'lr': args.value_lrate})
            params.append({'params': self.behaviour_net.value_dicts[0].cost_head.parameters(
            ), 'lr': args.cost_head_lrate})
        else:
            params.append(
                {'params': self.behaviour_net.value_dicts.parameters(), 'lr': args.value_lrate})
        if self.args.encoder:
            params.append(
                {'params': self.behaviour_net.encoder.parameters(), 'lr': args.encoder_lrate})
        self.value_optimizer = optim.RMSprop(params, alpha=0.99, eps=1e-5)
        # mixer optim
        if self.args.mixer:
            self.mixer_optimizer = optim.RMSprop(
                self.behaviour_net.mixer.parameters(), lr=args.mixer_lrate, alpha=0.99, eps=1e-5)
        if self.args.multiplier:
            params = []
            params.append(
                {'params': self.behaviour_net.multiplier, 'lr': args.lambda_lrate})
            if hasattr(self.behaviour_net, "cost_dicts"):
                if self.args.encoder:
                    params.append(
                        {'params': self.behaviour_net.encoder.parameters(), 'lr': args.encoder_lrate})
                params.append(
                    {'params': self.behaviour_net.cost_dicts.parameters(), 'lr': args.value_lrate})
            self.lambda_optimizer = optim.RMSprop(params, alpha=0.99, eps=1e-5)
        if self.args.auxiliary:
            assert self.args.encoder == True
            self.auxiliary_optimizer = optim.RMSprop([{'params': self.behaviour_net.auxiliary_dicts.parameters(), 'lr': args.auxiliary_lrate}, {
                                                     'params': self.behaviour_net.encoder.parameters(), 'lr': args.auxiliary_lrate}], alpha=0.99, eps=1e-5)
        self.init_action = th.zeros(
            1, self.args.agent_num, self.args.action_dim).to(self.device)
        self.steps = 0
        self.episodes = 0
        self.entr = self.args.entr


        # self.model_reload()

    def get_loss(self, batch):
        policy_loss, value_loss, logits, lambda_loss = self.behaviour_net.get_loss(
            batch)
        return policy_loss, value_loss, logits, lambda_loss

    def policy_compute_grad(self, stat, loss, retain_graph, trans):
        if self.entr > 0:
            if self.args.continuous:
                policy_loss, means, log_stds = loss
                entropy = normal_entropy(means, log_stds.exp())
            else:
                policy_loss, logits = loss
                entropy = multinomial_entropy(logits)
            policy_loss -= self.entr * entropy
            stat['mean_train_entropy'] = entropy.item()
        if self.tpa:
            prop_loss = self.behaviour_net.get_prop_loss(trans)
            policy_loss = policy_loss + prop_loss
            policy_loss.backward(retain_graph=retain_graph)
        else:
            policy_loss.backward(retain_graph=retain_graph)
    

    def value_compute_grad(self, value_loss, retain_graph):
        value_loss.backward(retain_graph=retain_graph)

    def grad_clip(self, params):
        for param in params:
            param.grad.data.clamp_(-self.args.grad_clip_eps,
                                   self.args.grad_clip_eps)

    def policy_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.policy_transition_process(stat, batch)

    def value_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.value_transition_process(stat, batch)

    def mixer_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.mixer_transition_process(stat, batch)

    def lambda_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.lambda_transition_process(stat, batch)

    def auxiliary_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.auxiliary_transition_process(stat, batch)

    def policy_transition_process(self, stat, trans):
        if self.args.continuous:
            policy_loss, _, logits, _ = self.get_loss(trans)
            means, log_stds = logits
        else:
            policy_loss, _, logits, _ = self.get_loss(trans)
        self.policy_optimizer.zero_grad()
        if self.args.continuous:
            self.policy_compute_grad(
                stat, (policy_loss, means, log_stds), False, trans)
        else:
            self.policy_compute_grad(stat, (policy_loss, logits), False, trans)
        param = self.policy_optimizer.param_groups[0]['params']
        policy_grad_norms = get_grad_norm(self.args, param)
        self.policy_optimizer.step()


        # np.array(policy_grad_norms).mean()
        stat['mean_train_policy_grad_norm'] = policy_grad_norms.item()
        stat['mean_train_policy_loss'] = policy_loss.clone().mean().item()

    def value_transition_process(self, stat, trans):
        _, value_loss, _, _ = self.get_loss(trans)
        # if self.args.predict_loss:
        #     value_loss, pred_loss = value_loss
        #     stat['mean_train_pred_loss'] = pred_loss.clone().mean().item()
        #     value_loss = value_loss + pred_loss
        self.value_optimizer.zero_grad()
        self.value_compute_grad(value_loss, False)
        param = self.value_optimizer.param_groups[0]['params']
        value_grad_norms = get_grad_norm(self.args, param)
        self.value_optimizer.step()
        stat['mean_train_value_grad_norm'] = value_grad_norms.item()
        stat['mean_train_value_loss'] = value_loss.clone().mean().item()

    def mixer_transition_process(self, stat, trans):
        _, value_loss, _, _ = self.get_loss(trans)
        self.mixer_optimizer.zero_grad()
        self.value_compute_grad(value_loss, False)
        param = self.mixer_optimizer.param_groups[0]['params']
        mixer_grad_norms = get_grad_norm(self.args, param)
        self.mixer_optimizer.step()
        stat['mean_train_mixer_grad_norm'] = mixer_grad_norms.item()
        stat['mean_train_mixer_loss'] = value_loss.clone().mean().item()

    def lambda_transition_process(self, stat, trans):
        _, _, _, lambda_loss = self.get_loss(trans)
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        if th.any(self.behaviour_net.multiplier < 0):
            self.behaviour_net.reset_multiplier()
        stat['mean_train_lambda'] = self.behaviour_net.multiplier.detach().mean().item()

    def auxiliary_transition_process(self, stat, trans):
        auxiliary_loss = self.behaviour_net.get_auxiliary_loss(trans)
        self.auxiliary_optimizer.zero_grad()
        auxiliary_loss.backward()
        param = self.auxiliary_optimizer.param_groups[0]['params']
        auxiliary_grad_norms = get_grad_norm(self.args, param)
        self.auxiliary_optimizer.step()
        # np.array(policy_grad_norms).mean()
        stat['mean_train_auxiliary_grad_norm'] = auxiliary_grad_norms.item()
        stat['mean_train_auxiliary_loss'] = auxiliary_loss.clone().mean().item()

    def run(self, stat, episode, save_path, log_name):
        # self.behaviour_net.train()
        # self.behaviour_net.evaluation(stat, self, 'All')
        
        self.behaviour_net.train_process(stat, self)
        if (episode % self.args.eval_freq == self.args.eval_freq-1) or (episode == 0):
            # self.behaviour_net.eval()
            # self.behaviour_net.evaluation(stat, self, 'June')
            self.behaviour_net.evaluation(stat, self, 'All')
            CR = self.get_CR(stat)
            if CR > self.best_CR:
                self.best_CR = CR
                print("Best Valid CR now is:", CR)


    def logging(self, stat, use_wandb=False):
        for k, v in stat.items():
            self.logger.add_scalar('data/' + k, v, self.episodes)
        if use_wandb:
            wandb.log(stat, self.episodes)

    def print_info(self, stat):
        string = [f'\nEpisode: {self.episodes}']
        for k, v in stat.items():
            string.append(k + f': {v:2.4f}')
        string = "\n".join(string)
        print(string)

    def get_CR(self,stat):
        key = "All_mean_test_totally_controllable_ratio"
        result = 0
        for k, v in stat.items():
            if k == key:
                result = v
        return result
    