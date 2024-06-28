import torch as th
from utilities.util import translate_action, prep_obs
import numpy as np
import time
from datetime import datetime

class PGTester(object):
    def __init__(self, args, behaviour_net, env, render=False):
        self.env = env
        self.behaviour_net = behaviour_net.cuda(
        ).eval() if args.cuda else behaviour_net.eval()
        self.args = args
        self.device = th.device(
            "cuda" if th.cuda.is_available() and self.args.cuda else "cpu")
        self.n_ = self.args.agent_num
        self.obs_dim = self.args.obs_size
        self.act_dim = self.args.action_dim
        self.render = render
        self.tp = self.args.tp
        self.time_dims = self.args.time_dims

    def run(self, day, hour, quarter):
        # reset env
        comb_state, global_state, epi_year = self.env.manual_reset(day, hour, quarter)

        state = comb_state["agents_obs"]
        state_time = comb_state["time_obs"]
        month = comb_state["month"]

        # init hidden states
        last_hid = self.behaviour_net.policy_dicts[0].init_hidden()

        record = {"pv_active": [],
                  "pv_reactive": [],
                  "bus_active": [],
                  "bus_reactive": [],
                  "bus_voltage": [],
                  "line_loss": []
                  }

        record["pv_active"].append(self.env._get_sgen_active())
        record["pv_reactive"].append(self.env._get_sgen_reactive())
        record["bus_active"].append(self.env._get_res_bus_active())
        record["bus_reactive"].append(self.env._get_res_bus_reactive())
        record["bus_voltage"].append(self.env._get_res_bus_v())
        record["line_loss"].append(self.env._get_res_line_loss())

        for t in range(self.args.max_steps):
            if self.render:
                self.env.render()
                time.sleep(0.01)
            state_ = prep_obs(state).contiguous().view(
                1, self.n_, self.obs_dim).to(self.device)
            state_time_ = prep_obs(state_time).to(self.device).contiguous().view(
                self.n_ ,self.tp, self.time_dims)
            with th.no_grad():
                action, _, _, _, hid = self.behaviour_net.get_actions(state_, state_time_, month, status='test', exploration=False, actions_avail=th.tensor(self.env.get_avail_actions()), target=False, last_hid=last_hid)
            _, actual = translate_action(self.args, action, self.env)
            reward, done, info = self.env.step(actual, add_noise=False)
            done_ = done or t == self.args.max_steps-1
            record["pv_active"].append(self.env._get_sgen_active())
            record["pv_reactive"].append(self.env._get_sgen_reactive())
            record["bus_active"].append(self.env._get_res_bus_active())
            record["bus_reactive"].append(self.env._get_res_bus_reactive())
            record["bus_voltage"].append(self.env._get_res_bus_v())
            record["line_loss"].append(self.env._get_res_line_loss())
            # set the next state
            next_comb_state = self.env.get_obs()
            next_state = next_comb_state["agents_obs"]
            next_state_time = next_comb_state["time_obs"]
            next_month = next_comb_state["month"]
            state = next_state
            state_time = next_state_time
            month = next_month
            # set the next last_hid
            last_hid = hid
            if done_:
                break
        return record

    def batch_run(self, num_epsiodes=100):
        test_results = {}
        for epi in range(num_epsiodes):
            # reset env
            state, global_state = self.env.reset()

            # init hidden states
            last_hid = self.behaviour_net.policy_dicts[0].init_hidden()

            for t in range(self.args.max_steps):
                if self.render:
                    self.env.render()
                    time.sleep(0.01)
                state_ = prep_obs(state).contiguous().view(
                    1, self.n_, self.obs_dim).to(self.device)
                action, _, _, _, hid = self.behaviour_net.get_actions(state_, status='test', exploration=False, actions_avail=th.tensor(
                    self.env.get_avail_actions()), target=False, last_hid=last_hid)
                _, actual = translate_action(self.args, action, self.env)
                reward, done, info = self.env.step(actual, add_noise=False)
                done_ = done or t == self.args.max_steps-1
                next_state = self.env.get_obs()
                for k, v in info.items():
                    if 'mean_test_'+k not in test_results.keys():
                        test_results['mean_test_'+k] = [v]
                    else:
                        test_results['mean_test_'+k].append(v)
                # set the next state
                state = next_state
                # set the next last_hid
                last_hid = hid
                if done_:
                    break
            print(f"This is the test episode: {epi}")
        for k, v in test_results.items():
            test_results[k] = (np.mean(v), 2 * np.std(v))
        self.print_info(test_results)
        return test_results

    def test_data_run(self, test_data):
        test_results = {}
        for month in range(1, 13):
            test_results[month] = {}
        num_epsiodes = len(test_data)
        for epi in range(num_epsiodes):
            # reset env
            comb_state, global_state, epi_year = self.env.manual_reset(
                test_data.iloc[epi]['day_id'], 23, 2)
                #test_data.iloc[epi]['day_id'], self.env._select_start_hour(), self.env._select_start_interval())

            state = comb_state["agents_obs"]
            state_time = comb_state["time_obs"]
            month = comb_state["month"]
            month_idx = test_data.iloc[epi]['month']

            # init hidden states
            last_hid = self.behaviour_net.policy_dicts[0].init_hidden()
            result = {}
            for t in range(self.args.max_steps):
                if self.render:
                    self.env.render()
                    time.sleep(0.01)
                state_ = prep_obs(state).contiguous().view(
                    1, self.n_, self.obs_dim).to(self.device)
                state_time_ = prep_obs(state_time).to(self.device).contiguous().view(
                    self.n_ ,self.tp, self.time_dims)
                with th.no_grad():
                    action, _, _, _, hid = self.behaviour_net.get_actions(state_, state_time_, month, status='test', exploration=False, actions_avail=th.tensor(
                        self.env.get_avail_actions()), target=False, last_hid=last_hid)
                _, actual = translate_action(self.args, action, self.env)
                reward, done, info = self.env.step(actual, add_noise=False)
                done_ = done or t == self.args.max_steps-1
                next_comb_state = self.env.get_obs()
                next_state = next_comb_state["agents_obs"]
                next_state_time = next_comb_state["time_obs"]
                next_month = next_comb_state["month"]
                for k, v in info.items():
                    if 'mean_test_'+k not in result.keys():
                        result['mean_test_'+k] = [v]
                    else:
                        result['mean_test_'+k].append(v)
                # set the next state
                state = next_state
                state_time = next_state_time
                month = next_month
                # set the next last_hid
                last_hid = hid
                if done_:
                    break
            print(f"This is the test episode: {epi}")
            for k, v in result.items():
                if k not in test_results[month_idx].keys():
                    test_results[month_idx][k] = [np.mean(v)]
                else:
                    test_results[month_idx][k].append(np.mean(v))
        for month_idx in range(1, 13):
            for k, v in test_results[month_idx].items():
                test_results[month_idx][k] = (np.mean(v), 2 * np.std(v))
        self.print_info(test_results, True)
        return test_results

    def print_info(self, stat, split_month=False):
        if not split_month:
            string = [f'Test Results:']
            for k, v in stat.items():
                string.append(k+f': mean: {v[0]:2.4f}, \t2std: {v[1]:2.4f}')
            string = "\n".join(string)
        else:
            string = []
            for month in range(1, 13):
                string.append('{} Test Results:'.format(month))
                for k, v in stat[month].items():
                    string.append(
                        k+f': mean: {v[0]:2.4f}, \t2std: {v[1]:2.4f}')
            string = "\n".join(string)
        print(string)

    def visual_run(self, start_id, end_id):
        day_id = np.arange(start_id,end_id+1)
        save_list = list()
        num_epsiodes = end_id - start_id + 1
        for epi in range(num_epsiodes):
            # reset env
            comb_state, global_state, epi_year = self.env.manual_reset(day_id[epi], 23, 2)
                #test_data.iloc[epi]['day_id'], self.env._select_start_hour(), self.env._select_start_interval())

            state = comb_state["agents_obs"]
            state_time = comb_state["time_obs"]
            month = comb_state["month"]
            with th.no_grad():
                state_ = prep_obs(state).contiguous().view(
                    1, self.n_, self.obs_dim).to(self.device)
                state_time_ = prep_obs(state_time).to(self.device).contiguous().view(
                    self.n_ ,self.tp, self.time_dims)
                emb_agent_glimpsed, _, mean_emb = self.behaviour_net.encode(state_, state_time_, month)
                x_glb = self.behaviour_net.policy_dicts[0].rt_glb(emb_agent_glimpsed, None, month, None)
                print(f"This is the episode: {epi}")
                #mean or policy_glb
                #save_list.append(th.mean(emb_agent_glimpsed, dim=0))
                save_list.append(th.mean(x_glb, dim=0))
        return save_list

    def day_run(self):
        test_results = {}
        num_epsiodes = 365
        day_run_step = 480
        start_2014_dayid = (datetime(2014,1,1) - datetime(2012,1,1)).days
        for epi in range(num_epsiodes):
            # reset env
            comb_state, global_state, epi_year = self.env.manual_reset(
                start_2014_dayid + epi, 0, 0)
                #test_data.iloc[epi]['day_id'], self.env._select_start_hour(), self.env._select_start_interval())

            state = comb_state["agents_obs"]
            state_time = comb_state["time_obs"]
            month = comb_state["month"]

            # init hidden states
            last_hid = self.behaviour_net.policy_dicts[0].init_hidden()
            steps_count = 0
            for t in range(day_run_step):
                if self.render:
                    self.env.render()
                    time.sleep(0.01)
                state_ = prep_obs(state).contiguous().view(
                    1, self.n_, self.obs_dim).to(self.device)
                state_time_ = prep_obs(state_time).to(self.device).contiguous().view(
                    self.n_ ,self.tp, self.time_dims)
                with th.no_grad():
                    action, _, _, _, hid = self.behaviour_net.get_actions(state_, state_time_, month, status='test', exploration=False, actions_avail=th.tensor(
                        self.env.get_avail_actions()), target=False, last_hid=last_hid)
                _, actual = translate_action(self.args, action, self.env)
                reward, done, info = self.env.step(actual, add_noise=False)
                done_ = done or t == self.args.max_steps-1
                next_comb_state = self.env.get_obs()
                next_state = next_comb_state["agents_obs"]
                next_state_time = next_comb_state["time_obs"]
                next_month = next_comb_state["month"]
                for k, v in info.items():
                    if 'mean_test_'+k not in test_results.keys():
                        test_results['mean_test_'+k] = [v]
                    else:
                        test_results['mean_test_'+k].append(v)
                # set the next state
                state = next_state
                state_time = next_state_time
                month = next_month
                # set the next last_hid
                last_hid = hid
                steps_count += 1
                if done_:
                    break 
            print(f"This is the test episode: {epi}")
        for k, v in test_results.items():
            test_results[k] = (np.mean(v), 2 * np.std(v))
        #self.print_info(test_results)
        return test_results
 
    def long_run(self):
        test_results = {}
        for month in range(1, 13): 
            test_results[month] = {}
        num_epsiodes = 12
        long_run_step = 30 * 480
        self.env.set_episode_limit(long_run_step)
        for epi in range(num_epsiodes):
            # reset env
            start_2014_dayid = (datetime(2014,epi + 1,1) - datetime(2012,1,1)).days
            comb_state, global_state, epi_year = self.env.manual_reset(
                start_2014_dayid, 0, 0)
                #test_data.iloc[epi]['day_id'], self.env._select_start_hour(), self.env._select_start_interval())

            state = comb_state["agents_obs"]
            state_time = comb_state["time_obs"]
            month = comb_state["month"]
            month_idx = epi + 1

            # init hidden states
            last_hid = self.behaviour_net.policy_dicts[0].init_hidden()
            result = {}
            steps_count = 0
            for t in range(long_run_step):
                if self.render:
                    self.env.render()
                    time.sleep(0.01)
                state_ = prep_obs(state).contiguous().view(
                    1, self.n_, self.obs_dim).to(self.device)
                state_time_ = prep_obs(state_time).to(self.device).contiguous().view(
                    self.n_ ,self.tp, self.time_dims)
                with th.no_grad():
                    action, _, _, _, hid = self.behaviour_net.get_actions(state_, state_time_, month, status='test', exploration=False, actions_avail=th.tensor(
                        self.env.get_avail_actions()), target=False, last_hid=last_hid)
                _, actual = translate_action(self.args, action, self.env)
                reward, done, info = self.env.step(actual, add_noise=False)
                done_ = done or t == self.args.max_steps-1
                next_comb_state = self.env.get_obs()
                next_state = next_comb_state["agents_obs"]
                next_state_time = next_comb_state["time_obs"]
                next_month = next_comb_state["month"]
                for k, v in info.items():
                    if 'mean_test_'+k not in result.keys():
                        result['mean_test_'+k] = [v]
                    else:
                        result['mean_test_'+k].append(v)
                # set the next state
                state = next_state
                state_time = next_state_time
                month = next_month
                # set the next last_hid
                last_hid = hid
                steps_count += 1
                if done_:
                    break
            print(f"This is the test episode: {epi}")
            for k, v in result.items():
                if k not in test_results[month_idx].keys():
                    test_results[month_idx][k] = [np.mean(v)]
                else:
                    test_results[month_idx][k].append(np.mean(v))
            test_results[month_idx]["running_steps"] = steps_count
        for month_idx in range(1, 13):
            for k, v in test_results[month_idx].items():
                test_results[month_idx][k] = (np.mean(v), 2 * np.std(v))
        self.print_info(test_results, True)
        return test_results

    def year_run(self, num_epsiodes=1):
        test_results = {}
        for epi in range(num_epsiodes):
            # reset env
            start_2013_dayid = (datetime(2013,1,1) - datetime(2012,1,1)).days
            year_steps = 480 * 365
            comb_state, global_state,_ = self.env.manual_reset(start_2013_dayid,0,0)

            state = comb_state["agents_obs"]
            state_time = comb_state["time_obs"]
            month = comb_state["month"]

            # init hidden states
            last_hid = self.behaviour_net.policy_dicts[0].init_hidden()
            steps_count = 0
            for t in range(year_steps):
                if self.render:
                    self.env.render()
                    time.sleep(0.01)
                state_ = prep_obs(state).contiguous().view(
                    1, self.n_, self.obs_dim).to(self.device)
                state_time_ = prep_obs(state_time).to(self.device).contiguous().view(
                    self.n_ ,self.tp, self.time_dims)
                with th.no_grad():
                    action, _, _, _, hid = self.behaviour_net.get_actions(state_, state_time_, month, status='test', exploration=False, actions_avail=th.tensor(
                        self.env.get_avail_actions()), target=False, last_hid=last_hid)
                _, actual = translate_action(self.args, action, self.env)
                reward, done, info = self.env.step(actual, add_noise=False)
                done_ = done or t == self.args.max_steps-1
                next_comb_state = self.env.get_obs()
                next_state = next_comb_state["agents_obs"]
                next_state_time = next_comb_state["time_obs"]
                next_month = next_comb_state["month"]
                for k, v in info.items():
                    if 'mean_test_'+k not in test_results.keys():
                        test_results['mean_test_'+k] = [v]
                    else:
                        test_results['mean_test_'+k].append(v)
                # set the next state
                state = next_state
                state_time = next_state_time
                month = next_month
                # set the next last_hid
                last_hid = hid
                steps_count += 1
                if done_:
                    break
            print(f"This is the test episode: {epi}")
        for k, v in test_results.items():
            test_results[k] = (np.mean(v), 2 * np.std(v))
        test_results["steps"] = steps_count
        #self.print_info(test_results)
        return test_results

    def all_year_run(self, num_epsiodes=1):
        test_results = {}
        for epi in range(num_epsiodes):
            # reset env
            start_2014_dayid = (datetime(2013,1,1) - datetime(2012,1,1)).days
            year_steps = 480 * 365 * 2 - 480 * 4
            comb_state, global_state,_ = self.env.manual_reset(start_2014_dayid,0,0)

            state = comb_state["agents_obs"]
            state_time = comb_state["time_obs"]
            month = comb_state["month"]

            # init hidden states
            last_hid = self.behaviour_net.policy_dicts[0].init_hidden()
            steps_count = 0
            for t in range(year_steps):
                if self.render:
                    self.env.render()
                    time.sleep(0.01)
                state_ = prep_obs(state).contiguous().view(
                    1, self.n_, self.obs_dim).to(self.device)
                state_time_ = prep_obs(state_time).to(self.device).contiguous().view(
                    self.n_ ,self.tp, self.time_dims)
                with th.no_grad():
                    action, _, _, _, hid = self.behaviour_net.get_actions(state_, state_time_, month, status='test', exploration=False, actions_avail=th.tensor(
                        self.env.get_avail_actions()), target=False, last_hid=last_hid)
                _, actual = translate_action(self.args, action, self.env)
                reward, done, info = self.env.step(actual, add_noise=False)
                done_ = done or t == self.args.max_steps-1
                next_comb_state = self.env.get_obs()
                next_state = next_comb_state["agents_obs"]
                next_state_time = next_comb_state["time_obs"]
                next_month = next_comb_state["month"]
                for k, v in info.items():
                    if 'mean_test_'+k not in test_results.keys():
                        test_results['mean_test_'+k] = [v]
                    else:
                        test_results['mean_test_'+k].append(v)
                # set the next state
                state = next_state
                state_time = next_state_time
                month = next_month
                # set the next last_hid
                last_hid = hid
                steps_count += 1
                if done_:
                    break
            print(f"This is the test episode: {epi}")
        for k, v in test_results.items():
            test_results[k] = (np.mean(v), 2 * np.std(v))
        test_results["steps"] = steps_count
        #self.print_info(test_results)
        return test_results
    
    def all_year_run(self, num_epsiodes=1):
        test_results = {}
        for epi in range(num_epsiodes):
            # reset env
            start_2014_dayid = 3
            year_steps = 480 * 365 * 3 - 480 * 8
            comb_state, global_state,_ = self.env.manual_reset(start_2014_dayid,0,0)

            state = comb_state["agents_obs"]
            state_time = comb_state["time_obs"]
            month = comb_state["month"]

            # init hidden states
            last_hid = self.behaviour_net.policy_dicts[0].init_hidden()
            steps_count = 0
            for t in range(year_steps):
                if self.render:
                    self.env.render()
                    time.sleep(0.01)
                state_ = prep_obs(state).contiguous().view(
                    1, self.n_, self.obs_dim).to(self.device)
                state_time_ = prep_obs(state_time).to(self.device).contiguous().view(
                    self.n_ ,self.tp, self.time_dims)
                with th.no_grad():
                    action, _, _, _, hid = self.behaviour_net.get_actions(state_, state_time_, month, status='test', exploration=False, actions_avail=th.tensor(
                        self.env.get_avail_actions()), target=False, last_hid=last_hid)
                _, actual = translate_action(self.args, action, self.env)
                reward, done, info = self.env.step(actual, add_noise=False)
                done_ = done or t == self.args.max_steps-1
                next_comb_state = self.env.get_obs()
                next_state = next_comb_state["agents_obs"]
                next_state_time = next_comb_state["time_obs"]
                next_month = next_comb_state["month"]
                for k, v in info.items():
                    if 'mean_test_'+k not in test_results.keys():
                        test_results['mean_test_'+k] = [v]
                    else:
                        test_results['mean_test_'+k].append(v)
                # set the next state
                state = next_state
                state_time = next_state_time
                month = next_month
                # set the next last_hid
                last_hid = hid
                steps_count += 1
                if done_:
                    break
            print(f"This is the test episode: {epi}")
        for k, v in test_results.items():
            test_results[k] = (np.mean(v), 2 * np.std(v))
        test_results["steps"] = steps_count
        #self.print_info(test_results)
        return test_results
    
    def three_year_run(self, num_epsiodes=1):
        test_results = {}
        for epi in range(num_epsiodes):
            # reset env
            start_2014_dayid = 3
            year_steps = 480 * 365 * 3 - 480 * 8
            comb_state, global_state,_ = self.env.manual_reset(start_2014_dayid,0,0)

            state = comb_state["agents_obs"]
            state_time = comb_state["time_obs"]
            month = comb_state["month"]

            # init hidden states
            last_hid = self.behaviour_net.policy_dicts[0].init_hidden()
            steps_count = 0
            for t in range(year_steps):
                if self.render:
                    self.env.render()
                    time.sleep(0.01)
                state_ = prep_obs(state).contiguous().view(
                    1, self.n_, self.obs_dim).to(self.device)
                state_time_ = prep_obs(state_time).to(self.device).contiguous().view(
                    self.n_ ,self.tp, self.time_dims)
                with th.no_grad():
                    action, _, _, _, hid = self.behaviour_net.get_actions(state_, state_time_, month, status='test', exploration=False, actions_avail=th.tensor(
                        self.env.get_avail_actions()), target=False, last_hid=last_hid)
                _, actual = translate_action(self.args, action, self.env)
                reward, done, info = self.env.step(actual, add_noise=False)
                done_ = done or t == self.args.max_steps-1
                next_comb_state = self.env.get_obs()
                next_state = next_comb_state["agents_obs"]
                next_state_time = next_comb_state["time_obs"]
                next_month = next_comb_state["month"]
                for k, v in info.items():
                    if 'mean_test_'+k not in test_results.keys():
                        test_results['mean_test_'+k] = [v]
                    else:
                        test_results['mean_test_'+k].append(v)
                # set the next state
                state = next_state
                state_time = next_state_time
                month = next_month
                # set the next last_hid
                last_hid = hid
                steps_count += 1
                if done_:
                    break
            print(f"This is the test episode: {epi}")
        for k, v in test_results.items():
            test_results[k] = (np.mean(v), 2 * np.std(v))
        test_results["steps"] = steps_count
        #self.print_info(test_results)
        return test_results