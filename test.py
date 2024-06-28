import torch
import argparse
import yaml
import pickle
import os,random
import numpy as np

from models.model_registry import Model, Strategy
from environments.var_voltage_control.voltage_control_env import VoltageControl
from utilities.util import convert
from utilities.tester import PGTester
from datetime import datetime

def test_one_step(net, env):
    net = net.to(net.device)
    import numpy as np
    obs, _ = env.reset()
    act = env.get_action()
    obs_reshape = torch.tensor(np.array(obs))[None, :, :].float().cuda()
    act_reshape = torch.tensor(np.array(act))[None, :, None].float().cuda()
    act_reshape, _, _ = net.policy(
        obs_reshape, last_hid=net.policy_dicts[0].init_hidden())
    value, cost = net.value(obs_reshape, act_reshape)
    act = act_reshape.detach().squeeze().cpu().numpy()
    reward, done, info = env.step(act)
    true_cost = info['percentage_of_v_out_of_control']
    print(true_cost)
    print(cost)
    print("ok")

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description="Train rl agent.")
parser.add_argument("--save-path", type=str, nargs="?", default="./trial/model_save",
                    help="Please enter the directory of saving model.")
parser.add_argument("--alg", type=str, nargs="?",
                    default="icstransmaddpg", help="Please enter the alg name.")
parser.add_argument("--seed", type=int, nargs="?",
                    default="53")
parser.add_argument("--env", type=str, nargs="?",
                    default="var_voltage_control", help="Please enter the env name.")
parser.add_argument("--alias", type=str, nargs="?", default="k=1_3layer_actor_t_aux_critic_raw_t_6",
                    help="Please enter the alias for exp control.")
parser.add_argument("--mode", type=str, nargs="?", default="distributed",
                    help="Please enter the mode: distributed or decentralised.")
parser.add_argument("--scenario", type=str, nargs="?", default="case322_3min_final",
                    help="Please input the valid name of an environment scenario.")
parser.add_argument("--qweight", type=float, nargs="?", default=0.1,
                    help="Please input the q weight of env: 0.01 for case141 and 0.1 for case322")
parser.add_argument("--voltage-barrier-type", type=str, nargs="?", default="l2",
                    help="Please input the valid voltage barrier type: l1, courant_beltrami, l2, bowl or bump.")
parser.add_argument("--date-emb",  action='store_true')
parser.add_argument("--test-mode", type=str, nargs="?", default="single",
                    help="Please input the valid test mode: single or batch.")
parser.add_argument("--test-day", type=int, nargs="?", default=454, # 454 903 621 704
                    help="Please input the day you would test if the test mode is single.")
parser.add_argument("--render", action="store_true",
                    help="Activate the rendering of the environment.")
parser.add_argument("--best", action="store_true",
                    help="Activate the rendering of the environment.")
argv = parser.parse_args()

seed_torch(argv.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# load env args
with open("./args/env_args/"+argv.env+".yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"].split("/")
data_path[-1] = argv.scenario
env_config_dict["data_path"] = "/".join(data_path)
net_topology = argv.scenario

# set the action range
assert net_topology in ['case33_3min_final', 'case141_3min_final',
                        'case322_3min_final'], f'{net_topology} is not a valid scenario.'
if argv.scenario == 'case33_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8
elif argv.scenario == 'case141_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.6
elif argv.scenario == 'case322_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8

assert argv.mode in ['distributed', 'decentralised',
                     'centralised'], "Please input the correct mode, e.g. distributed or decentralised."
env_config_dict["mode"] = argv.mode
env_config_dict["voltage_barrier_type"] = argv.voltage_barrier_type
env_config_dict["voltage_loss"] = argv.voltage_barrier_type

env_config_dict["q_weight"] = argv.qweight

if argv.date_emb:
    env_config_dict["state_space"].append("date")

# load default args
with open("./args/default.yaml", "r") as f:
    default_config_dict = yaml.safe_load(f)
default_config_dict["max_steps"] = 480
default_config_dict["cuda"] = True

# for one-day test
if argv.test_mode == "long":
    env_config_dict["episode_limit"] = 480 * 30
    default_config_dict["max_steps"] = 480 * 30 
    print("long test")
elif argv.test_mode == "year":
    env_config_dict["episode_limit"] = 480 * 365
    default_config_dict["max_steps"] = 480 * 365
    print("year test")
elif argv.test_mode == "two_year":
    env_config_dict["episode_limit"] = 480 * 365 * 2 - 480 * 4
    default_config_dict["max_steps"] = 480 * 365 * 2 - 480 * 4
    print("all year test")
elif argv.test_mode == "three_year":
    env_config_dict["episode_limit"] = 480 * 365 * 3 - 480 * 8
    default_config_dict["max_steps"] = 480 * 365 * 3 - 480 * 8
    print("three year test")
else:
    env_config_dict["episode_limit"] = 480

# load alg args
with open("./args/alg_args/"+argv.alg+".yaml", "r") as f:
    alg_config_dict = yaml.safe_load(f)["alg_args"]
    alg_config_dict["action_scale"] = env_config_dict["action_scale"]
    alg_config_dict["action_bias"] = env_config_dict["action_bias"]

log_name = "-".join([argv.env, net_topology, argv.mode,
                    argv.alg, argv.voltage_barrier_type, str(argv.seed) ,argv.alias])#str(argv.seed) ,argv.alias])
alg_config_dict = {**default_config_dict, **alg_config_dict}

# define envs
env = VoltageControl(env_config_dict)
env.voltage_barrier_type = "l1"

alg_config_dict["agent_num"] = env.get_num_of_agents()
alg_config_dict["obs_size"] = env.get_obs_size()
alg_config_dict["obs_bus_dim"] = env.get_obs_dim()
alg_config_dict["obs_bus_num"] = env.get_obs_bus_num()
alg_config_dict["action_dim"] = env.get_total_actions()
alg_config_dict["bus_num"] = env.get_num_of_buses()
# alg_config_dict["obs_position_list"] = env.get_obs_position_list()
alg_config_dict["region_num"] = env.get_num_of_regions()
alg_config_dict['constraint_mask'] = env.get_constraint_mask()
alg_config_dict['agent2region'] = env.get_agent2region()
alg_config_dict['agent_index_in_obs'] = env.get_agent_index_in_obs()
alg_config_dict['region_adj'] = env.get_region_adj()
alg_config_dict['tp'] = env.tp
alg_config_dict['time_dims'] = env.time_dims
#alg_config_dict['jieqi_info'] = env.get_24jieqi(year=2012)  
alg_config_dict['jieqi_info'] = env.get_24jieqi(year=2012)  

if argv.date_emb:
    alg_config_dict['agent_type'] = "rnn_with_date"
    alg_config_dict['use_date'] = True

args = convert(alg_config_dict)

# define the save path
if argv.save_path[-1] is "/":
    save_path = argv.save_path
else:
    save_path = argv.save_path+"/"

if argv.best:
    LOAD_PATH = save_path+log_name+"/best_model.pt"
    print("LOAD best")
else:
    LOAD_PATH = save_path+log_name+"/model.pt"

model = Model[argv.alg]

strategy = Strategy[argv.alg]

if args.target:
    target_net = model(args)
    behaviour_net = model(args, target_net)
else:
    behaviour_net = model(args)
checkpoint = torch.load(
    LOAD_PATH, map_location='cpu') if not args.cuda else torch.load(LOAD_PATH)
behaviour_net.load_state_dict(checkpoint['model_state_dict'],strict=True)

'''
jieqi_info = torch.FloatTensor(args.jieqi_info).to(0)
print(jieqi_info.shape)
pp = behaviour_net.policy_dicts[0].rt_pp(jieqi_info)
torch.save(pp,"./saved_pp.h5")
exit()
'''

# test_one_step(behaviour_net, env)
if strategy == "pg":
    test = PGTester(args, behaviour_net, env, argv.render)
elif strategy == "q":
    raise NotImplementedError("This needs to be implemented.")
else:
    raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

print(env.voltage_loss_type)

if argv.test_mode == 'single':
    # record = test.run(199, 23, 2) # (day, hour, 3min)
    # record = test.run(730, 23, 2) # (day, hour, 3min)
    record = test.run(argv.test_day, 23, 2)
    with open('test_record/test_record_'+log_name+f'_day{argv.test_day}'+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
elif argv.test_mode == 'batch':
    record = test.batch_run(100)
    with open('test_record/test_record_'+log_name+'_'+argv.test_mode+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
elif argv.test_mode == 'test_data':
    import pandas as pd
    dataframe = pd.read_csv("test_data.csv")
    record = test.test_data_run(dataframe)
    with open('test_record/test_record_'+log_name+'_'+argv.test_mode+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
elif argv.test_mode == 'day':
    import pandas as pd
    record = test.day_run()
    with open('test_record/test_record_'+log_name+'_'+argv.test_mode+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
elif argv.test_mode == 'long':
    import pandas as pd
    record = test.long_run()
    with open('test_record/test_record_'+log_name+'_'+argv.test_mode+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
elif argv.test_mode == 'year':
    record = test.year_run()
    with open('test_record/test_record_'+log_name+'_'+argv.test_mode+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
elif argv.test_mode == 'two_year':
    record = test.all_year_run()
    with open('test_record/test_record_'+log_name+'_'+argv.test_mode+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
elif argv.test_mode == 'three_year':
    record = test.three_year_run()
    with open('test_record/test_record_'+log_name+'_'+argv.test_mode+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
