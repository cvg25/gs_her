from src.data.gc_datasampler import GCDataSampler
from src.data.ogbench import OGBenchDataSampler
from src.agent import agents_dict
from src.utils import get_params_from_yaml_file, seed_everything, get_device, compute_num_trainable_params, compute_gradient_norm, save_model_checkpoint
from src.normalize import EMANormalize
from src.logger import Logger 
from src.eval import eval_agent

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help='name of env.')
    parser.add_argument('--agent', type=str, help='name of agent.')
    parser.add_argument('--agent_alpha', type=float, help='awr parameter.', default=None)
    parser.add_argument('--her_type', type=str, help='HER type.')
    parser.add_argument('--query_strategy', type=str, help='Sampling stratery for the query (uniform, blockwise, semantic).', default=None)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    params = get_params_from_yaml_file(fpath='./configs/train.yaml')
    params['data']['task'] = args.env_name
    params['agent'] = {
        'name': args.agent,
        'alpha': args.agent_alpha
    }
    params['her_type'] = args.her_type
    params['query_strategy'] = args.query_strategy

    # -- Set up --
    device = get_device(device_number=params['training']['device_number'])
    seed = params['training']['seed'] if params['training']['seed'] != 'rand' else torch.randint(4096, size=()).item()
    seed_everything(seed=seed)
    print(f'Training seed: {seed}')
    params['training']['seed'] = seed

    # -- Data --
    ds = OGBenchDataSampler(dataset_name=params['data']['task'],
                            dataset_path=params['data']['path'])
    env = ds.env
    ds = GCDataSampler(datasampler=ds,
                       device=device, 
                       keep_dataset_on_device=True)

    # -- Models --
    params['state_dim'] = ds.state_dim
    params['action_dim'] = ds.action_dim
    params['proj_goal_idxs'] = ds.proj_goal_idxs
    params['max_task_length'] = ds.max_task_length  
    agent = agents_dict[params['agent']['name']](params=params).to(device)
    
    # -- Normalize --
    normalize_state = EMANormalize(shape=ds.state_dim).to(device)
    normalize_action = EMANormalize(shape=ds.action_dim).to(device)
    
    normalize_state, normalize_action = ds.compute_normalization_stats(
        normalize_state=normalize_state, 
        normalize_action=normalize_action
    )

    # -- Logger --
    run_name = Logger.get_run_name()
    run_name = run_name + f'_{args.agent}_{args.her_type}{("_" + args.query_strategy if args.query_strategy is not None else "")}'
    print(f'Training agent: {args.agent} with HER strategy: {args.her_type}')

    logger = Logger(
        root_folder=params['logging']['root_folder'],
        run_name=run_name,
        project_name=params['logging']['project_name'] + params['data']['task'],
        log_to_wandb=params['logging']['log_to_wandb'],
        config=params)

    trainable_params_dict = agent.get_trainable_params()
    logger.log_metadata(metadata_dict=trainable_params_dict)

    _ = save_model_checkpoint(root_path=logger.run_path, model=normalize_state, tag=f'normalize_state') 
    _ = save_model_checkpoint(root_path=logger.run_path, model=normalize_action, tag=f'normalize_action')   

    # -- Optimizer --
    param_groups = [
        {
            'params': list(agent.parameters()),
            'lr': params['training']['lr']
        }
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # -- Training Loop --
    num_iterations = params['training']['num_iterations']
    batch_size = params['training']['batch_size']
    value_p_goal_curr = agent.value_p_goal_curr
    value_p_goal_traj = agent.value_p_goal_traj
    value_p_goal_rand = agent.value_p_goal_rand
    actor_p_goal_curr = agent.actor_p_goal_curr
    actor_p_goal_traj = agent.actor_p_goal_traj
    actor_p_goal_rand = agent.actor_p_goal_rand
    query_strategy = params['query_strategy']

    last3_task_success = []
    last3_avg_steps = []

    for iter_idx in tqdm(range(num_iterations)):
        logger.increment_iteration()

        data_dict = ds.sample(
            batch_size=batch_size,
            value_p_goal_curr=value_p_goal_curr,
            value_p_goal_traj=value_p_goal_traj,
            value_p_goal_rand=value_p_goal_rand,
            actor_p_goal_curr=actor_p_goal_curr,
            actor_p_goal_traj=actor_p_goal_traj,
            actor_p_goal_rand=actor_p_goal_rand,
            query_strategy=query_strategy
        )
        loss, info = agent.forward_train(data_dict=data_dict)

        optimizer.zero_grad()
        loss.backward()
        
        info = agent.clip_grad_norm(max_norm=params['training']['max_norm'], info=info)

        optimizer.step()
        
        info['lr_sched'] = optimizer.param_groups[0]["lr"]

        agent.ema_update()

        # Log training stats.
        for k, v in info.items():
            logger.update_iteration(key=k, value=v)

        #if iter_idx % params['training']['eval_every_num_iters'] == 0:
        if iter_idx in params['training']['eval_at_iters']:
            if iter_idx == 0:
                eval_info = {
                    'eval/task_success': 0.0,
                    'eval/avg_steps': agent.max_task_length,
                }
            else:
                # Run evaluation
                (agent, 
                normalize_state, 
                normalize_action,
                eval_info, 
                _, 
                _) = eval_agent(
                    env=env, 
                    num_episodes_per_task=params['training']['eval_episodes_per_task'], 
                    agent=agent, 
                    normalize_state=normalize_state,
                    normalize_action=normalize_action,
                    device=device)

                # Update last 3 success and steps.
                last3_task_success.append(eval_info['eval/task_success'])
                last3_task_success = last3_task_success[-3:]
                last3_avg_steps.append(eval_info['eval/avg_steps'])
                last3_avg_steps = last3_avg_steps[-3:]

            # Log training stats.
            for k, v in eval_info.items():
                logger.update_iteration(key=k, value=v)

        logger.log_iteration()

        if (params['logging']['save_chkpt'] and
            iter_idx % params['logging']['save_chkpt_every_niters'] == 0):
            _ = save_model_checkpoint(root_path=logger.run_path, model=agent, tag=f'agent_{iter_idx}')

    _ = save_model_checkpoint(root_path=logger.run_path, model=agent, tag=f'agent')

    # Run evaluation
    (_, 
    _, 
    _,
    info, 
    rewards, 
    num_steps) = eval_agent(
        env=env, 
        num_episodes_per_task=params['training']['eval_episodes_per_task'], 
        agent=agent, 
        normalize_state=normalize_state,
        normalize_action=normalize_action,
        device=device)

    # Compute final result as average of last 3 evals.
    last3_task_success.append(info['eval/task_success'])
    last3_task_success = last3_task_success[-3:]
    last3_avg_steps.append(info['eval/avg_steps'])
    last3_avg_steps = last3_avg_steps[-3:]

    avg_task_success = torch.tensor(last3_task_success).float().mean()
    avg_avg_steps = torch.tensor(last3_avg_steps).float().mean()
    
    info['eval/avg_task_success'] = avg_task_success.item()
    info['eval/avg_avg_steps'] = avg_avg_steps.item()

    logger.log_eval(info_dict=info, rewards=rewards, num_steps=num_steps)

    print(info)



