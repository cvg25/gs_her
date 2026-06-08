import torch
from tqdm import tqdm

def eval_agent(env, num_episodes_per_task, agent, normalize_state, normalize_action, device):
    agent = agent.eval()
    normalize_state = normalize_state.eval()
    normalize_action = normalize_action.eval()

    rewards = []
    num_steps = []

    for task_id in tqdm([1, 2, 3, 4, 5], desc="Tasks"):
        for episode_id in tqdm(range(num_episodes_per_task), desc=f"Task {task_id} episodes", leave=False):
            # Reset the agent inference params
            agent.reset()

            # Reset the environment and set the evalutation task.
            obs, info = env.reset(
                options=dict(task_id=task_id, render_goal=False)
            )

            goal = torch.tensor(info['goal'], dtype=torch.float, device=device)[None, :]
            s_t0 = torch.tensor(obs, dtype=torch.float, device=device)[None, :]
            goal = normalize_state(goal, update_stats=False)
            s_t0 = normalize_state(s_t0, update_stats=False)
            
            env_step = 0
            done = False

            while not done:
                action = agent.act(state=s_t0, goal=goal)
                action = normalize_action.undo(action)
                action = action.clamp(-1.0, 1.0)
                action = action[0].detach().cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                s_t0 = torch.tensor(obs, dtype=torch.float, device=device)[None, :]
                s_t0 = normalize_state(s_t0, update_stats=False)
                
                done = terminated or truncated
                env_step += 1

                if done:
                    rewards.append(float(reward))
                    num_steps.append(float(env_step))
    
    rewards = torch.tensor(rewards)
    num_steps = torch.tensor(num_steps)
    success_mask = rewards == 1.0
    info = {
        'eval/task_success': rewards.mean().item() * 100.0,
        'eval/avg_steps': num_steps[success_mask].mean().item() if success_mask.any() else agent.max_task_length,
    }
    
    agent = agent.train()
    normalize_state = normalize_state.train()
    normalize_action = normalize_action.train()

    return agent, normalize_state, normalize_action, info, rewards, num_steps


