import torch
import numpy as np
import ogbench

env_specific_configuration = {
    'pointmaze-medium-navigate-v0': {
        'max_task_length': 500,
        'proj_goal_idxs': [0, 1],
        'state_semantics': {
            'pos': [0, 1]
        }
    },
    'pointmaze-large-navigate-v0': {
        'max_task_length': 1000,
        'proj_goal_idxs': [0, 1],
        'state_semantics': {
            'pos': [0, 1]
        }
    },
    'antmaze-medium-navigate-v0': {
        'max_task_length': 1000,
        'proj_goal_idxs': [0,1],
        'state_semantics': {
            'root_xy': [0,1],
            'root_z': [2],
            'root_quat': [3,4,5,6],
            'root_lin_vel': [15,16,17],
            'root_ang_vel': [18,19,20],
            'front_left_leg': [7,8,21,22],
            'front_right_leg': [9,10,23,24],
            'back_left_leg': [11,12,25,26],
            'back_right_leg': [13,14,27,28]
        }
    },
    'cube-single-noisy-v0': {
        'max_task_length': 200,
        'proj_goal_idxs': [19,20,21], # cube position.
        'state_semantics': {
            'joint_pos': [0, 1, 2, 3, 4, 5],
            'joint_vel': [6, 7, 8, 9, 10, 11],
            'eff_pos': [12, 13, 14],
            'eff_yaw': [15, 16],
            'gripper': [17, 18],
            'block0_pos': [19, 20, 21],
            'block0_quat': [22, 23, 24, 25],
            'block0_yaw': [26, 27],
        }
    },
    'cube-single-play-v0': {
        'max_task_length': 200,
        'proj_goal_idxs': [19,20,21], # cube position.
        'state_semantics': {
            'joint_pos': [0, 1, 2, 3, 4, 5],
            'joint_vel': [6, 7, 8, 9, 10, 11],
            'eff_pos': [12, 13, 14],
            'eff_yaw': [15, 16],
            'gripper': [17, 18],
            'block0_pos': [19, 20, 21],
            'block0_quat': [22, 23, 24, 25],
            'block0_yaw': [26, 27],
        }
    },
    'cube-double-noisy-v0': {
        'max_task_length': 500,
        'proj_goal_idxs': [19,20,21,28,29,30], # cube positions.
        'state_semantics': {
            'joint_pos': [0, 1, 2, 3, 4, 5],
            'joint_vel': [6, 7, 8, 9, 10, 11],
            'eff_pos': [12, 13, 14],
            'eff_yaw': [15, 16],
            'gripper': [17, 18],
            'block0_pos': [19, 20, 21],
            'block0_quat': [22, 23, 24, 25],
            'block0_yaw': [26, 27],
            'block1_pos': [28, 29, 30],
            'block1_quat': [31, 32, 33, 34],
            'block1_yaw': [35, 36]
        }
    },
    'cube-double-play-v0': {
        'max_task_length': 500,
        'proj_goal_idxs': [19,20,21,28,29,30], # cube positions.
        'state_semantics': {
            'joint_pos': [0, 1, 2, 3, 4, 5],
            'joint_vel': [6, 7, 8, 9, 10, 11],
            'eff_pos': [12, 13, 14],
            'eff_yaw': [15, 16],
            'gripper': [17, 18],
            'block0_pos': [19, 20, 21],
            'block0_quat': [22, 23, 24, 25],
            'block0_yaw': [26, 27],
            'block1_pos': [28, 29, 30],
            'block1_quat': [31, 32, 33, 34],
            'block1_yaw': [35, 36]
        }
    },
    'scene-play-v0': {
        'max_task_length': 1000,
        'proj_goal_idxs': [19, 20, 21, 29, 33, 36, 38],
        'state_semantics': {
            'joint_pos': [0, 1, 2, 3, 4, 5],
            'joint_vel': [6, 7, 8, 9, 10, 11],
            'eff_pos': [12, 13, 14],
            'eff_yaw': [15, 16],
            'gripper': [17, 18],
            'block0_pos': [19, 20, 21],
            'block0_quat': [22, 23, 24, 25],
            'block0_yaw': [26, 27],    
            'button0_state': [28, 29],
            'button0_pos': [30],
            'button0_vel': [31],
            'button1_state': [32, 33],
            'button1_pos': [34],
            'button1_vel': [35],
            'drawer_pos': [36],
            'drawer_vel': [37],
            'window_pos': [38],
            'window_vel': [39],
        }
    }
}

class OGBenchDataSampler():

    def __init__(self, 
                 dataset_name,
                 dataset_path='./ogbench_datasets',
                 add_info=False,
                 verbose=True):
        super().__init__()
        env, data_dict, _ = ogbench.make_env_and_datasets(
            dataset_name, 
            dataset_dir=dataset_path,
            add_info=add_info
        )

        self.env = env
        self.obs_dim = data_dict['observations'].shape[-1]
        self.act_dim = data_dict['actions'].shape[-1]
        self.max_task_length = env_specific_configuration[dataset_name]['max_task_length']
        self.proj_goal_idxs = env_specific_configuration[dataset_name]['proj_goal_idxs']
        self.state_semantics = env_specific_configuration[dataset_name]['state_semantics']

        (self.states, 
        self.actions, 
        self.qpos, 
        self.qvel) = self._extract_trajectories(data_dict, add_info)
        
        self.num_trajectories = len(self.actions)
        self.num_steps = self.actions[0].shape[0]

        if verbose:
            print(f'{self.__class__.__name__}: loaded {dataset_name} with {self.num_trajectories} trajectories of {self.num_steps} steps. Obs dim: {self.obs_dim}, Act dim: {self.act_dim}')

    def __len__(self):
        return len(self.actions)

    def _extract_trajectories(self, data_dict, add_info):
        states = []
        actions = []
        qpos = []
        qvel = []

        t_init = 0
        for i in range(len(data_dict['observations'])):
            if data_dict['terminals'][i] == 1:
                t_end = i
                traj_states = np.concatenate((data_dict['observations'][t_init:t_end+1], 
                                         data_dict['observations'][t_end:t_end+1]), axis=0)
                traj_actions = data_dict['actions'][t_init:t_end+1]
                if add_info:
                    traj_qpos = data_dict['qpos'][t_init:t_end+1]
                    traj_qvel = data_dict['qvel'][t_init:t_end+1]
                
                states.append(traj_states)
                actions.append(traj_actions)
                if add_info:
                    qpos.append(traj_qpos)
                    qvel.append(traj_qvel)
                t_init = t_end + 1
        
        states = torch.from_numpy(np.asarray(states))
        actions = torch.from_numpy(np.asarray(actions))
        if add_info:
            qpos = torch.from_numpy(np.asarray(qpos))
            qvel = torch.from_numpy(np.asarray(qvel))

        return states, actions, qpos, qvel

    def to(self, device, non_blocking=True):
        self.states = self.states.to(device, non_blocking=non_blocking)
        self.actions = self.actions.to(device, non_blocking=non_blocking)

        if self.qpos is not None and len(self.qpos) > 0:
            self.qpos = self.qpos.to(device, non_blocking=non_blocking)
        if self.qvel is not None and len(self.qpos) > 0:
            self.qvel = self.qvel.to(device, non_blocking=non_blocking)

        return self

    def sample_batch(self, batch_size):   
             
        # Sample random indexes.
        t_idxs = torch.randint(
            0, 
            self.num_trajectories, 
            (batch_size,),
            device=self.states.device,)

        # Select trajectories: [batch_size, num_steps, 2]
        states = self.states[t_idxs]
        actions = self.actions[t_idxs]

        return states, actions
