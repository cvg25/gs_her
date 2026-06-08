from pathlib import Path
import os
import yaml
import time
import wandb
import csv
import numpy as np

class Logger():

    def __init__(self, root_folder, project_name, run_name, log_to_wandb, config):
        self.project_name = project_name
        self.run_name = run_name
        self.root_folder = Path(root_folder)
        if not self.root_folder.exists():
            os.mkdir(self.root_folder)
        self.project_path = self.root_folder/project_name
        if not self.project_path.exists():
            os.mkdir(self.project_path)
        self.run_path = self.project_path/run_name
        if not self.run_path.exists():
            os.mkdir(self.run_path)
        self.iterations_fpath = self.run_path/'iterations.csv'
        self.config_fpath = self.run_path/'config.yaml'
        self.metadata_fpath = self.run_path/'metadata.csv'
        self.rewards_fpath = self.run_path/'rewards.csv'
        self.num_steps_fpath = self.run_path/'num_steps.csv'
        self.episodes_path = self.run_path/'episodes'
        self.log_to_wandb = log_to_wandb
        if self.log_to_wandb:
            wandb.login()
            wandb.init(
                project=self.project_name, 
                name=f"{self.run_name}", 
                config=config, 
                save_code=True,
                settings=wandb.Settings(code_dir="."))
        self.data_iteration = dict()
        self.iter_step = 0
        self.save_config_file(config)
    
    @staticmethod
    def get_run_name():
        return str(time.time()).split(".")[0]
    
    def save_config_file(self, args):
        with open(self.config_fpath, 'w') as f:
            yaml.dump(args, f)

    def update_iteration(self, key, value):
        self.data_iteration[key] = value

    def increment_iteration(self):
        self.iter_step += 1

    def _log_to_file(self, fpath, data):
        write_header = not fpath.exists()
        with open(fpath, 'a', newline='') as f:
            csvwriter = csv.writer(f)
            if write_header:
                csvwriter.writerow(list(data.keys())) 
            csvwriter.writerow(data.values())

    def log_iteration(self):
        # Log to wandb
        if self.iter_step == 1 and self.log_to_wandb:
            for key in self.data_iteration.keys():
                wandb.define_metric(key, step_metric='iter_step')
        
        self.data_iteration['iter_step'] = self.iter_step 
        if self.log_to_wandb:
            wandb.log(self.data_iteration)
        
        # Log to file
        self._log_to_file(fpath=self.iterations_fpath, data=self.data_iteration)
            
        # Reset
        self.data_iteration = dict()

    def log_metadata(self, metadata_dict):
        if self.log_to_wandb:
            wandb.log(metadata_dict)
        # Log to file
        self._log_to_file(fpath=self.metadata_fpath, data=metadata_dict)
    
    def log_eval(self, info_dict, rewards, num_steps):
        if self.log_to_wandb:
            wandb.log(info_dict)

        np.savetxt(
            self.rewards_fpath, 
            rewards.detach().cpu().numpy(),
            delimiter=',')

        np.savetxt(
            self.num_steps_fpath, 
            num_steps.detach().cpu().numpy(),
            delimiter=',')
