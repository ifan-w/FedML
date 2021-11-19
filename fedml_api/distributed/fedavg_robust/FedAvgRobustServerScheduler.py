import os
import time
import re
import wandb

class FedAvgRobustServerScheduler(object):
    def __init__(self, server_manager, args):
        self.server_manager = server_manager
        self.args = args
        
        # load model, if path specified
        self.model_files = {}
        self.model_load_path = None if self.args.load_model_path == "" else self.args.load_model_path
        if self.model_load_path is not None:
            regex = re.compile('round-(\d+).(pth)')
            files = os.listdir(self.model_load_path)
            for fname in files:
                if os.path.isfile(os.path.join(self.model_load_path, fname)):
                    result = regex.findall(fname)
                    if len(result) > 0 and result[0][1] == 'pth':
                        self.model_files[int(result[0][0])] = os.path.join(
                            self.model_load_path,
                            fname
                        )
                    
        # save model, if load model set, set to none
        self.model_save_path = None
        self.model_save_freq = args.save_model_freq
        if self.model_save_freq > 0 and self.model_load_path is None:
            self.model_save_path = os.path.join(
                os.getcwd(),
                "models",
                f"{self.args.dataset}-{self.args.model}",
                f"{self.args.client_optimizer}-{self.args.lr}-{self.args.comm_round}-{self.args.epochs}-" +
                time.asctime().replace(" ","-").replace(":", "-")
            )
            os.makedirs(self.model_save_path)

        # select schedule generator
        # no attack
        if self.args.attack_type == "none":
            self.schedule_gen = self.na_save()
        # Single-shot attack
        elif self.args.attack_type == "single_shot":
            if self.model_load_path is None:
                self.schedule_gen = self.sa_save()
            else:
                self.schedule_gen = self.sa_load()
        # Multi-shot attack
        elif self.args.attack_type == "multi_shot":
            if self.model_load_path is None:
                self.schedule_gen = self.ma_save()
            else:
                self.schedule_gen = self.ma_load()
        else:
            raise NotImplementedError()

        # variable expose to manager
        self.model_replacement = False
        self.client_idx = []
        self.round_idx = -1
        self.model_params = None
        self.require_test = False

    def na_save(self):
        self.round_idx = 0
        self.client_idx = self.server_manager.gen_client_idx(self.round_idx)
        yield
        for i in range(1, self.args.comm_round):
            # update round index
            self.round_idx = i
            # test target accuracy on clean model
            wandb.log({
                "Round": self.round_idx,
                f"Target/Clean-Acc": self.server_manager.get_target_acc()
            })
            # save model
            if self.model_save_freq > 0 and (self.round_idx % self.model_save_freq == 0):
                self.server_manager.save_model(
                    os.path.join(
                        self.model_save_path,
                        f'round-{self.round_idx}.pth'
                    )
                )
            # update client index
            self.client_idx = self.server_manager.gen_client_idx(self.round_idx)
            if self.round_idx % self.args.frequency_of_the_test == 0:
                self.require_test = True
            else:
                self.require_test = False
            yield

    def sa_load(self):
        self.require_test = True
        for round, fname in self.model_files.items():
            self.round_idx = round
            self.server_manager.load_model(fname)
            self.client_idx = self.server_manager.gen_client_idx(self.round_idx)
            self.client_idx[0] = -1 # set worker 0 to attack
            self.model_replacement = True
            yield
            wandb.log({
                "Round": self.round_idx - round,
                f"Target/{round}-Acc": self.server_manager.get_target_acc()
            })            
            for i in range(round + 1, round + self.args.attack_afterward):
                self.model_replacement = False
                self.round_idx = i
                self.client_idx = self.server_manager.gen_client_idx(self.round_idx)
                yield
                wandb.log({
                    "Round": self.round_idx - round,
                    f"Target/{round}-Acc": self.server_manager.get_target_acc()
                })  

    def sa_save(self):
        raise NotImplementedError()

    def ma_load(self):
        pass

    def ma_save(self):
        raise NotImplementedError()

    def step(self):
        try:
            next(self.schedule_gen)
        except StopIteration:
            return False
        return True