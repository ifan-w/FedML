import logging

import torch
from torch import nn

from fedml_api.distributed.fedavg.utils import transform_tensor_to_list


class FedAvgRobustTrainer(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, model,
                 poisoned_train_loader, num_dps_poisoned_dataset, test_data_local_dict,
                 args):
        # TODO(@hwang595): double check if this makes sense with Chaoyang
        # here we always assume the client with `client_index=1` as the attacker
        self.client_index = client_index
        self.rank = client_index    # record the rank of this process, self.client_index will be changed in training
        self.adversarial = False
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num

        self.poisoned_train_loader = poisoned_train_loader
        self.num_dps_poisoned_dataset = num_dps_poisoned_dataset

        self.device = device
        self.args = args
        self.logger = args.logger
        self.model = model
        # logging.info(self.model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # TODO(hwang): since we only added the black-box attack now, we assume that the attacker uses the same hyper-params with the honest clients
        self.update_dataset(client_index)

    def new_optimizer(self):
        if not self.adversarial:
            if self.args.client_optimizer == "sgd":
                return torch.optim.SGD(
                    [{"params" :self.model.parameters(), "initial_lr": self.args.attack_lr}],
                    lr=self.args.lr,
                    momentum=0.9
                )
            else:
                return torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.args.lr,
                    weight_decay=self.args.wd, amsgrad=True
                )
        else:
            if self.args.attack_optimizer == "sgd":
                return torch.optim.SGD(
                    [{"params" :self.model.parameters(), "initial_lr": self.args.attack_lr}],
                    lr=self.args.attack_lr,
                    momentum=0.9
                )
            else:
                return torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.args.attack_lr,
                    weight_decay=self.args.wd, amsgrad=True
                )

    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.adversarial = False
        self.client_index = client_index
        # if self.client_index == 1: # TODO(@hwang595): double check if this makes sense with Chaoyang, we make it the attacker
        if self.client_index < 0:
            self.train_local = self.poisoned_train_loader
            self.local_sample_number = self.num_dps_poisoned_dataset
            self.adversarial = True
        else:
            self.train_local = self.train_data_local_dict[client_index]
            self.local_sample_number = self.train_data_local_num_dict[client_index]            

    def train(self, round_idx, reversable_train=False):
        self.model.to(self.device)
        # change to train mode
        self.model.train()

        optimizer = self.new_optimizer()
        epochs = self.args.attack_epochs if self.adversarial else self.args.epochs
        t_max = (self.args.comm_round + self.args.attack_afterward) * self.args.epochs + self.args.attack_epochs
        last_epoch = round_idx * self.args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            last_epoch=last_epoch
        )

        epoch_loss = []
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_local):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                self.logger.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(
                        self.client_index,
                        epoch,
                        sum(batch_loss) / len(batch_loss)
                    )
                )
                if self.adversarial and ((sum(batch_loss) / len(batch_loss)) <= self.args.attack_threshold):
                    break
            scheduler.step()           

        weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number

    def test(self):
        # divide clients to each worker, get index
        n_workers = self.args.client_num_per_round
        n_clients = self.args.client_num_in_total
        client_per_worker = n_clients // n_workers
        if self.rank == n_workers - 1:
            client_idxs = list(range(self.rank * client_per_worker, n_clients))
        else:
            client_idxs = list(range(self.rank * client_per_worker, (self.rank + 1) * client_per_worker))

        # test on train data
        train_total_correct = train_total_num = train_total_loss = 0
        for client_idx in client_idxs:
            n_correct, n_sample, loss = self._infer(self.train_data_local_dict[client_idx])
            train_total_correct += n_correct
            train_total_num += n_sample
            train_total_loss += loss
        train_acc = train_total_correct
        train_loss = train_total_loss
        train_sample_num = train_total_num

        # test on test data
        test_total_correct = test_total_num = test_total_loss = 0
        for client_idx in client_idxs:
            n_correct, n_sample, loss = self._infer(self.test_data_local_dict[client_idx])
            test_total_correct += n_correct
            test_total_num += n_sample
            test_total_loss += loss
        test_acc = test_total_correct
        test_loss = test_total_loss
        test_sample_num = test_total_num
        return train_acc, train_loss, train_sample_num, test_acc, test_loss, test_sample_num

    def _infer(self, test_data):
        self.model.eval()
        self.model.to(self.device)

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)
        return test_acc, test_total, test_loss