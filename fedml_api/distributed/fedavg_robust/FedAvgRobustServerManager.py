import logging
import time
from copy import deepcopy
from typing_extensions import final

from torch.distributed.distributed_c10d import send

from fedml_api.distributed.fedavg_robust.message_define import MyMessage
from fedml_api.distributed.fedavg.utils import transform_tensor_to_list
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager
import wandb


class FedAvgRobustServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.logger = args.logger
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0

        # {
        #     round_index: {
        #         process_idx: result,
        #         ...
        #     }...
        # }
        self.test_result = {}
        self.timestamp = time.time()

        # single-shot attack
        self.attack_round_idxs = [100, 300] + [i * 500 for i in range(1, 11)] + [i * 1000 for i in range(6, 10)] + [9500]
        # self.attack_round_idxs = [] # no attack round
        self.after_attack_countdown = -1
        self.clean_model_params = None
        self.round_freezing = False
        self.last_attack_round = -1

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(
            self.round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round
        )
        global_model_params = self.aggregator.get_global_model_params()
        
        # set timestamp
        self.timestamp = time.time()
        for process_id in range(1, self.size):
            self.send_message_sync_model_to_client(process_id, global_model_params, client_indexes[process_id-1], require_test=False)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_TEST_RESULT_TO_SERVER,
            self.handle_message_receive_test_result_from_client
        )

    # single shot attack
    def sa_restore_clean_model(self, target_acc, client_indexes):
        if self.round_freezing:
            wandb.log({
                "Round": self.args.attack_afterward - self.after_attack_countdown,
                f"Target/{self.round_idx}-Acc": target_acc
            })
        self.logger.info("---------- Finish Attack ----------")
        self.round_freezing = False
        clean_model_params = self.clean_model_params
        self.aggregator.model.load_state_dict(clean_model_params)
        self.after_attack_countdown -= 1
        return client_indexes, clean_model_params

    def sa_normal_agg(self, target_acc, client_indexes):
        # under attack
        if self.round_freezing:
            wandb.log({
                "Round": self.args.attack_afterward - self.after_attack_countdown,
                f"Target/{self.round_idx}-Acc": target_acc
            })
        # no attack
        else:
            wandb.log({
                "Round": self.round_idx,
                f"Target/Clean-Acc": target_acc
            })
        self.after_attack_countdown -= 1
        return client_indexes, self.aggregator.get_global_model_params()

    def sa_start_attack(self, target_acc, client_indexes):
        if not self.round_freezing:
            wandb.log({
                "Round": self.round_idx,
                f"Target/Clean-Acc": target_acc
            })
        self.logger.info("---------- Start Attack ----------")
        self.round_freezing = True
        self.after_attack_countdown = self.args.attack_afterward
        client_indexes[0] = -1 # mark attack
        self.clean_model_params = deepcopy(self.aggregator.get_global_model_params())
        return client_indexes, self.aggregator.get_global_model_params()

    # no attack
    def none_agg(self, target_acc, client_indexes):
        wandb.log({
            "Round": self.round_idx,
            f"Target/Clean-Acc": target_acc
        })
        return client_indexes, self.aggregator.get_global_model_params()

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        self.logger.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            self.logger.info(f'training time cost {time.time() - self.timestamp}')
            self.timestamp = time.time()
            global_model_params = self.aggregator.aggregate(self.args.attack_afterward == self.after_attack_countdown)

            # measure the target task accuracy
            final_acc = self.aggregator.test_target_accuracy(self.round_idx)
            if self.args.attack_afterward == self.after_attack_countdown: # attack just start, get attacker model to test acc
                attacker_acc = self.aggregator.test_target_accuracy_on_local_params(0)
                wandb.log({
                    "Round": self.round_idx,
                    "Target/Attacker_Acc": attacker_acc
                })
            # wandb.log({
            #     "Round": self.round_idx,
            #     "Target/Acc": final_acc
            # })

            # start the next round
            # sampling clients
            client_indexes = self.aggregator.client_sampling(
                self.round_idx,
                self.args.client_num_in_total,
                self.args.client_num_per_round
            )
            if not self.round_freezing:
                self.round_idx += 1
                if self.round_idx == self.round_num:
                    self.finish()
                    return

            if self.args.attack_type == 'none':
                next_round_gen = self.none_agg
            elif self.args.attack_type == 'single_shot':
                if self.after_attack_countdown == 0:
                    next_round_gen = self.sa_restore_clean_model
                elif self.round_idx in self.attack_round_idxs and (not self.round_freezing):
                    next_round_gen = self.sa_start_attack
                else:
                    next_round_gen = self.sa_normal_agg
            else: # repeated
                raise NotImplementedError()
            client_indexes, global_model_params = next_round_gen(final_acc, client_indexes)

            print("size = %d" % self.size)
            if self.args.is_mobile == 1:
                print("transform_tensor_to_list")
                global_model_params = transform_tensor_to_list(global_model_params)

            self.logger.info(f'aggregate+test time cost {time.time() - self.timestamp}')
            self.timestamp = time.time()
            round_idx = self.round_idx if (not self.round_freezing) else (self.round_idx + self.self.args.attack_afterward - self.after_attack_countdown)
            require_test = (not self.round_freezing) and ((self.round_idx % self.args.frequency_of_the_test) == 0)
            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id,
                    global_model_params,
                    client_indexes[receiver_id-1],
                    round_index=round_idx,
                    require_test=require_test,
                    during_attack=self.round_freezing
                )

    def handle_message_receive_test_result_from_client(self, msg_params):
        round_idx = msg_params.get(MyMessage.MSG_ARG_KEY_ROUNDIDX)
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        result = msg_params.get(MyMessage.MSG_ARG_KEY_TEST_RESULT)
        try:
            self.test_result[round_idx][sender_id] = result
        except KeyError:
            self.test_result[round_idx] = {}
            self.test_result[round_idx][sender_id] = result
        # self.logger.info(f"Round-{round_idx}, {result}")
        if len(self.test_result[round_idx]) == self.args.client_num_per_round:
            train_total_correct, train_total_loss, train_total_num, \
            test_total_correct, test_total_loss, test_total_num = 0, 0, 0, 0, 0, 0
            for _, result in self.test_result[round_idx].items():
                train_correct, train_loss, train_num, \
                    test_correct, test_loss, test_num = result
                train_total_correct += train_correct
                train_total_loss += train_loss
                train_total_num += train_num
                test_total_correct += test_correct
                test_total_loss += test_loss
                test_total_num += test_num
            train_all_acc = train_total_correct / train_total_num
            train_all_loss = train_total_loss / train_total_num
            test_all_acc = test_total_correct / test_total_num
            test_all_loss = test_total_loss / test_total_num
            self.logger.info(
                f"Round-{round_idx} Train Acc: {train_all_acc}, Train Loss: {train_all_loss}, Train Num: {train_total_num}"
            )
            self.logger.info(
                f"Round-{round_idx} Test Acc: {test_all_acc}, Test Loss {test_all_loss}, Test Num: {test_total_num}"
            )
            wandb.log({
                "Round": round_idx,
                "Train/Acc": train_all_acc,
                "Train/Loss": train_all_loss,
                "Test/Acc": test_all_acc,
                "Test/Loss": test_all_loss
            })

    def send_message_sync_model_to_client(
        self,
        receive_id, 
        global_model_params,
        client_index,
        round_index=None,
        require_test=False,
        during_attack=False
    ):
        if round_index == None:
            round_index = self.round_idx
        self.logger.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUNDIDX, round_index)
        message.add_params(MyMessage.MSG_ARG_KEY_REQUIRE_TEST, require_test)
        message.add_params(MyMessage.MSG_ARG_KEY_ATTACKING, during_attack)
        self.send_message(message)