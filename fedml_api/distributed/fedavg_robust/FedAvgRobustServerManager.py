import logging
import os
import time
from copy import deepcopy
from typing_extensions import final

from torch.distributed.distributed_c10d import send

from fedml_api.distributed.fedavg_robust.message_define import MyMessage
from fedml_api.distributed.fedavg_robust.FedAvgRobustServerScheduler import FedAvgRobustServerScheduler
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
        self.clean_model_params = None

        self.scheduler = FedAvgRobustServerScheduler(self, args)

    def run(self):
        super().run()

    def send_init_msg(self):
        self.scheduler.step()
        self.round_idx = self.scheduler.round_idx
        client_indexes = self.scheduler.client_idx
        global_model_params = self.aggregator.get_global_model_params()
        
        # set timestamp
        self.timestamp = time.time()
        for process_id in range(1, self.size):
            self.send_message_sync_model_to_client(process_id, global_model_params, client_indexes[process_id-1], round_index=self.round_idx, require_test=False)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_TEST_RESULT_TO_SERVER,
            self.handle_message_receive_test_result_from_client
        )

    def save_model(self, path):
        self.aggregator.save_model(path)

    def load_model(self, path):
        self.aggregator.load_model(path)
    
    def gen_client_idx(self, round):
        return self.aggregator.client_sampling(
                round,
                self.args.client_num_in_total,
                self.args.client_num_per_round
            )
    
    def get_target_acc(self):
        return self.aggregator.test_target_accuracy(self.round_idx)

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
            global_model_params = self.aggregator.aggregate(self.scheduler.model_replacement)

            # after aggregation, one round finish, invoke scheduler to generate next round
            if not self.scheduler.step():   # return False, no more step
                self.finish()
                return
            self.round_idx = self.scheduler.round_idx

            client_indexes = self.scheduler.client_idx
            global_model_params = self.aggregator.get_global_model_params()

            if self.args.is_mobile == 1:
                print("transform_tensor_to_list")
                global_model_params = transform_tensor_to_list(global_model_params)

            round_idx = self.round_idx
            require_test = self.scheduler.require_test
            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id,
                    global_model_params,
                    client_indexes[receiver_id-1],
                    round_index=round_idx,
                    require_test=require_test
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
        require_test=False
    ):
        if round_index == None:
            round_index = self.round_idx
        self.logger.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUNDIDX, round_index)
        message.add_params(MyMessage.MSG_ARG_KEY_REQUIRE_TEST, require_test)
        self.send_message(message)