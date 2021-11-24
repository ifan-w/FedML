import logging

try:
    from fedml_api.distributed.fedavg_robust.message_define import MyMessage
    from fedml_api.distributed.fedavg.utils import transform_list_to_tensor
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_api.distributed.fedavg_robust.message_define import MyMessage
    from FedML.fedml_api.distributed.fedavg.utils import transform_list_to_tensor
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message


class FedAvgRobustClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.logger = args.logger

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.handle_message_receive_model_from_server
        )                                    

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        self.logger.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        round_idx = msg_params.get(MyMessage.MSG_ARG_KEY_ROUNDIDX)
        require_test = msg_params.get(MyMessage.MSG_ARG_KEY_REQUIRE_TEST)
        self.round_idx = round_idx

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)

        # test and send result for new global model
        if require_test:
            self.__test()

        # local train
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        self.__train()

        # if all round finish, skip training, wait for server finish
        if self.round_idx == self.num_rounds:
            return
            self.finish() # don't send finish by worker as it would stop the last test round

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)
    
    def send_test_result_to_server(self, receive_id, test_result, round_idx):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_TEST_RESULT_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_RESULT, test_result)
        message.add_params(MyMessage.MSG_ARG_KEY_ROUNDIDX, round_idx)
        self.send_message(message)

    def __train(self):
        self.logger.info("#######training########### round_id = %d" % self.round_idx)
        weights, local_sample_num = self.trainer.train(round_idx=self.round_idx)
        self.send_model_to_server(0, weights, local_sample_num)

    def __test(self):
        # if not test round, skip
        # if (self.round_idx + 1) % self.args.frequency_of_the_test == 0 or self.round_idx == self.args.comm_round - 1:
        self.logger.info("#######testing########### round_id = %d" % self.round_idx)
        test_result = self.trainer.test()
        self.send_test_result_to_server(0, test_result, self.round_idx)
