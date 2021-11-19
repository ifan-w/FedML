class MyMessage(object):
    """
        message type definition
    """
    # model distribute & aggregation
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3

    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4

    # test result
    MSG_TYPE_C2S_SEND_TEST_RESULT_TO_SERVER = 5

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_REQUIRE_TEST = "require_test"
    MSG_ARG_KEY_TEST_RESULT = "test_result"
    MSG_ARG_KEY_ROUNDIDX = "round_index"
    MSG_ARG_KEY_ATTACKING = "during_attack" # for testing after single-shot attack
