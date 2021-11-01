import logging
import socket

import torch
import yaml


def mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, gpu_util_file, gpu_util_key, logger=None):
    if logger == None:
        logger = logging
    if gpu_util_file == None:
        device = torch.device("cpu")
        logger.info(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.info(" ################## You do not indicate gpu_util_file, will use CPU training  #################")
        logger.info(device)
        # return gpu_util_map[process_id][1]
        return device
    else:
        with open(gpu_util_file, 'r') as f:
            gpu_util_yaml = yaml.load(f, Loader=yaml.FullLoader)
            # gpu_util_num_process = 'gpu_util_' + str(worker_number)
            # gpu_util = gpu_util_yaml[gpu_util_num_process]
            gpu_util = gpu_util_yaml[gpu_util_key]
            logger.info("gpu_util = {}".format(gpu_util))
            gpu_util_map = {}
            i = 0
            for host, gpus_util_map_host in gpu_util.items():
                for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                    for _ in range(num_process_on_gpu):
                        gpu_util_map[i] = (host, gpu_j)
                        i += 1
            logger.info("Process %d running on host: %s, gethostname: %s, local_gpu_id: %d ..." % (
                process_id, gpu_util_map[process_id][0], socket.gethostname(), gpu_util_map[process_id][1]))
            logger.info("i = {}, worker_number = {}".format(i, worker_number))
            assert i == worker_number
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_util_map[process_id][1])
        logger.info("setting device")
        device = torch.device("cuda:" + str(gpu_util_map[process_id][1]) if torch.cuda.is_available() else "cpu")
        logger.info("process_id = {}, GPU device = {}".format(process_id, device))
        # return gpu_util_map[process_id][1]
        return device
