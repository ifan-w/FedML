import logging
import traceback
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s - %(name)-9s : %(filename)s-%(lineno)s: %(message)s",
    filename="output.log",
    filemode="a"
)
import argparse
from functools import partial
import os
import random
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
import wandb

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
try:
    from fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

    from fedml_api.distributed.fedavg_robust.FedAvgRobustAPI import FedML_init, FedML_FedAvgRobust_distributed

    from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
    from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg

    from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
    from fedml_api.model.linear.lr import LogisticRegression

    from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
    from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
    from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
    from fedml_api.model.cv.mobilenet import mobilenet
    from fedml_api.model.cv.resnet import resnet56
    from fedml_api.model.cv.resnet_cifar import ResNet18

    # for loading poisoned dataset
    from fedml_api.data_preprocessing.edge_case_examples.data_loader import load_poisoned_dataset
except ImportError:
    from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

    from FedML.fedml_api.distributed.fedavg_robust.FedAvgRobustAPI import FedML_init, FedML_FedAvgRobust_distributed

    from FedML.fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
    from FedML.fedml_api.model.nlp.rnn import RNN_OriginalFedAvg

    from FedML.fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
    from FedML.fedml_api.model.linear.lr import LogisticRegression

    from FedML.fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
    from FedML.fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
    from FedML.fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
    from FedML.fedml_api.model.cv.mobilenet import mobilenet
    from FedML.fedml_api.model.cv.resnet import resnet56
    from FedML.fedml_api.model.cv.resnet_cifar import ResNet18

    # for loading poisoned dataset
    from FedML.fedml_api.data_preprocessing.edge_case_examples.data_loader import load_poisoned_dataset

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument(
        '--model', type=str, default='mobilenet', metavar='N',
        help='neural network used in training'
    )

    parser.add_argument(
        '--dataset', type=str, default='cifar10', metavar='N',
        help='dataset used for training'
    )

    parser.add_argument(
        '--data_dir', type=str, default='./../../../data/cifar10',
        help='data directory'
    )

    parser.add_argument(
        '--partition_method', type=str, default='hetero', metavar='N',
        help='how to partition the dataset on local workers'
    )

    parser.add_argument(
        '--partition_alpha', type=float, default=0.5, metavar='PA',
        help='partition alpha (default: 0.5)'
    )

    parser.add_argument(
        '--client_num_in_total', type=int, default=1000, metavar='NN',
        help='number of workers in a distributed cluster'
    )

    parser.add_argument(
        '--client_num_per_round', type=int, default=4, metavar='NN',
        help='number of workers'
    )
    
    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default="gpu_mapping.yaml",
        help="the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key", type=str, default="mapping_default", help="the key in gpu utilization file"
    )
    
    parser.add_argument(
        '--no_cuda', type=bool, default=False,
        help='disable cuda'
    )
    parser.add_argument(
        '--is_mobile', type=int, default=0,
        help='whether the program is running on the FedML-Mobile server side'
    )

    # scheduler-only
    parser.add_argument(
        '--comm_round', type=int, default=10,
        help='how many round of communications we shoud use'
    )
    parser.add_argument(
        '--frequency_of_the_test', type=int, default=1,
        help='the frequency of the algorithms'
    )
    parser.add_argument(
        '--save_model_freq', type=int, default=0,
        help='frequency of saving model, <=0 means do not save'
    )
    parser.add_argument(
        '--load_model_path', type=str, default='',
        help='the path to load model params, override "--save_model_freq"'
    )

    # defending
    parser.add_argument(
        '--defense_type', type=str, default='weak_dp', metavar='N',
        help='the robust aggregation method to use on the server side. norm_diff_clipping, weak_dp, none'
    )

    parser.add_argument(
        '--norm_bound', type=float, default=30.0, metavar='N',
        help='the norm bound of the weight difference in norm clipping defense.'
    )

    parser.add_argument(
        '--stddev', type=float, default=0.025, metavar='N',
        help='the standard deviation of the Gaussian noise added in weak DP defense.'
    )

    #parser.add_argument('--attack_method', type=str, default="blackbox",
    #                    help='describe the attack type: blackbox|pgd|graybox|no-attack|')

    # attacker argument
    # attack type
    parser.add_argument(
        '--attack_type', type=str, default='single_shot',
        help='single_shot or repeated'
    )
    parser.add_argument(
        '--attack_freq', type=int, default=10,
        help='a single adversary per X federated learning rounds e.g. 10 means there will be an attacker in each 10 FL rounds.'
    )
    parser.add_argument(
        '--attack_afterward', type=int, default=10,
        help='for single_shot, how many rounds normal train after attack'
    )

    # attacker data load
    parser.add_argument(
        '--poison_type', type=str, default='southwest',
        help='specify source of data poisoning: |ardis|(for EMNIST), |southwest|howto|(for CIFAR-10)'
    )
    parser.add_argument(
        '--poison_frac', type=float, default=0.5,
        help='frac of poison in poisoned data'
    )
    parser.add_argument(
        '--attack_case', type=str, default='edge-case',
        help='attack case'
    )
    parser.add_argument(
        '--attack_num', type=int, default=1,
        help='number of attacker'
    )
    # attacker training procedure
    parser.add_argument(
        '--attack_epochs', type=int, default=5,
        help='how many epochs attacker will do'
    )
    parser.add_argument(
        '--attack_loss_threshold', type=float, default=0.01,
        help='loss threshold of attacker'
    )
    parser.add_argument(
        '--attack_acc_threshold', type=float, default=60,
        help='accuracy threshold of attacker'
    )
    parser.add_argument(
        '--attack_lr', type=float, default=0.001,
        help='attacker learning rate'
    )
    parser.add_argument(
        '--attack_optimizer', type=str, default='sgd',
        help='attacker optimizer'
    )


    # TODO(hwang): we will add PGD attack soon, stay tuned!
    #parser.add_argument('--adv_lr', type=float, default=0.02,
    #                   help='learning rate for adv in PGD setting')

    # normal client argument
    parser.add_argument(
        '--batch_size', type=int, default=64, metavar='N',
        help='input batch size for training (default: 64)'
    )

    parser.add_argument(
        '--lr', type=float, default=0.001, metavar='LR',
        help='learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--epochs', type=int, default=5, metavar='EP',
        help='how many epochs will be trained locally'
    )

    parser.add_argument(
        '--client_optimizer', type=str, default='sgd',
        help='SGD with momentum; adam'
    )

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0005)

    parser.add_argument('--note', help='note for this run', type=str, default='')
    parser.add_argument('--title', help='title for this run', type=str, default='Fedavg-Robust')

    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    # handle the normal data partition
    if dataset_name == "mnist":
        args.logger.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size)
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num
    elif dataset_name == "shakespeare":
        args.logger.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num
    else:
        if dataset_name == "cifar10":
            data_loader = partial(load_partition_data_cifar10, logger=args.logger) 
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    
    # handle the poisoned data loader
    #  load poisoned dataset
    poisoned_train_loader, targetted_task_test_loader, num_dps_poisoned_dataset = load_poisoned_dataset(args=args, dataset=dataset)

    return dataset, poisoned_train_loader, targetted_task_test_loader, num_dps_poisoned_dataset


def create_model(args, model_name, output_dim):
    args.logger.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        model = LogisticRegression(28 * 28, output_dim)
        args.client_optimizer = "sgd"
    elif model_name == "rnn" and args.dataset == "shakespeare":
        model = RNN_OriginalFedAvg(28 * 28, output_dim)
        args.client_optimizer = "sgd"
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "resnet18":
        model = ResNet18()
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    return model


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    args.logger.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    args.logger.info(device)
    return device

def get_logger(logger_name):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)-8s - %(name)-9s : %(filename)s-%(lineno)s: %(message)s"
    )
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    return logger

if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger = get_logger("Server" if process_id == 0 else f"Worker-{process_id-1}")
    if process_id == 0:
        logger.info(args)

    # customize the process name
    str_process_name = "FedAvgRobust (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)


    hostname = socket.gethostname()
    logger.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fedml-robust",
            # name="FedAVG(d)" + str(args.partition_method) + "r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(
            #     args.lr),
            name=args.title,
            config=args
        )

    setattr(args, 'logger', logger)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
    logger.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = mapping_processes_to_gpu_device_from_yaml_file(
        process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key, logger=logger
    )
    # load data
    logger.info('loading data')
    dataset, poisoned_train_loader, targetted_task_test_loader, num_dps_poisoned_dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])

    # start "robust federated averaging (FedAvg)"
    try:
        FedML_FedAvgRobust_distributed(
            process_id, worker_number, device, comm,
            model, train_data_num, train_data_global, test_data_global,
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
            poisoned_train_loader, targetted_task_test_loader, num_dps_poisoned_dataset, args
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e
