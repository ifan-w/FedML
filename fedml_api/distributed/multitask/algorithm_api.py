from mpi4py import MPI

from fedml_api.distributed.multitask.decentralized_worker import DecentralizedWorker
from fedml_api.distributed.multitask.decentralized_worker_manager import DecentralizedWorkerManager
from fedml_core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_Decentralized_MTL_distributed(process_id, worker_number, device, comm,
                 model, train_data_num, train_data_global, test_data_global,
                 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args):
    # initialize the topology (ring)
    tpmgr = SymmetricTopologyManager(worker_number, worker_number)
    tpmgr.generate_topology()
    # logging.info(tpmgr.topology)

    # initialize the decentralized trainer (worker)
    worker_index = process_id
    trainer = DecentralizedWorker(worker_index, tpmgr, train_data_local_dict, test_data_local_dict,
                                  train_data_local_num_dict, train_data_num, device, model, args)

    client_manager = DecentralizedWorkerManager(args, comm, process_id, worker_number, trainer, tpmgr)
    client_manager.run()
