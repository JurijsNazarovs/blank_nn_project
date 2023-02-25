import numpy as np
import os
import torch
import subprocess
import matplotlib.pyplot as plt
import pickle as pkl


def printline():
    print("--------------------")


def save_model(args, model, optim, ckpt_path, epoch=0, best_loss=np.infty):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    torch.save(
        {
            'args': args,
            'state_dict': model.state_dict(),
            'optimizer_state': optim.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        }, ckpt_path)


def load_model(args, ckpt_path, model, optim, device):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    else:
        print("Loading model from %s" % ckpt_path)

    # Load checkpoint
    checkpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpt['args']  # Not used
    state_dict = checkpt['state_dict']
    model_dict = model.state_dict()
    optim_state = checkpt['optimizer_state']
    epoch_st = checkpt['epoch']
    best_loss = checkpt['best_loss'] if 'best_loss' in checkpt.keys()\
        else np.infty

    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # 3. load the new state dict
    model.load_state_dict(state_dict)
    model.to(device)

    # Load optimizer
    optim.load_state_dict(optim_state)
    return epoch_st, best_loss, model


def dump_pickle(data, filename):
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        filecontent = pickle.load(pkl_file)
    return filecontent


def compute_running_loss(running_loss, curr_loss, step):
    running_loss = (running_loss * (step - 1) + curr_loss) / step
    return running_loss


def get_lr(optimizer):
    lr = [group['lr'] for group in optimizer.param_groups]

    return lr


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


class LinearScheduler(object):
    # kl schedular
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


def plot_grad_flow(named_parameters, fname='tmp.png'):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.figure(figsize=(10, 25))
    plt.rcParams['xtick.labelsize'] = 8
    #plt.tight_layout()
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(fname)
    plt.close()


def plot_weights(weights, fname='weight.png'):
    #ave_grads = []
    layers = []
    plt.figure(figsize=(10, 25))
    plt.rcParams['xtick.labelsize'] = 8
    #plt.tight_layout()
    plt.plot(weights.data.cpu(), alpha=0.3, color="b")
    plt.hlines(0, 0, len(weights) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(weights), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(weights))
    plt.xlabel("Weights")
    plt.ylabel("Weights value")
    plt.title("Weights value")
    plt.grid(True)
    plt.savefig(fname)
    plt.close()


def warnmsg(msg):
    print("******************************")
    print("Warning:")
    print(msg)
    print("******************************")


def get_gpu_memory_map(device_ids=None):
    """Get the current gpu usage.
    device_ids can be a list of specified ids to return
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ],
                                     encoding='utf-8')

    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        vis_ids = [
            int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        ]
        gpu_memory = [gpu_memory[i] for i in vis_ids]

    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    if device_ids is not None:
        gpu_memory_map = {i: gpu_memory_map[i] for i in device_ids}

    return gpu_memory_map
