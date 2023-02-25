import os
import sys
import matplotlib
from torch.utils.tensorboard import SummaryWriter

matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import datetime
import numpy as np
import pandas as pd
from random import SystemRandom

import torch
import torch.nn as nn
import torch.optim as optim
import pickle

import lib.utils as utils
from configs import get_arguments

from tqdm import trange


class Logger(object):

    def __init__(self, fname='logs/output.log'):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.log = open(fname, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def train(
    args,
    data_loader,
    model,
    optimizer,
    epoch,
    best_loss=None,
    tb_writer=None,
):
    # with torch.cuda.device(args.device):
    #     torch.cuda.empty_cache()
    model.train()
    n_batches = len(data_loader)

    start_time = time.time()
    avg_loss = 0
    avg_batch_time = 0
    for itr, (x, y) in enumerate(data_loader):  #itr in range(n_batches):
        # if itr > 2:
        #     break
        # if itr == 0:
        #     print(x[-1, -1])
        #     print(y[-1])
        start_time_batch = time.time()
        print("Training: Batch: %04d/%04d; " % (itr + 1, n_batches), end='')
        #x, y = data_obj["train_dataloader"].__next__()

        y_hat = model(x)
        loss = model.compute_loss(y_hat, y)
        avg_loss = utils.compute_running_loss(avg_loss, loss, itr + 1)
        print("Loss: %.8f, Running Loss: %.8f" %\
              (loss.item(), avg_loss),
              end='\r')
        optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        # torch.nn.utils.clip_grad_norm_(sel_params, 1.)
        # utils.plot_grad_flow(model.named_parameters())

        optimizer.step()

        # Save every batch
        utils.save_model(args, model, optimizer, args.ckpt_path + "_batch",
                         epoch)

        if args.check_memory_time:
            occupied_memory = utils.get_gpu_memory_map()
            print("Occupied GPU memory: ", occupied_memory)
            end_time = time.time()
            avg_batch_time = (avg_batch_time * itr +
                              (end_time - start_time_batch)) / (itr + 1)
            print("Batch time: %f" % avg_batch_time)
            if itr > 10:
                exit(1)

    end_time = time.time()
    print("\nEpoch time: %f" % ((end_time - start_time)))

    # Save every epochs
    utils.save_model(args, model, optimizer, args.ckpt_path, epoch, best_loss)

    if best_loss is None:
        best_loss = np.infty

    # Write loss to tb
    if tb_writer is not None:
        tb_writer.add_scalar('train_avg_loss', avg_loss, epoch)

    return best_loss


def valid(args,
          data_loader,
          model,
          optimizer,
          epoch,
          best_loss=None,
          tb_writer=None,
          data_obj=None):
    # with torch.cuda.device(args.device):
    #     torch.cuda.empty_cache()
    model.eval()
    avg_loss = 0

    n_batches = len(data_loader)
    for itr, (x, y) in enumerate(data_loader):  #itr in range(n_batches):
        print("Validation: Batch: %04d/%04d; " % (itr + 1, n_batches), end='')

        y_hat = model(x)
        loss = model.compute_loss(y_hat, y)

        avg_loss = utils.compute_running_loss(avg_loss, loss, itr + 1)
        print("Loss: %.8f, Running Loss: %.8f" %\
              (loss.item(), avg_loss),
              end='\r')
    print()
    ## Save model with best loss
    if best_loss is not None:
        if avg_loss <= best_loss:
            best_loss = avg_loss
            print("New best loss: %.8f" % best_loss)
            utils.save_model(args, model, optimizer, args.ckpt_path + '_best',
                             epoch, best_loss)
            # Write a result in the file
            with open(args.best_loss_path, 'w') as f:
                f.write("Epoch:%03d,loss:%.8f" % (epoch, best_loss))
        else:
            print("Best loss still the same: %.8f" % best_loss)

    # Write loss to tb
    if tb_writer is not None:
        tb_writer.add_scalar('valid_avg_loss', avg_loss, epoch)

    return best_loss


def test(args, model, data_obj):
    '''
    data_obj is important to conduct right normalization for test data
    data_obj contains infromation about how to normalize test data.
    '''
    # Perform prediction and fill the file
    model.eval()
    prediction_file_path = "%s/%s_yhat.csv" % (args.save_dir,
                                               args.experiment_id)
    # Fill the header from yhat to our prediction file
    with open(args.test_data_path, 'r') as f:
        header = f.readline()

    prediction_file_writer = open(prediction_file_path, 'w')
    prediction_file_writer.write(header)

    # Make predictions and write to the file
    for itr, (x, t) in enumerate(zip(xs, ts)):
        # We do not use Batch here on purpose but use a single sample,
        # to be able to test with methods, independent of time steps.
        x = torch.tensor(x, device=args.device)[None]  # add batch to x
        y_hat = model(x)

        # Transform y_hat back to original scale
        y_hat = y_hat.squeeze().data.cpu().numpy()
        y_hat = data_obj.reverse_transform(y_hat, xs_raw[itr])

        # Create a record with time and prediciton
        str_write = np.concatenate([t, y_hat])
        str_write = ','.join([str(i) for i in str_write]) + '\n'
        prediction_file_writer.write(str_write)

    prediction_file_writer.close()
    print("File with results was saved in %s" % prediction_file_path)


################################################################################
## Below is the main function, which executes training/testing/inference
################################################################################
if __name__ == '__main__':

    ## Getting args, establishing experiments ID, related logs pathes, ...
    parser = get_arguments()
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = "cuda:%d" % args.device
    else:
        device = 'cpu'  #torch.device('cpu')

    print("Device:", device)
    args.device = device
    os.makedirs(args.save_dir, exist_ok=True)

    experiment_id = args.experiment_id + args.note
    if experiment_id is None:
        experiment_id = int(SystemRandom().random() * 100000)
    args.ckpt_path = os.path.join(args.save_dir,
                                  "experiment_" + str(experiment_id) + '.ckpt')
    sys.stdout = Logger(fname="%s/%s.log" % (args.logs_dir, experiment_id))
    print("Input command: %s " % " ".join(sys.argv))

    # To save best loss result for future plots
    os.makedirs("%s/best_loss/" % args.logs_dir, exist_ok=True)
    args.best_loss_path = '%s/best_loss/%s.csv' % (args.logs_dir,
                                                   args.experiment_id)

    # Tensorboard writer
    os.makedirs("%s/tb/" % args.logs_dir, exist_ok=True)
    tb_writer = SummaryWriter('%s/tb/%s' % (args.logs_dir, experiment_id))

    # Seed for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    ## Load Data
    data_obj = parse_datasets(args)
    args.input_dim = data_obj["input_dim"]
    args.output_dim = data_obj["output_dim"]

    ## Create model
    model = None  #put your model here

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.96)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=.1,
        patience=10,
        threshold=0.0001,
        threshold_mode='rel',
        cooldown=10,  #how long to do nothing after step
        min_lr=1e-5)

    ## Load checkpoint and evaluate the model
    if args.load:
        # In case we load model to contrinue from last epoch
        if args.best:
            ckpt_path_load = args.ckpt_path + "_best"
        elif args.batch:
            ckpt_path_load = args.ckpt_path + "_batch"
        else:
            ckpt_path_load = args.ckpt_path
        epoch_st, best_loss, model = utils.load_model(args, ckpt_path_load,
                                                      model, optimizer, device)

        print("Current best loss: %.8f, achieved at epoch: %d" %
              (best_loss, epoch_st))
        epoch_st += 1
    else:
        epoch_st, best_loss = 1, np.infty
    ##################################################################
    if args.reset_lr:
        utils.warnmsg("Learning Rate was reset to: %f" % args.lr)
        utils.set_lr(optimizer, args.lr)

    if args.test_only:
        with torch.no_grad():
            test(args, model, data_obj["data_train_object"])
    else:
        prev_lr = utils.get_lr(optimizer)  # to reload best model

        for epoch in range(epoch_st, args.n_epochs + 1):
            print('Epoch %04d' % epoch)
            try:
                print('lr: ', utils.get_lr(optimizer))
            except:
                pass

            # Training
            best_loss = train(
                args,
                data_obj["data_loader_train"],
                model,
                optimizer,
                epoch=epoch,
                best_loss=best_loss,
                tb_writer=tb_writer,
            )

            # Do validation and report summary
            if epoch % args.n_epochs_to_viz == 0 and\
               epoch >= args.n_epochs_start_viz:
                with torch.no_grad():
                    best_loss = valid(args,
                                      data_obj["data_loader_valid"],
                                      model,
                                      optimizer,
                                      epoch,
                                      best_loss=best_loss,
                                      tb_writer=tb_writer,
                                      data_obj=data_obj["data_train_object"])

                lr_scheduler.step(best_loss)
                # Reload optimizer and best model
                curr_lr = utils.get_lr(optimizer)
                if prev_lr != curr_lr:
                    prev_lr = curr_lr
                    _, _, model = utils.load_model(args,
                                                   args.ckpt_path + "_best",
                                                   model, optimizer, device)
                    # Because loaded optimizer has old LR, we need to update
                    utils.set_lr(optimizer, curr_lr[0])

            #lr_scheduler.step()  #update lr
            #print("lr:", lr_scheduler.get_last_lr())
            #print([group['lr'] for group in optimizer.param_groups])
            #print("lr:", lr_scheduler._last_lr)
