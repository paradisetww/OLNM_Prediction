import logging
import time
import os
import numpy as np
import math
import torch
from copy import deepcopy

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_logger(args):
    dataset = args.data_name
    log_dir = args.logdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = f"{dataset}_{time_str}.log"
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Args are set as follow--------------------")
    logger.info(args)
    logger.info("-------------------------------------------------------------")
    return logger, log_file


def adjust_learning_rate(args, optimizer, epoch, writer):
    # cosine attenuation method
    if epoch < args.n_epoch:
        lr = args.lr
    else:
        lr = args.lr * (1 + np.cos((epoch - args.n_epoch) * math.pi / args.max_epochs)) / 2

    # log to TensorBoard
    writer.add_scalar('learning_rate', lr, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer, writer


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
