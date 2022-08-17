"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license 
"""
import os
import torch


### Original ops.py
def queue_data(data, k):
    return torch.cat([data, k], dim=0)


def dequeue_data(data, K=1024):
    if len(data) > K:
        return data[-K:]
    else:
        return data


def initialize_queue(model_k, device, train_loader, feat_size=128):
    queue = torch.zeros((0, feat_size), dtype=torch.float)
    queue = queue.to(device)

    for _, (data, _) in enumerate(train_loader):
        x_k = data[1]
        x_k = x_k.cuda(device)
        outs = model_k(x_k)
        k = outs['cont']
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K=1024)
        break
    return queue


### Original utils.py

class Logger:
    def __init__(self, experiment):
        self.last = None
        self.experiment

    def scalar_summary(self, tag, value, step):
        if self.last and self.last['step'] != step:
            print(self.last)
            self.last = None
        if self.last is None:
            self.last = {'step':step,'iter':step,'epoch':1}
        self.last[tag] = value

    def images_summary(self, tag, images, step, nrow=8):
        """Log a list of images."""
        self.viz.images(
            images,
            opts=dict(title='%s/%d' % (tag, step), caption='%s/%d' % (tag, step)),
            nrow=nrow
        )


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(state, name, log_path, epoch=0, experiment=None, conserve_space=True):
    file_name = f'{name}_model_{epoch:04d}.ckpt'
    save_file = log_path / file_name
    torch.save(save_dict, save_file)
    if experiment: experiment.log_model(name, save_file)
    if conserve_space:
        file_name = f'{name}_model_{int(epoch-1):04d}.ckpt'
        del_file = log_path / file_name
        del_file.unlink()

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def add_logs(args, logger, tag, value, step):
    logger.add_scalar(tag, value, step)
