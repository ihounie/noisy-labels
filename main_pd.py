# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse
import wandb
import numpy as np
from utils.utils import setup_gpus


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--dual_lr', type = float, help="dual learning rate", default=1)
parser.add_argument('--perturbation_lr', type = float, help="perturbation (u) learning rate", default=1)
parser.add_argument('--penalization', type = str, help="huber, l2 or linear", default="huber")
parser.add_argument('--huber_delta', type = float, help="huber delta parameter", default=0.2)
parser.add_argument('--huber_a', type = float, help="huber quadratic parameter", default=10)
parser.add_argument('--grad_clip', type = float, help="gradient clipping", default=1.0)
parser.add_argument('--epsilon', type = float, help="constraint level", default=0.6)
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--wandb_log', action='store_true', default=False)
parser.add_argument('--num_log', type=int, default=200, help="how many lambdas/u's to wandb")
parser.add_argument('--project', type = str, help="wandb project name", default="CIFARN")
parser.add_argument('--run', type = str, help="wandb run name", default="Resilient")
parser.add_argument('--is_human', action='store_true', default=False)

class Huber_loss:
    def __init__(self, delta=0.5, a=1.0):
        self.delta=delta
        self.slope = 1
        self.a = a
    def eval(self, u):
        return self.a*0.5*(torch.abs(u)<= self.delta)*u**2 + self.a*self.delta*(u>self.delta)(torch.abs(u)-0.5*self.delta)

    def grad(self, u):
        return self.a*(torch.abs(u)<=self.delta) * u + self.a*self.delta * (u>self.delta)

def log_histogram(metric, key):
    np_hist = metric, np.arange(len(metric)+1)
    wandb.log({f"{key} values": wandb.Histogram(np_histogram=np_hist)})
    wandb.log({f"{key} histogram": wandb.Histogram(metric)})
    return


# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(epoch, train_loader, model, optimizer, lambdas, u):
    train_total=0
    train_correct=0

    for i, (images, labels, indexes) in enumerate(train_loader):
        batch_size = len(images)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        indexes = Variable(indexes).cuda()
        # Forward + Backward + Optimize
        logits = model(images)
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total+=1
        train_correct+=prec
        loss = F.cross_entropy(logits, labels, reduce = True)
        slacks = loss - u[indexes] - args.epsilon
        lagrangian = torch.sum(lambdas[indexes]*slacks)
        # primal update
        lagrangian.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        try:
            assert(not torch.isnan(lagrangian))
        except:
            print(loss)
            print(slacks)
            print(lambdas[indexes])
            assert(0)
        optimizer.zero_grad()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))

    train_acc=float(train_correct)/float(train_total)
    return train_acc

# Update duals
def dual_update(train_loader, model, lambdas, u, h):
    model.eval()
    with torch.no_grad():
        for i, (images, labels, indexes) in enumerate(train_loader):
            ind=indexes.cpu().numpy().transpose()
        
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
        
            # Forward + Backward + Optimize
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduce = True)
            slacks = loss - u[indexes] - args.epsilon
            # perturbation update
            u[indexes] = F.relu(u[indexes]-args.perturbation_lr*(h.grad(u[indexes])-lambdas[indexes]))
            # Dual Update
            lambdas[indexes] = F.relu(lambdas[indexes]+args.dual_lr*slacks)
    return lambdas, u
# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc

########################
# choose and config GPU
########################
best_gpu = setup_gpus()
torch.cuda.set_device(best_gpu)
torch.backends.cudnn.benchmark = True

#####################################main code ################################################
args = parser.parse_args()
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#wandb logging
if args.wandb_log:
    wandb.init(project=args.project, entity="hounie", name=args.run)
    wandb.config.update(args)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else: 
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human)
noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
if args.wandb_log:
    indexes = np.arange(len(train_dataset))
    log_idxs = {}
    log_idxs["noisy"] = indexes[noise_or_not][:args.num_log]
    log_idxs["clean"] = indexes[~noise_or_not][:args.num_log]
print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])
# load model
print('building model...')
model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = 128,
                                   num_workers=args.num_workers,
                                   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 64,
                                  num_workers=args.num_workers,
                                  shuffle=False)
alpha_plan = [0.1] * 60 + [0.01] * 40
model.cuda()
print('Initialising Dual Variables & Perturbation...')
lambdas = torch.ones(num_training_samples).cuda()/128
u = torch.ones(num_training_samples).cuda()
epoch=0
train_acc = 0
if args.penalization == "huber":
    h = Huber_loss(delta=args.huber_delta, a=args.huber_a)
else:
    raise NotImplementedError
# training
noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
# train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    model.train()
    train_acc = train(epoch, train_loader, model, optimizer, lambdas, u)
    # Update duals
    lambdas, u = dual_update(train_loader, model, lambdas, u, h)
    # evaluate models
    test_acc = evaluate(test_loader=test_loader, model=model)
    # save results
    print('train acc on train images is ', train_acc)
    print('test acc on test images is ', test_acc)
    if args.wandb_log:
        wandb.log({"train acc": train_acc, "test_acc": test_acc, "epoch":epoch})
        for array, name in zip([lambdas, u], ["dual variable", "perturbation"]):
            for mode, idxs in log_idxs.items():
                log_histogram(array.cpu().numpy()[idxs], f"{name} {mode}")


        
