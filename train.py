import argparse
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm
import wandb
import random
import time

from data import CIFData
from data import collate_pool
from utils import Normalizer, init_gbnn, AverageMeter, class_eval
from models.ensembles import BoostingNet
from models.cgcnn import GbGraphConvNet

parser = argparse.ArgumentParser(description='Gradient Boosting Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='Dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='Number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='Sets epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum') 
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument('--n_folds', default=5, type=int, metavar='N',
                    help='number of folds for cross validation')
parser.add_argument('--boosting_rate', default=0.25, type=float, metavar='N',
                    help='boosting rate for stable training')
parser.add_argument('--ensemble_method', default='gb', type=str, metavar='gb',
                    help='ensemble method choice : gb or bagging')

args = parser.parse_args()

args.cuda = not args.disable_cuda and torch.cuda.is_available()
best_mae_error = 0.        

# load data
dataset = CIFData(args.data_options, random_seed = args.random_seed)

k_folds = args.n_folds
kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=args.random_seed)
collate_fn = collate_pool

# obtain target value normalizer
if args.task == 'classification':
    normalizer = Normalizer(torch.zeros(2))
    normalizer.load_state_dict({'mean': 0., 'std': 1.})

# build model
orig_atom_fea_len = dataset[0][0].x.shape[1]
nbr_fea_len = dataset[0][0].edge_attr.shape[1]

N=len(dataset)
X = [dataset[i][0] for i in tqdm(range(N))]
y = [dataset[i][1].tolist() for i in range(N)]

for fold, (train_ids, test_ids) in enumerate(kfold.split(X, y)):        
    print(f"{fold} FOLD")
    print("--------------------------------------------------------------------------")
    c0 = init_gbnn(np.array(y)[train_ids])
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_ids)

    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.workers,
                              collate_fn=collate_pool, pin_memory=args.cuda)

    test_loader = DataLoader(dataset, batch_size=args.batch_size,
                            sampler=test_sampler,
                            num_workers=args.workers,
                            collate_fn=collate_pool, pin_memory=args.cuda)

    date = time.strftime("%Y%m%d-%H%M%S")
    name = f'{args.ensemble_method}_cv{fold}'
    wandb.init(project="GBCGCNN-MIT", name=name, reinit=True, group=args.ensemble_method)
    config = wandb.config
    config.batch_size =  args.batch_size
    config.epochs = args.epochs
    config.stages = args.num_nets
    config.lr = args.lr
    config.n_penul = args.n_penul
    config.imbalance_ratio = "0.01"
    config.random_seed = args.random_seed
    config.p = args.p

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    net_ensemble = BoostingNet(c0)
    loss_f1 = nn.MSELoss(reduction='none')
    loss_f2 = nn.BCEWithLogitsLoss(reduction='none')

    all_ensm_losses = []
    all_ensm_losses_te = []
    all_mdl_losses = []
    dynamic_br = []

    best_fscore = 0.
    p=args.p
    random_seed = args.random_seed

    for stage in range(args.num_nets):
        random_seed = random_seed + (stage+1) * (2*fold+1)
        t0 = time.time()
#         print('stage : ', stage)

        atom_fea_len = 64
        random.seed(random_seed)

        model = GbGraphConvNet.get_model(orig_atom_fea_len, nbr_fea_len, stage, args)
        if args.cuda:
            model.cuda()

        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), args.lr,
                                   weight_decay=args.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')

        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

        
        # model.train()
        stage_mdlloss=[]

        for epoch in range(args.start_epoch, args.epochs):
            # net_ensemble.to_train()
            model.train()
#             print(epoch)
            batch_time = AverageMeter()
            data_time = AverageMeter()

            end = time.time()
            y_true = []
            y_pred = []

            for i, (input, target, _) in enumerate(train_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                input = input.to('cuda', non_blocking=True)
                # input = input.requires_grad_(True)
                target = target.cuda()
                
                middle_feat, out = net_ensemble.forward(input)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                out = out * args.boosting_rate
                ### method 1
                h = 4/((1+torch.exp(-2*target*out))*(1+torch.exp(2*target*out)))
                grad_direction = target * (1.0 + torch.exp(-2 * target * out))/2
                _, out = model(input, middle_feat)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                loss = loss_f1(out, grad_direction)  # T
                loss = loss*h
                loss = loss.mean()

                model.zero_grad()
                loss.backward()
                optimizer.step()
                stage_mdlloss.append(loss.item()) 

                batch_time.update(time.time() - end)
                end = time.time()

            scheduler.step()

            if stage == 0:
                loss_tr_0, precision_tr_0, recall_tr_0, gscore_tr_0, fscore_tr_0 = class_eval(net_ensemble, train_loader, single=model)
                loss_te_0, precision_te_0, recall_te_0, gscore_te_0, fscore_te_0 = class_eval(net_ensemble, test_loader, single=model)
                
                wandb.log({
                   'Loss0@train':loss_tr_0, 'precision0@train':precision_tr_0, 'recall0@train':recall_tr_0, 'Gmean0@train':gscore_tr_0, 'Fscore0@train':fscore_tr_0, 
                   'Loss0@validation':loss_te_0, 'precision0@validation':precision_te_0, 'recall0@validation':recall_te_0, 'Gmean0@validation':gscore_te_0, 'Fscore0@validation':fscore_te_0})
                
                if epoch == args.epochs - 1:
                    wandb.init(project="GBCGCNN-MIT", name=name, reinit=True, group=args.ensemble_method)

        net_ensemble.add(model)
        sml = np.mean(stage_mdlloss)

        stage_loss = []
        lr_scaler = 2

        elapsed_tr = time.time()-t0

        if args.cuda:
            net_ensemble.to_cuda()

        # Training score
        loss_tr, precision_tr, recall_tr, gscore_tr, fscore_tr = class_eval(net_ensemble, train_loader)

        # Test score
        loss_te, precision_te, recall_te, gscore_te, fscore_te = class_eval(net_ensemble, test_loader)

        print(f'Stage - {stage}, training time: {elapsed_tr: .1f} sec')
        print(f'Loss@train : {loss_tr:.2f}, precision@train: {precision_tr:.2f}, recall@train: {recall_tr:.2f}, Gmean@train: {gscore_tr:.2f}, Fscore@train: {fscore_tr:.2f}')
        print(f'Loss@validation : {loss_te:.2f}, precision@validation: {precision_te:.2f}, recall@validation: {recall_te:.2f}, Gmean@validation: {gscore_te:.2f}, Fscore@validation: {fscore_te:.2f}')   

        wandb.log({
            'Loss@train':loss_tr, 'precision@train':precision_tr, 'recall@train':recall_tr, 'Gmean@train':gscore_tr, 'Fscore@train':fscore_tr, 
            'Loss@validation':loss_te, 'precision@validation':precision_te, 'recall@validation':recall_te, 'Gmean@validation':gscore_te, 'Fscore@validation':fscore_te,
            'training time':elapsed_tr})
        
        if best_fscore < fscore_te:
            best_fscore = fscore_te
            class_eval(net_ensemble, test_loader, to_file=name+"_best")

    wandb.finish()

    # store model
    out_file = os.getcwd() + f'/save_model/gbgnn/{name}.pth' 
    net_ensemble.to_file(out_file)