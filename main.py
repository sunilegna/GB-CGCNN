import lightning.pytorch as pl
# from lightning.pytorch.loggers import WandbLogger
import os
import yaml
import wandb
import time

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from data import CIFData
from data import collate_pool
from utils import init_gbnn, class_eval
from models.ensembles import BoostingNet
from models.cgcnn import GbGraphConvNet

from torch.optim.lr_scheduler import MultiStepLR
from lightning.pytorch import seed_everything


class GradientBoosting(pl.LightningModule):
    def __init__(self, config, orig_atom_fea_len, nbr_fea_len, net_ensemble):
        super().__init__()
        self.config = config
        self.net_ensemble = net_ensemble
        self.model = GbGraphConvNet.get_model(orig_atom_fea_len, nbr_fea_len, config["model"])
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, x, m):
        return self.model(x, m)
        
    def training_step(self, batch):
        input, target, _ = batch
        
        self.model.train()
        middle_feat, out = self.net_ensemble.forward(input)
        out = torch.as_tensor(out, dtype=torch.float32).to(target.device).view(-1, 1)
        out = out * self.config["ensemble"]["boosting_rate"]
        
        h = 4/((1+torch.exp(-2*target*out))*(1+torch.exp(2*target*out)))
        grad_direction = target * (1.0 + torch.exp(-2 * target * out))/2
        _, out = self.model(input, middle_feat)
        out = torch.as_tensor(out, dtype=torch.float32).to(target.device).view(-1, 1)
        loss = self.mse_loss(out, grad_direction)  # T
        loss = loss*h
        loss = loss.mean()
        
        self.log("loss@train", loss, prog_bar=True, logger=True, batch_size=self.config["batch_size"])
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), config["lr"],
                        weight_decay=config["weight_decay"])
        
        scheduler = MultiStepLR(optimizer, milestones=config["lr_milestones"], gamma=0.1)
        return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
    
    

class Trainer(pl.LightningModule):
    def __init__(self, config, orig_atom_fea_len, nbr_fea_len, net_ensemble):
        super().__init__()

        self.gb = GradientBoosting(config, orig_atom_fea_len, nbr_fea_len, net_ensemble)
        
        # pl trainner
        # if config["wandb"]['wandb_on']:
        #     wandblogger = WandbLogger(project=config["wandb"]['project'], name=config["wandb"]['name'], group=config["wandb"]['group'])
        #     self.trainer = pl.Trainer(accelerator="gpu", devices=config["gpu_num"], max_epochs=config["model"]["epochs"], logger=wandblogger)
        # else:
        self.trainer = pl.Trainer(accelerator="gpu", devices=config["gpu_num"], max_epochs=config["model"]["epochs"], profiler='simple')
            
    def fit(self, train_loader):
        self.trainer.fit(self.gb, train_loader)
        
    
def train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path = config["data_path"]
    dataset_tar_file = dataset_path + ".tar"
    if not os.path.exists(dataset_path) and os.path.exists(dataset_tar_file):
        import tarfile
        print(f"Initalizing dataset from {os.path.abspath(dataset_tar_file)}")
        print(f"Unpacking dataset to {os.path.abspath(dataset_path)} ...")
        with tarfile.open(dataset_tar_file, "r") as tar:
            tar.extractall(path=dataset_path)
        print("Done!")
    dataset = CIFData(dataset_path, random_seed = config["random_seed"], radius = config["r_cut"], max_num_nbr = config["max_num_nbr"])

    k_folds = config["n_folds"]
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config["random_seed"])
    collate_fn = collate_pool

    # build model
    orig_atom_fea_len = dataset[0][0].x.shape[1]
    nbr_fea_len = dataset[0][0].edge_attr.shape[1]

    N=len(dataset)
    X = [dataset[i][0] for i in tqdm(range(N))]
    y = [dataset[i][1].tolist() for i in range(N)]
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X, y)):   
        mean_value = init_gbnn(np.array(y)[train_ids])  
        if config['ensemble']['c0'] == 'auto':  
            c0 = mean_value
        else:
            c0 = config['ensemble']['c0']

        print(f"c0:{c0}")
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(dataset, batch_size=config["batch_size"],
                                sampler=train_sampler,
                                num_workers=config["workers"], 
                                collate_fn=collate_pool, pin_memory=device)

        test_loader = DataLoader(dataset, batch_size=config["batch_size"],
                                sampler=test_sampler,
                                num_workers=config["workers"],
                                collate_fn=collate_pool, pin_memory=device)
        
        net_ensemble = BoostingNet(c0)
        name = f"cv{fold}"
        group = config["wandb"]["group"]
        config["wandb"]["name"] = name
        best_fscore = 0.
        if config["wandb"]['wandb_on']:
            wandb.init(project="GBCGCNN-MIT", name=config["wandb"]["name"], reinit=True, group=group)
        
        for stage in range(config["ensemble"]["num_stages"]):
            t0 = time.time()
            seed_everything(config["training_seed"], workers=True)
            trainer = Trainer(config, orig_atom_fea_len, nbr_fea_len, net_ensemble)
            trainer.fit(train_loader)
            net_ensemble.add(trainer.gb.model)
            net_ensemble.to_cuda()
            
            elapsed_tr = time.time()-t0
            
            # Training score
            loss_tr, precision_tr, recall_tr, gscore_tr, fscore_tr = class_eval(net_ensemble, train_loader)

            # Test score
            loss_te, precision_te, recall_te, gscore_te, fscore_te = class_eval(net_ensemble, test_loader)

            print(f'Stage - {stage}, training time: {elapsed_tr: .1f} sec')
            print(f'Loss@train : {loss_tr:.2f}, precision@train: {precision_tr:.2f}, recall@train: {recall_tr:.2f}, Gmean@train: {gscore_tr:.2f}, Fscore@train: {fscore_tr:.2f}')
            print(f'Loss@validation : {loss_te:.2f}, precision@validation: {precision_te:.2f}, recall@validation: {recall_te:.2f}, Gmean@validation: {gscore_te:.2f}, Fscore@validation: {fscore_te:.2f}')   

            if config["wandb"]['wandb_on']:
                wandb.log({
                    'Loss@train':loss_tr, 'precision@train':precision_tr, 'recall@train':recall_tr, 'Gmean@train':gscore_tr, 'Fscore@train':fscore_tr, 
                    'Loss@validation':loss_te, 'precision@validation':precision_te, 'recall@validation':recall_te, 'Gmean@validation':gscore_te, 'Fscore@validation':fscore_te,
                    'training time':elapsed_tr, 'boost rate': config["ensemble"]["boosting_rate"]})
            
            if best_fscore < fscore_te:
                best_fscore = fscore_te
                class_eval(net_ensemble, test_loader, to_file=config["wandb"]["group"]+"_"+name+"_best")
        
        folder_path = f'saved/{group}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
                  
        out_file = f'{folder_path}/{name}.pth' 
        net_ensemble.to_file(out_file)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GB-CGCNN')
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="configuration file") 
    args = parser.parse_args()
    config_file = args.config
    config_file = os.path.join(os.path.dirname("./"), config_file)
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    train(config)
    
