import numpy as np
import pandas as pd
import torch
import shutil

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from imblearn.metrics import geometric_mean_score

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os

#grownet
def init_gbnn(y):
    counts = np.unique(y, return_counts=True)[1]
    odds = min(counts) / max(counts)
    
    blind_acc = max(counts) / len(y)
    print(f'Blind accuracy: {blind_acc}')

    return np.log(odds)

def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def class_eval(net_ensemble, data_loader, to_file=False, single=None):
    if single is not None:
        single.eval()
        
    net_ensemble.to_eval() # Set the models in ensemble net to eval mode
    loss_BCE = torch.nn.BCELoss()
    
    y_true = []
    y_pred = []
    y_prob = []
    test_cif_ids = []
    loss = 0
    
    for i, (input, target, batch_cif_ids) in enumerate(data_loader):
        input = input.to('cuda', non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        with torch.no_grad():
            if single is not None:
                _, pred = single.forward(input, None)
                pred = pred + net_ensemble.c0
                
            else:
                _, pred = net_ensemble.forward(input)
                
        prob = 1 / (1 + torch.exp(-pred))
        
        pred = torch.sign(pred)
        pred[pred==-1] = 0.
        target = (target + 1) / 2
        loss += loss_BCE(pred, target).item()
        
        pred = pred.squeeze()
        pred = pred.cpu().numpy().tolist()
        prob = prob.squeeze()
        prob = prob.cpu().numpy().tolist()
        target = target.squeeze()
        target = target.cpu().numpy().tolist()
        
        y_pred.extend(pred)
        y_true.extend(target)
        y_prob.extend(prob)
        test_cif_ids.extend(batch_cif_ids)

    metrics = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    gscore = geometric_mean_score(y_true, y_pred, average='binary', pos_label=1)
    precision = metrics[0]
    recall = metrics[1]
    fscore = metrics[2]
    
    if to_file is not False:
        star_label = '**'
        import csv
        output_dir = 'test_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(f'{output_dir}/test_results_{to_file}.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred, prob in zip(test_cif_ids, y_true,
                                            y_pred, y_prob):
                writer.writerow((cif_id, target, pred, prob))


    return loss/(i+1), precision, recall, gscore, fscore


def avg_curve(files):
    df_results = pd.DataFrame()
    X = np.linspace(0, 1, 10000)
    precisions = []
    tprs = []

    for fold in range(10):
        df = pd.read_csv(files[fold], header=0)
        y_test = df.iloc[:,1].values
        y_score = df.iloc[:,3].values

        precision, recall, _ = precision_recall_curve(y_test, y_score)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ROC_AUC = auc(fpr, tpr)
        PR_AUC = auc(recall, precision)

        precision_ = [precision[np.where(recall == recall[recall >= x].min())[0][-1]] for x in X]
        tpr_ = [tpr[np.where(fpr == fpr[fpr <= x].max())[0][-1]] for x in X]

        precisions.append(precision_)
        tprs.append(tpr_)

        df_results.loc[f'cv{fold}', 'ROC_AUC'] = ROC_AUC
        df_results.loc[f'cv{fold}', 'PR_AUC'] = PR_AUC

    p_avg = np.array(precisions).mean(axis=0)
    tpr_avg = np.array(tprs).mean(axis=0)

    X_p0 = np.append(X, 1.)
    p_avg0 = np.append(p_avg, 0.)

    X_tpr0 = np.append(0., X)
    tpr_avg0 = np.append(0., tpr_avg)
    
    return df_results