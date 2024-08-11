import numpy as np
import torch

class BoostingNet(object):
    def __init__(self, c0):
        self.models = []
        self.c0 = c0

    def add(self, model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self, graphs):
        if len(self.models) == 0:
            n_crystal=graphs.num_graphs
            return None, np.full(n_crystal, self.c0)
        
        middle_feat_cum = None
        prediction = None
        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, prediction = m(graphs, middle_feat_cum)
                else:
                    middle_feat_cum, pred = m(graphs, middle_feat_cum)
                    prediction += pred
                
        return middle_feat_cum, self.c0 + prediction

    @classmethod
    def from_file(cls, base_model, path, orig_atom_fea_len, nbr_fea_len, num_stage, args):
        d = torch.load(path)
        net = BoostingNet(d['c0'])
        for stage, m in enumerate(d['models'][:num_stage]):
            submod = base_model.get_model(orig_atom_fea_len, nbr_fea_len, args)
            submod.load_state_dict(m)
            net.add(submod)
        return net

    def to_file(self, path):
        models = [m.state_dict() for m in self.models]
        d = {'models': models, 'c0': self.c0}
        torch.save(d, path)
