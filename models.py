import torch
import timm
import math
import numpy
import pickle

from torch import nn
from sklearn.mixture import GaussianMixture
from converter import convert_model_to_3d

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FisherLayer(nn.Module):
    def __init__(self, params):
        super(FisherLayer, self).__init__()
        self.gmm = GaussianMixture(len(params[0]),covariance_type='diag')
        self.gmm._set_parameters(params)

    def _get_shape(self, dim):
        return (2*dim+1)*self.gmm.n_components

    def _predict_proba(self, x):
        device = x.device
        x = x.reshape(-1,x.shape[-1])
        x = x.detach().cpu().numpy()
        x = self.gmm.predict_proba(x)
        return torch.tensor(x,dtype=torch.float,requires_grad=False).to(device)

    def forward(self, x):
        device = x.device
        K = self.gmm.n_components
        N,T,D = x.shape

        weights = self.gmm.weights_
        means = self.gmm.means_.reshape(1,K,D)
        covars = self.gmm.covariances_.reshape(1,K,D)
        
        weights = torch.tensor(weights,dtype=torch.float,requires_grad=False).to(device)
        means = torch.tensor(means,dtype=torch.float,requires_grad=False).to(device)
        covars = torch.tensor(covars,dtype=torch.float,requires_grad=False).to(device)
        # Gamma
        p = self._predict_proba(x)*weights
        p = (p/p.sum(axis=1).reshape(-1,1)).reshape(N,T,K,1)
        # Compute Statistics
        x = x.reshape(N,T,1,D)
        s0 = p.sum(axis=1).reshape(N,K,1)
        p = p*x
        s1 = p.sum(axis=1)
        p = p*x
        s2 = p.sum(axis=1)
        
        # Compute Fisher Vector signature
        weights = weights.reshape(1,-1,1)
        v0 = (s0-T*weights)
        weights = torch.sqrt(weights)
        v0 = v0/weights
        
        v1 = (s1-means*s0)/(weights*torch.sqrt(covars))
        v2 = (s2-2*means*s1+(means**2-covars)*s0)/(numpy.sqrt(2)*weights*covars)
        
        x = torch.cat((v0,v1,v2),2).reshape(N,-1)
        x = torch.sign(x)*torch.sqrt(torch.abs(x)) # Power normalization
        x = x / torch.linalg.norm(x, axis=1).reshape(-1,1) #L2 normalization
        return x.float()
        

MODEL = 'efficientvit_b2'

class FisherNet(nn.Module):
    def __init__(self, ds_info, path, features_only=False, load_classifier=False):
        super(FisherNet, self).__init__()
        self.net = timm.create_model(MODEL, in_chans=ds_info['n_channels'], num_classes=ds_info['n_classes'])
        if ds_info['n_dims'] == 3:
            self.net = convert_model_to_3d(self.net)
            
        params = torch.load(path)
        self.net.load_state_dict(params['net'],strict=True)
        self.net.eval()

        if features_only:
            self.fisher_vector = torch.nn.Identity()
            self.net.head.classifier = torch.nn.Identity()
        else:
            assert 'gmm' in params, "GMM not found"
            self.fisher_vector = FisherLayer(params['gmm'])
            #in_shape = self.net.stages[2].blocks[-1].local_module.main.point_conv.norm.num_features
            in_shape = self.net.stages[1].blocks[-1].main.point_conv.norm.num_features
            in_shape = self.fisher_vector._get_shape(in_shape)
            out_shape = self.net.head.classifier[0].out_features
            bias = self.net.head.classifier[0].bias
            self.net.head.classifier[0] = torch.nn.Linear(in_shape, out_shape, bias)
            if load_classifier:
                assert 'cfv' in params, "Classifier weights not found"
                self.net.head.classifier.load_state_dict(params['cfv'],strict=True)
    
    def local_features(self, x):
        x = self.net.stem(x)
        x = self.net.stages[0](x)
        x = self.net.stages[1](x)
        w = self.net.stages[2](x)
        v = self.net.stages[3](w)
        N = x.shape[0]
        D = x.shape[1]
        x = torch.cat((
            x.reshape(N,D,-1),
            w.reshape(N,D,-1),
            v.reshape(N,D,-1),
        ),2)
        x = x.swapaxes(1,2)
        return x

    def forward(self, x):
        with torch.no_grad():
            x = self.local_features(x)
            x = self.fisher_vector(x)
        x = self.net.head.classifier(x)
        return x
