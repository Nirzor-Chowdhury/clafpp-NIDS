from __future__ import annotations
from dataclasses import dataclass
import numpy as np, torch
from scipy.spatial.distance import mahalanobis
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
class _AE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=12, dropout=0.1):
        super().__init__(); self.encoder=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,latent_dim)); self.decoder=nn.Sequential(nn.Linear(latent_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,input_dim),nn.Sigmoid())
    def forward(self,x): z=self.encoder(x); return self.decoder(z), z
@dataclass
class AESummary: train_loss:list[float]; latent_mean: np.ndarray|None=None; latent_precision: np.ndarray|None=None
class AutoencoderDetector:
    def __init__(self,input_dim,hidden_dim=64,latent_dim=12,dropout=0.1,lr=1e-3,epochs=6,batch_size=64,denoising_sigma=0.02,device=None):
        self.model=_AE(input_dim,hidden_dim,latent_dim,dropout); self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); self.model=self.model.to(self.device); self.lr=lr; self.epochs=epochs; self.batch_size=batch_size; self.denoising_sigma=denoising_sigma; self.summary_=AESummary([])
    def fit(self,X_normal):
        ds=TensorDataset(torch.tensor(X_normal,dtype=torch.float32)); dl=DataLoader(ds,batch_size=min(self.batch_size,len(ds)),shuffle=True); opt=torch.optim.Adam(self.model.parameters(),lr=self.lr); loss_fn=nn.MSELoss(); self.model.train()
        for _ in range(self.epochs):
            losses=[]
            for (xb,) in dl:
                xb=xb.to(self.device); noisy=xb+torch.randn_like(xb)*self.denoising_sigma; opt.zero_grad(); recon,_=self.model(noisy); loss=loss_fn(recon,xb); loss.backward(); opt.step(); losses.append(float(loss.item()))
            self.summary_.train_loss.append(float(np.mean(losses)) if losses else 0.0)
        latent=self.encode(X_normal); mean=latent.mean(axis=0); cov=np.cov(latent,rowvar=False)+np.eye(latent.shape[1])*1e-6; self.summary_.latent_mean=mean.astype(np.float32); self.summary_.latent_precision=np.linalg.pinv(cov).astype(np.float32); return self
    def encode(self,X):
        self.model.eval();
        with torch.no_grad(): xt=torch.tensor(X,dtype=torch.float32,device=self.device); _,z=self.model(xt)
        return z.cpu().numpy().astype(np.float32)
    def reconstruct(self,X):
        self.model.eval();
        with torch.no_grad(): xt=torch.tensor(X,dtype=torch.float32,device=self.device); recon,_=self.model(xt)
        return recon.cpu().numpy().astype(np.float32)
    def score_samples(self,X):
        z=self.encode(X); recon=self.reconstruct(X); re=((recon-X)**2).mean(axis=1).astype(np.float32); mean=self.summary_.latent_mean; prec=self.summary_.latent_precision; ld=np.array([mahalanobis(v,mean,prec) for v in z],dtype=np.float32); return {'reconstruction_error':re,'latent_distance':ld,'latent':z}
    def state_dict(self): return {'model_state_dict':self.model.state_dict(),'summary':{'train_loss':self.summary_.train_loss}}
