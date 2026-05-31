from __future__ import annotations
from dataclasses import dataclass
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
class _E(nn.Module):
    def __init__(self,input_dim,hidden_dim,latent_dim): super().__init__(); self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LeakyReLU(0.2),nn.Linear(hidden_dim,latent_dim))
    def forward(self,x): return self.net(x)
class _D(nn.Module):
    def __init__(self,latent_dim,hidden_dim,input_dim): super().__init__(); self.net=nn.Sequential(nn.Linear(latent_dim,hidden_dim),nn.LeakyReLU(0.2),nn.Linear(hidden_dim,input_dim),nn.Sigmoid())
    def forward(self,z): return self.net(z)
class _Disc(nn.Module):
    def __init__(self,input_dim,hidden_dim): super().__init__(); self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LeakyReLU(0.2),nn.Linear(hidden_dim,1),nn.Sigmoid())
    def forward(self,x): return self.net(x)
@dataclass
class GANSummary: generator_loss:list[float]; discriminator_loss:list[float]
class GANomalyDetector:
    def __init__(self,input_dim,hidden_dim=64,latent_dim=12,lr=1e-3,epochs=4,batch_size=64,lambda_adv=1.0,lambda_con=20.0,lambda_enc=1.0,device=None):
        self.e1=_E(input_dim,hidden_dim,latent_dim); self.d=_D(latent_dim,hidden_dim,input_dim); self.e2=_E(input_dim,hidden_dim,latent_dim); self.c=_Disc(input_dim,hidden_dim)
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); [m.to(self.device) for m in [self.e1,self.d,self.e2,self.c]]; self.lr=lr; self.epochs=epochs; self.batch_size=batch_size; self.lambda_adv=lambda_adv; self.lambda_con=lambda_con; self.lambda_enc=lambda_enc; self.history_=GANSummary([],[])
    def fit(self,X_normal):
        ds=TensorDataset(torch.tensor(X_normal,dtype=torch.float32)); dl=DataLoader(ds,batch_size=min(self.batch_size,len(ds)),shuffle=True); gparams=list(self.e1.parameters())+list(self.d.parameters())+list(self.e2.parameters()); go=torch.optim.Adam(gparams,lr=self.lr); co=torch.optim.Adam(self.c.parameters(),lr=self.lr); mse=nn.MSELoss(); bce=nn.BCELoss()
        for _ in range(self.epochs):
            gl=[]; cl=[]
            for (xb,) in dl:
                xb=xb.to(self.device); ones=torch.ones((xb.size(0),1),device=self.device); zeros=torch.zeros((xb.size(0),1),device=self.device)
                go.zero_grad(); z=self.e1(xb); recon=self.d(z); zhat=self.e2(recon); adv=mse(self.c(recon), ones); con=mse(recon, xb); enc=mse(zhat, z.detach()); gloss=self.lambda_adv*adv+self.lambda_con*con+self.lambda_enc*enc; gloss.backward(); go.step()
                co.zero_grad(); dloss=0.5*(bce(self.c(xb.detach()), ones)+bce(self.c(recon.detach()), zeros)); dloss.backward(); co.step(); gl.append(float(gloss.item())); cl.append(float(dloss.item()))
            self.history_.generator_loss.append(float(np.mean(gl)) if gl else 0.0); self.history_.discriminator_loss.append(float(np.mean(cl)) if cl else 0.0)
        return self
    def score_samples(self,X):
        with torch.no_grad(): xt=torch.tensor(X,dtype=torch.float32,device=self.device); z=self.e1(xt); recon=self.d(z); zhat=self.e2(recon); realism=self.c(recon).squeeze(1)
        recon_np=recon.cpu().numpy().astype(np.float32); z_np=z.cpu().numpy().astype(np.float32); zhat_np=zhat.cpu().numpy().astype(np.float32); realism_np=realism.cpu().numpy().astype(np.float32)
        re=((recon_np-X)**2).mean(axis=1).astype(np.float32); lc=((z_np-zhat_np)**2).mean(axis=1).astype(np.float32); score=(0.6*re+0.3*lc+0.1*(1-realism_np)).astype(np.float32)
        return {'ganomaly_score':score,'reconstruction_error':re,'latent_consistency':lc,'discriminator_realism':realism_np}
    def state_dict(self): return {'history':{'generator_loss':self.history_.generator_loss,'discriminator_loss':self.history_.discriminator_loss}}
