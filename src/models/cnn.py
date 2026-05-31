from __future__ import annotations
from dataclasses import dataclass
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
class _CNN(nn.Module):
    def __init__(self,input_dim,channels=16,dropout=0.1):
        super().__init__(); self.features=nn.Sequential(nn.Conv1d(1,channels,5,padding=2),nn.ReLU(),nn.Conv1d(channels,channels,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(8)); self.head=nn.Sequential(nn.Flatten(),nn.Linear(channels*8,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,x): return self.head(self.features(x)).squeeze(1)
@dataclass
class CNNSummary: train_loss:list[float]
class CNNPatternDetector:
    def __init__(self,input_dim,channels=16,lr=1e-3,epochs=3,batch_size=64,dropout=0.1,device=None):
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); self.model=_CNN(input_dim,channels,dropout).to(self.device); self.lr=lr; self.epochs=epochs; self.batch_size=batch_size; self.summary_=CNNSummary([])
    def fit(self,X,y):
        ds=TensorDataset(torch.tensor(X[:,None,:],dtype=torch.float32), torch.tensor(y,dtype=torch.float32)); dl=DataLoader(ds,batch_size=min(self.batch_size,len(ds)),shuffle=True)
        pos=max((y==1).sum(),1); neg=max((y==0).sum(),1); loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],dtype=torch.float32,device=self.device)); opt=torch.optim.Adam(self.model.parameters(),lr=self.lr)
        for _ in range(self.epochs):
            losses=[]
            for xb,yb in dl:
                xb=xb.to(self.device); yb=yb.to(self.device); opt.zero_grad(); loss=loss_fn(self.model(xb), yb); loss.backward(); opt.step(); losses.append(float(loss.item()))
            self.summary_.train_loss.append(float(np.mean(losses)) if losses else 0.0)
        return self
    def predict_proba(self,X):
        with torch.no_grad(): xt=torch.tensor(X[:,None,:],dtype=torch.float32,device=self.device); probs=torch.sigmoid(self.model(xt)).cpu().numpy()
        return probs.astype(np.float32)
    def state_dict(self): return {'summary':{'train_loss':self.summary_.train_loss}}
