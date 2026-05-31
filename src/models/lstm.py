from __future__ import annotations
from dataclasses import dataclass
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
class _Net(nn.Module):
    def __init__(self,input_dim,hidden_dim=32,num_layers=1,dropout=0.1):
        super().__init__(); self.lstm=nn.LSTM(input_dim,hidden_dim,num_layers=num_layers,batch_first=True,bidirectional=True); self.head=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,1))
    def forward(self,x): out,_=self.lstm(x); return self.head(out[:,-1,:]).squeeze(1)
@dataclass
class LSTMSummary: train_loss:list[float]
class SequenceLSTMDetector:
    def __init__(self,input_dim,seq_len=6,hidden_dim=32,num_layers=1,lr=1e-3,epochs=3,batch_size=64,dropout=0.1,device=None):
        self.seq_len=seq_len; self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu'); self.model=_Net(input_dim,hidden_dim,num_layers,dropout).to(self.device); self.lr=lr; self.epochs=epochs; self.batch_size=batch_size; self.summary_=LSTMSummary([])
    def build_sequences(self,X):
        seqs=[]
        for i in range(len(X)):
            start=max(0,i-self.seq_len+1); chunk=X[start:i+1]
            if len(chunk)<self.seq_len: chunk=np.vstack([np.repeat(chunk[:1], self.seq_len-len(chunk), axis=0), chunk])
            seqs.append(chunk)
        return np.stack(seqs).astype(np.float32)
    def fit(self,X,y):
        Xs=self.build_sequences(X); ds=TensorDataset(torch.tensor(Xs,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)); dl=DataLoader(ds,batch_size=min(self.batch_size,len(ds)),shuffle=True)
        pos=max((y==1).sum(),1); neg=max((y==0).sum(),1); loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],dtype=torch.float32,device=self.device)); opt=torch.optim.Adam(self.model.parameters(),lr=self.lr)
        for _ in range(self.epochs):
            losses=[]
            for xb,yb in dl:
                xb=xb.to(self.device); yb=yb.to(self.device); opt.zero_grad(); loss=loss_fn(self.model(xb), yb); loss.backward(); opt.step(); losses.append(float(loss.item()))
            self.summary_.train_loss.append(float(np.mean(losses)) if losses else 0.0)
        return self
    def predict_proba(self,X):
        Xs=self.build_sequences(X)
        with torch.no_grad(): xt=torch.tensor(Xs,dtype=torch.float32,device=self.device); probs=torch.sigmoid(self.model(xt)).cpu().numpy()
        return probs.astype(np.float32)
    def state_dict(self): return {'summary':{'train_loss':self.summary_.train_loss},'seq_len':self.seq_len}
