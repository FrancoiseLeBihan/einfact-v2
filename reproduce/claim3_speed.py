#!/usr/bin/env python3
"""Paired NNEinFact-vs-Adam benchmark; GPU events, common seed/split/init/budget."""
import argparse, sys, pathlib, time, numpy as np, torch
sys.path.insert(0,str(pathlib.Path(__file__).parent)); from common import *
def adam(y,h,model,sh,a,b,args):
 np.random.seed(args.seed); torch.manual_seed(args.seed); base=NNEinFact(model,sh,'cuda',a,b)
 ps=[torch.nn.Parameter(x.detach().clone()) for x in base.P_params]; opt=torch.optim.Adam(ps,lr=args.lr)
 Y=torch.as_tensor(y,device='cuda'); mask=torch.as_tensor(~h,device='cuda'); losses=[]; times=[]; torch.cuda.synchronize(); t=time.perf_counter()
 for _ in range(args.iterations):
  opt.zero_grad(set_to_none=True); pred=torch.einsum(model,*ps).clamp_min(1e-10); loss=base._calculate_ab_divergence(Y[mask],pred[mask]); loss.backward(); opt.step()
  with torch.no_grad():
   losses.append(float(base._calculate_ab_divergence(Y[~mask],torch.einsum(model,*ps).clamp_min(1e-10)[~mask]))); torch.cuda.synchronize(); times.append(time.perf_counter()-t)
 return np.array(losses),np.array(times)
def main():
 p=argparse.ArgumentParser();add_common(p);p.add_argument('--lr',type=float,default=.01);p.add_argument('--pairs',default='1,1;1,0;1,-1;.5,.5;.5,1;2,-1');a=p.parse_args()
 y,h=load(a); letters=''.join(chr(97+i) for i in range(y.ndim)); sh=dict(zip(letters,y.shape)); sh.update(r=10); model=','.join(f'{x}r' for x in letters)+'->'+letters
 rows=[]
 for pair in a.pairs.split(';'):
  al,be=map(float,pair.split(',')); _,mu=fit(y,h,model,sh,al,be,a); ml=np.asarray(mu['heldout_loss'][1:]); mt=np.asarray(mu['time'][1:]); gl,gt=adam(y,h,model,sh,al,be,a)
  target=min(ml[-1],gl[-1]); tm=mt[np.flatnonzero(ml<=target)[0]] if np.any(ml<=target) else np.inf; tg=gt[np.flatnonzero(gl<=target)[0]] if np.any(gl<=target) else np.inf
  rows.append({'alpha':al,'beta':be,'mu_loss':float(ml[-1]),'adam_loss':float(gl[-1]),'mu_seconds_to_common_target':float(tm),'adam_seconds_to_common_target':float(tg),'speedup':float(tg/tm)})
 report(a,{'claim':3,'rows':rows,'pass':all(x['mu_loss']<x['adam_loss'] and x['speedup']>=90 for x in rows)})
if __name__=='__main__': main()
