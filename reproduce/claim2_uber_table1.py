#!/usr/bin/env python3
"""Claim-2 Table-1 check, sharded exactly across two 16-GB-class CUDA GPUs.

The i (origin-cell) mode is split over the GPUs.  Each multiplicative-update
numerator/denominator is accumulated over shards *before* its factor is changed,
so this is the same full-batch update as the single-GPU algorithm (apart from
floating-point summation order).  It never materialises the 725M-entry prediction
on one T4.

Divergence: (alpha, beta) = (1, 0), the generalised KL / Poisson likelihood,
as specified in Section 6.2 of the paper for count data.
"""
import argparse, concurrent.futures, sys, pathlib, time
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from common import add_common, load, report
sys.path.insert(0, str(pathlib.Path(__file__).parents[1])); from einfact import swap

class TwoT4MU:
    def __init__(self, y, heldout, model, shapes, seed):
        if torch.cuda.device_count() < 2: raise RuntimeError('This script requires two CUDA GPUs.')
        self.model, self.terms = model, model.split('->')[0].split(','); self.out=model.split('->')[1]
        self.dev=[torch.device('cuda:0'),torch.device('cuda:1')]; self.i_axis=self.out.index('i')
        if y.shape[self.i_axis] % 2: raise ValueError('i mode must split evenly across two GPUs')
        np.random.seed(seed); self.cpu=[np.random.uniform(0,1,[shapes[c] for c in t]).astype('float32') for t in self.terms]
        self.slices=[slice(0,y.shape[self.i_axis]//2),slice(y.shape[self.i_axis]//2,y.shape[self.i_axis])]
        self.y=[]; self.obs=[]; self.hold=[]
        for d,s in zip(self.dev,self.slices):
            key=[slice(None)]*y.ndim; key[self.i_axis]=s; key=tuple(key)
            self.y.append(torch.as_tensor(np.ascontiguousarray(y[key]),device=d)); ho=np.ascontiguousarray(heldout[key])
            self.hold.append(torch.as_tensor(ho,device=d)); self.obs.append(torch.as_tensor(~ho,device=d))
        self.eq=[swap(model,n) for n in range(len(self.terms))]
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
    def _one(self, shard, which):
        d=self.dev[shard]
        with torch.cuda.device(d):
            p=[torch.as_tensor(x,device=d) for x in self.cpu]; p[3]=p[3][self.slices[shard]]
            pred=torch.einsum(self.model,*p).clamp_min_(1e-10)
            # alpha=1, beta=0: a(x,y) = x^alpha * y^(beta-1) = x * y^(-1) = x/y
            #                  b(x,y) = y^(alpha+beta-1) = y^0 = 1
            A=(self.y[shard] / pred) * self.obs[shard]
            B=self.obs[shard].float()
            others=p[:which]+p[which+1:]
            num=torch.einsum(self.eq[which],*others,A); den=torch.einsum(self.eq[which],*others,B).clamp_min_(1e-10)
            torch.cuda.synchronize(d); return num.cpu(),den.cpu()
    def step(self):
        for q in range(len(self.cpu)):
            pair=list(self.pool.map(lambda s:self._one(s,q),range(2)))
            if q==3: # i-factor is disjoint across shards.
                n=torch.cat([x[0] for x in pair],0); d=torch.cat([x[1] for x in pair],0)
            else: n=sum(x[0] for x in pair); d=sum(x[1] for x in pair)
            # gamma_ab = 1/alpha = 1 for (alpha=1, beta=0), so ratio^1 = ratio.
            self.cpu[q]*=np.clip((n/d).numpy(),1e-10,1e10)
    def heldout_divergence(self):
        """Average heldout (1,0)-divergence = generalised KL."""
        def one(s):
            d=self.dev[s]
            with torch.cuda.device(d):
                p=[torch.as_tensor(x,device=d) for x in self.cpu]; p[3]=p[3][self.slices[s]]
                yhat=torch.einsum(self.model,*p).clamp_min_(1e-10)
                ys=self.y[s]; ho=self.hold[s]
                yh=ys[ho]; yhat_h=yhat[ho]
                # D_{1,0}(x,y) = y - x + x*log(x/y)  for x > 0
                #              = y                     for x = 0
                pos = yh > 0
                div_sum = yhat_h.sum()  # sum of y_hat terms (covers both x>0 and x=0)
                div_sum = div_sum - yh.sum()  # subtract x for all (x=0 terms contribute 0)
                div_sum = div_sum + (yh[pos] * torch.log(yh[pos] / yhat_h[pos])).sum()  # x*log(x/y) for x>0
                count = ho.sum()
                torch.cuda.synchronize(d)
                return div_sum.cpu(), count.cpu()
        z=list(self.pool.map(one,range(2))); return float(sum(x[0] for x in z)/sum(x[1] for x in z))
    def fit(self, iters):
        hist=[]; start=time.perf_counter()
        for _ in range(iters): self.step(); hist.append(self.heldout_divergence())
        return hist,time.perf_counter()-start

def run(y,h,model,shapes,args):
    m=TwoT4MU(y,h,model,shapes,args.seed); return m.fit(args.iterations)
def main():
    p=argparse.ArgumentParser(); add_common(p); p.set_defaults(iterations=200); a=p.parse_args()
    y,h=load(a,(27,7,24,100,100)); shapes=dict(w=27,h=7,d=24,i=100,j=100,r=10,k=24)
    custom,ct=run(y,h,'wr,hr,dr,ikr,jkr->whdij',shapes,a)
    cp,pt=run(y,h,'wr,hr,dr,ir,jr->whdij',{**shapes,'r':188},a)
    v1,v2=custom[-1],cp[-1]
    report(a,{'claim':2,'implementation':'exact two-T4 i-sharded full-batch multiplicative updates','divergence':'alpha=1 beta=0 (KL/Poisson)','split':'provided' if a.heldout else 'hash-5%-NOT-paper-split','custom_parameters':48580,'cp_parameters':188*258,'custom_heldout_div':v1,'cp_heldout_div':v2,'seconds':{'custom':ct,'cp':pt},'targets':{'custom':.0101,'cp':.0104},'pass':abs(v1-.0101)<=.0005 and abs(v2-.0104)<=.0005})
if __name__=='__main__': main()
