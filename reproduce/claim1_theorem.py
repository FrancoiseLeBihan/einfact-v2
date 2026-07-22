#!/usr/bin/env python3
"""Numerical falsification check for Claim 1; it cannot replace its analytic proof."""
import json, sys, pathlib
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from einfact import NNEinFact
# Eq. 3.3 family: selected (alpha,beta) losses; multiple decomposable einsums.
CASES=[('ir,jr->ij',{'i':19,'j':17,'r':4},1.,1.),('ir,jr->ij',{'i':19,'j':17,'r':4},1.,0.),
 ('ar,br,cr->abc',{'a':9,'b':8,'c':7,'r':3},.5,.5),('ar,br,cr->abc',{'a':9,'b':8,'c':7,'r':3},1.,-1.)]
def main():
    out=[]
    for n,(model,shape,a,b) in enumerate(CASES):
        np.random.seed(29482+n); torch.manual_seed(29482+n)
        # strictly positive prevents boundary/zero-loss exceptional cases
        y=np.random.default_rng(29482+n).uniform(.05,1,tuple(shape[c] for c in model.split('->')[1])).astype('float32')
        m=NNEinFact(model,shape,device='cpu',alpha=a,beta=b); h=m.fit(y,max_iter=80,verbose=False)
        losses=np.asarray(h['loss'],float); dif=np.diff(losses)
        # finite difference stationarity proxy after convergence; analytic theorem remains source of proof
        params=m.P_params; pred=m.Y_hat; g=[]
        for p in params:
            q=p.detach().clone().requires_grad_(True); xs=[q if z is p else z.detach() for z in params]
            z=torch.einsum(model,*xs).clamp_min(1e-10); loss=m._calculate_ab_divergence(torch.tensor(y),z); loss.backward(); g.append(float(q.grad.abs().max()))
        out.append({'model':model,'alpha':a,'beta':b,'initial':float(losses[0]),'final':float(losses[-1]),'max_increase':float(dif.max(initial=0)),'max_gradient':max(g)})
    ok=all(x['max_increase']<=1e-5 and x['final']<=x['initial'] for x in out)
    print(json.dumps({'claim':1,'numeric_mm_descent':ok,'cases':out},indent=2)); return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
