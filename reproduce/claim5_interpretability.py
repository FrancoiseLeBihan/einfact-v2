#!/usr/bin/env python3
"""Fits the 48,580-parameter Uber model and saves normalized latent summaries."""
import argparse, sys, pathlib, numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent)); from common import *
def main():
 p=argparse.ArgumentParser(); add_common(p); p.set_defaults(iterations=300); a=p.parse_args()
 y,h=load(a,(27,7,24,100,100)); sh=dict(w=27,h=7,d=24,i=100,j=100,r=10,k=24)
 m,z=fit(y,h,'wr,hr,dr,ikr,jkr->whdij',sh,1.,1.,a); W,H,D,I,J=m.get_params()
 # Each r component's time profile, spatial intensity and top day/hour are auditable outputs.
 temporal=(W.sum(0)[:,None]*H.sum(0)[None,:])[:,:,None]*D.sum(0)[None,None,:]
 score=temporal.sum((0,1)); top=np.argsort(score)[-3:][::-1]
 np.savez_compressed(pathlib.Path(a.out).with_suffix('.classes.npz'), components=top, weekday=W[:,top], hour=H[:,top], spatial_left=I[:,:,top], spatial_right=J[:,:,top], score=score)
 report(a,{'claim':5,'parameters':sum(x.size for x in (W,H,D,I,J)),'top_three_components':top.tolist(),'heldout_loss':float(z['heldout_loss'][-1]),'interpretation_artifact':str(pathlib.Path(a.out).with_suffix('.classes.npz')),'pass':sum(x.size for x in (W,H,D,I,J))==48580})
if __name__=='__main__': main()
