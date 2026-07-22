#!/usr/bin/env python3
"""Capacity-controlled heldout comparison from an explicit JSON experiment plan."""
import argparse,json,sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).parent)); from common import *
def main():
 p=argparse.ArgumentParser(); add_common(p); p.add_argument('--plan',required=True,help='JSON: dataset shape plus custom/cp/tucker {model,shapes}'); a=p.parse_args(); plan=json.load(open(a.plan)); y,h=load(a,tuple(plan['shape'])); rows={}
 for name,c in plan['models'].items():
  sh={k:int(v) for k,v in c['shapes'].items()}; _,z=fit(y,h,c['model'],sh,float(plan.get('alpha',1)),float(plan.get('beta',1)),a)
  rows[name]={'heldout_loss':float(z['heldout_loss'][-1]),'parameters':sum(np.prod([sh[x] for x in term]) for term in c['model'].split('->')[0].split(','))}
 custom=rows['custom']['heldout_loss']; standard=min(v['heldout_loss'] for k,v in rows.items() if k!='custom'); improvement=(standard-custom)/standard
 report(a,{'claim':4,'dataset':plan.get('dataset'),'models':rows,'best_standard_loss':standard,'relative_improvement':improvement,'pass':improvement>.37})
if __name__=='__main__':main()
