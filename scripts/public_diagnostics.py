"""Paired public-controller analysis and control-response figures."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ORDER=['pgvps','ps','mppi','icem','cem','smppi']
LABEL=['PG-VPS','PS','MPPI','iCEM','CEM','Smooth-MPPI']
COLORS=['#0072B2','#999999','#009E73','#D55E00','#CC79A7','#332288']
STYLES=['-','--','-.',':',(0,(5,1,1,1)),(0,(3,1,1,1,1,1))]

def main(root,out):
 out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 folders={m:root/'results'/('pgvps_public_subset' if m=='pgvps' else 'pgtaiv_public_baselines') for m in ORDER}
 rows={m:sorted([r for r in json.loads((folders[m]/'summary.json').read_text()) if r['mode']==m],key=lambda r:(r['seed'],r['friction'],r['mass'])) for m in ORDER}
 assert all(len(r)==24 for r in rows.values())
 seeds=sorted({r['seed'] for r in rows['pgvps']});contrasts={}
 for m in ORDER[1:]:
  d=np.array([np.mean([r['success'] for r in rows['pgvps'] if r['seed']==s])-np.mean([r['success'] for r in rows[m] if r['seed']==s]) for s in seeds])
  null=np.array([np.mean(d*np.array(sign)) for sign in itertools.product([-1,1],repeat=6)])
  contrasts[m]=dict(seed_differences=d.tolist(),difference=float(d.mean()),exact_p=float(np.mean(abs(null)>=abs(d.mean())-1e-12)))
 ranked=sorted(contrasts,key=lambda m:contrasts[m]['exact_p']);running=0
 for i,m in enumerate(ranked):
  running=max(running,min(1.,(5-i)*contrasts[m]['exact_p']));contrasts[m]['holm_p']=running
 def finish(fig,name):
  from scripts.legend_layout import reserve_legend_space, annotate_results
  reserve_legend_space(fig)
  annotate_results(fig,name)
  for ax in fig.axes:ax.grid(alpha=.22)
  fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(7,4.8),layout='constrained')
 for m,l,c,st in zip(ORDER,LABEL,COLORS,STYLES):
  rates=[100*np.mean([r['success'] for r in rows[m] if r['seed']==s]) for s in seeds]
  axs[0,0].plot(range(1,7),rates,linestyle=st,color=c,marker='o',markersize=3,label=l)
  for ax,field in [(axs[0,1],'cost'),(axs[1,0],'dwell_s'),(axs[1,1],'work')]:
   vals=np.sort([r.get('physics_steps',20*r.get('rollout_steps',0)) if field=='work' else r[field] for r in rows[m]])
   ax.step(vals,np.arange(1,25)/24,where='post',linestyle=st,color=c,label=l)
 axs[0,0].set(xlabel='Task instance',ylabel='Success (%)',ylim=(-4,105),xticks=range(1,7));axs[0,0].legend(loc='lower left',ncol=2,framealpha=.65)
 axs[0,1].set(xlabel='Task cost',ylabel='Empirical cumulative fraction',xscale='log')
 axs[1,0].set(xlabel='Completion time (s)',ylabel='Empirical cumulative fraction')
 axs[1,1].set(xlabel='Prediction physics steps',ylabel='Empirical cumulative fraction',xscale='log')
 finish(fig,'fig_control_public_tradeoff')
 fig,axs=plt.subplots(2,2,figsize=(7,4.8),layout='constrained');t=.05*np.arange(1,401)
 robust={}
 for m,l,c,st in zip(ORDER,LABEL,COLORS,STYLES):
  data=[np.load(folders[m]/(r['name']+'.npz')) for r in rows[m]];err=np.array([d['errors'] for d in data]);u=np.array([np.linalg.norm(d['actions'],axis=1) for d in data])
  series=[np.median(err[:,:,0]*1000,axis=0),np.median(err[:,:,1],axis=0),np.mean(np.array([np.convolve(((e[:,0]<.025)&(e[:,1]<.15)).astype(int),np.ones(10,dtype=int),'full')[:400]==10 for e in err]),axis=0)*100,np.median(u,axis=0)]
  for ax,y in zip(axs.flat,series):ax.plot(t,y,linestyle=st,color=c,label=l)
  robust[m]=[dict(friction=f,mass=mass,success=int(sum(r['success'] for r in rows[m] if r['friction']==f and r['mass']==mass)),cost=float(np.mean([r['cost'] for r in rows[m] if r['friction']==f and r['mass']==mass]))) for f,mass in [(.1,.25),(.1,.5),(.6,.25),(.6,.5)]]
 for ax,ylabel in zip(axs.flat,['Median position error (mm)','Median yaw error (rad)','Sustained-in-tolerance (%)','Median input norm (m/s)']):ax.set(xlabel='Time (s)',ylabel=ylabel)
 axs[0,0].legend(loc='upper right',ncol=2,framealpha=.65);axs[0,0].axhline(25,color='k',ls=':',lw=.7);axs[0,1].axhline(.15,color='k',ls=':',lw=.7);axs[1,1].axhline(.15,color='k',ls=':',lw=.7)
 finish(fig,'fig_public_responses')
 fig,axs=plt.subplots(2,2,figsize=(7,5.2),layout='constrained',sharex=True)
 for ax,(friction,mass) in zip(axs.flat,[(.1,.25),(.1,.5),(.6,.25),(.6,.5)]):
  for m,l,c,st in zip(ORDER,LABEL,COLORS,STYLES):
   selected=[r for r in rows[m] if r['friction']==friction and r['mass']==mass]
   energies=[]
   for r in selected:
    with np.load(folders[m]/(r['name']+'.npz')) as d:energies.append(d['errors'][:,2])
   ax.plot(t,np.median(energies,axis=0),color=c,ls=st,lw=1.25,label=l)
  ax.set(xlabel='Time (s)',ylabel='Median task energy',title=rf'$\mu={friction:g}$, $m={mass:g}$ kg')
  ax.set_yscale('symlog',linthresh=.01)
 axs[0,0].legend(ncol=3,loc='upper right',frameon=False)
 finish(fig,'fig_public_robustness')
 benchmark=root/'results/planning_benchmark/summary.json'
 if benchmark.exists():
  br=json.loads(benchmark.read_text());budgets=sorted({r['samples'] for r in br if r['method']!='pgvps'});fig,axs=plt.subplots(1,2,figsize=(7,2.7),layout='constrained')
  for m,l,c,st in zip(ORDER,LABEL,COLORS,STYLES):
   for ax,field in zip(axs,['median_ms','physics_steps']):
    vals=[np.mean([r[field] for r in br if r['method']==m and r['samples']==(0 if m=='pgvps' else n)]) for n in budgets]
    ax.plot(budgets,vals,linestyle=st,color=c,marker='o',ms=3,label=l)
  for ax in axs:
   ax.set(xlabel='Initial samples',xscale='log',yscale='log',xticks=[32,128,256],xticklabels=['32','128','256'])
   ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
  axs[0].set(ylabel='Initial planning time (ms)');axs[1].set(ylabel='Initial prediction physics steps');axs[1].legend(ncol=2,loc='upper left',framealpha=.65)
  finish(fig,'fig_planning_budget')
 result=dict(seeds=seeds,paired_success=contrasts,robustness=robust)
 dest=root/'results/pgvps_analysis';dest.mkdir(parents=True,exist_ok=True);(dest/'public_diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(contrasts,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--figures',type=Path);a=p.parse_args();main(a.root,a.figures or a.root/'figures')
