"""Serial initial-planning latency on three fixed observed task states."""
import argparse,copy,json,os,sys,platform,time
from pathlib import Path
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
import reproduce
from scripts import baseline_episode,episode
from research_platform.polygon_push import make_polygon_model
from research_platform.known_servo import configure_servo
from research_platform.uncertainty_physics import configure_truth
from research_platform.tabletop_push import initialize
from research_platform.push_measurement import noisy_observation

BUDGETS = [32,48,64,96,128,160,192,256]

def run(output, resume=False):
 out=output/'results/planning_benchmark';out.mkdir(parents=True,exist_ok=True);results=[]
 if resume and (out/'summary.json').exists():results=json.loads((out/'summary.json').read_text())
 completed={(r['case'],r['method'],r['samples']) for r in results}
 rows=[r for r in reproduce.rows_for('pgvps_public_subset') if r['seed'] in [19840,19842,19845] and r['friction']==.6 and r['mass']==.5]
 for row in rows:
  model=configure_servo(configure_truth(make_polygon_model(row['vertices'],row['friction']),row['cfg']),row['cfg']['servo_kv']);data=initialize(model,row['seed'])
  state=noisy_observation(model,data,np.random.default_rng(row['seed']+300000),row['cfg']['noise'])
  for method in ['pgvps','ps','mppi','icem','cem','smppi']:
   for budget in ([0] if method=='pgvps' else BUDGETS):
    if (row['name'],method,budget) in completed:continue
    times=[];steps=[]
    # A warm-up call followed by three independently constructed repeats.
    for repeat in range(4):
     r=copy.deepcopy(row);r['mode']=method
     if method=='pgvps':ctrl=episode.make_controller(r)
     else:
      settings=copy.deepcopy(reproduce.CONFIG['public']);settings[method]['samples']=budget
      ctrl=baseline_episode.make_controller(r,settings)
     start=time.perf_counter();action=ctrl.action(state,row['goal']);elapsed=time.perf_counter()-start
     core=ctrl.core if method=='pgvps' else ctrl
     work=getattr(core,'physics_steps',20*core.rollout_steps)
     if repeat:times.append(elapsed*1000);steps.append(work)
    results.append(dict(case=row['name'],method=method,samples=budget,ms=times,median_ms=float(np.median(times)),physics_steps=steps[0]))
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    print(row['shape'],method,budget,round(np.median(times),3),flush=True)
 (out/'summary.json').write_text(json.dumps(results,indent=2))
 (out/'hardware.json').write_text(json.dumps(dict(platform=platform.platform(),python=sys.version,cpu=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),''),thread_count=1,serial=True,repeats=3,scope='initial planning call; setup excluded; one warm-up per configuration'),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.output,a.resume)
