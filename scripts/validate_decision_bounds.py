"""Compare candidate costs at three resolutions and check conditional bounds."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import episode
from research_platform.prediction_fidelity_v3 import rollout_cost
from research_platform.polygon_push import make_polygon_model
from research_platform.known_servo import configure_servo
from research_platform.uncertainty_physics import configure_truth

def run(output):
 import reproduce
 rows=reproduce.rows_for('pgvps_public_subset')
 rows=[r for r in rows if r['friction']==.1 and r['mass']==.25]
 out=output/'results/decision_bounds';out.mkdir(parents=True,exist_ok=True)
 records=[];original=episode.make_controller
 for row in rows:
  snapshots=[]
  def build(r):
   controller=original(r);core=controller.core;generate=core.candidate_controllers
   def capture(state,goal):
    candidates=generate(state,goal)
    if core.plans in [2,5,10,15,20,25,30,35]:
     snapshots.append(dict(plan=core.plans,qpos=state.qpos.copy(),qvel=state.qvel.copy(),vertices=core.geometry.vertices.copy(),candidates=copy.deepcopy(candidates),goal=np.array(goal),rho=core.switch_ratio))
    return candidates
   core.candidate_controllers=capture
   return controller
  episode.make_controller=build
  try: result=episode.run((out,row))
  finally:episode.make_controller=original
  for snap in snapshots:
   model=configure_servo(make_polygon_model(snap['vertices'],.3),row['cfg']['servo_kv']);data=mujoco.MjData(model)
   truth=configure_servo(configure_truth(make_polygon_model(row['vertices'],row['friction']),row['cfg']),row['cfg']['servo_kv']);td=mujoco.MjData(truth)
   def costs(m,d,dt):return np.array([rollout_cost(m,d,c,snap['qpos'],snap['qvel'],snap['goal'],dt) for c in snap['candidates']])
   C=costs(model,data,.005);J=costs(model,data,.0025);P=costs(truth,td,.00125)
   eps=float(np.max(abs(C-J)));delta=float(np.max(abs(J-P)));b=int(np.argmin(C));rho=snap['rho']
   status='retention' if b==0 or C[b]>=rho*C[0] else ('accepted' if J[b]<rho*J[0] else 'rejected')
   chosen=b if status=='accepted' else 0
   bound=2*eps+2*delta if status=='accepted' else (1-rho)*J[0]+((1+rho)*eps if status=='retention' else 2*eps)+2*delta
   slack=float(bound-(P[chosen]-P.min()));assert slack>=-1e-8
   certified=bool((1-rho)*J[0]>2*delta)
   if status=='accepted' and certified:assert P[b]<P[0]+1e-8
   records.append(dict(case=row['name'],plan=snap['plan'],rho=rho,status=status,C=C.tolist(),J=J.tolist(),P=P.tolist(),epsilon=eps,delta=delta,normalized_delta=float(delta/max(J[0],1e-15)),opportunity_slack=slack,certified=certified,reference_improvement=float(P[0]-P[b])))
  print(row['name'],len(snapshots),'snapshots',flush=True)
 summary=dict(n=len(records),cases=len(rows),accepted=sum(r['status']=='accepted' for r in records),certified_accepted=sum(r['status']=='accepted' and r['certified'] for r in records),min_slack=min(r['opportunity_slack'] for r in records),median_normalized_delta=float(np.median([r['normalized_delta'] for r in records])))
 (out/'analysis.json').write_text(json.dumps(dict(summary=summary,records=records),indent=2));print(summary)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);run(p.parse_args().output)
