"""CEM-MPC and Smooth MPPI on the common planar-pushing prediction backend.
Pinned author sources: README.md (Public baselines and attribution); notices in licenses/.
"""
import numpy as np
from scipy.stats import truncnorm
from research_platform.public_sampling import PublicSampler,bound


def smppi_samples(action, increment, noise, zero_noise):
    perturb=np.concatenate([increment+noise,zero_noise])
    perturb=np.clip(perturb,-.15,.15)
    samples=bound(action+perturb)
    effective=samples-action-increment
    return samples,effective


def smppi_update(action, increment, samples, noise, costs, sigma, temperature, gamma, smooth_weight):
    total=costs+gamma*np.sum(increment*noise/sigma**2,axis=(1,2))
    total+=smooth_weight*np.sum(np.diff(samples,axis=1)**2,axis=(1,2))
    weights=np.exp(-(total-total.min())/temperature);weights/=weights.sum()
    new_increment=increment+np.einsum('n,nhd->hd',weights,noise)
    return bound(action+new_increment),new_increment,weights


class ExtraPublicSampler(PublicSampler):
    def __init__(self,vertices,method,sampling_seed=0,settings=None):
        defaults=dict(samples=128,knots=12,hold=5,sigma=.06,temperature=1.,
                      contact_weight=200.,iterations=3,elites=10,alpha=.25,
                      variance_epsilon=.001*.15**2,smooth_weight=1.,gamma=1.)
        defaults.update(settings or {})
        super().__init__(vertices,method,sampling_seed,defaults)
        self.increment=np.zeros_like(self.mean)

    def plan(self,observation,goal):
        p=self.settings;n=p['samples'];h=p['knots'];before=self.rollout_steps
        if self.plans:
            self.mean=np.concatenate([self.mean[1:],self.mean[-1:]])
            self.increment=np.concatenate([self.increment[1:],np.zeros((1,2))])
        if self.method=='cem':
            variance=np.full_like(self.mean,p['sigma']**2);iterations=0;best=float('inf')
            while iterations<p['iterations'] and np.max(variance)>p['variance_epsilon']:
                constrained=np.minimum(variance,np.minimum(((self.mean+.15)/2)**2,((.15-self.mean)/2)**2))
                noise=truncnorm.rvs(-2,2,size=(n,h,2),random_state=self.rng)
                samples=bound(self.mean+noise*np.sqrt(constrained))
                costs=self.cost(observation,goal,samples)
                elites=samples[np.argsort(costs)[:p['elites']]]
                self.mean=p['alpha']*self.mean+(1-p['alpha'])*elites.mean(axis=0)
                variance=p['alpha']*variance+(1-p['alpha'])*elites.var(axis=0)
                best=min(best,float(costs.min()));iterations+=1
            self.command=self.mean[0].copy()
        elif self.method=='smppi':
            # Normalized optimizer integration step = 1, as recommended in the
            # author's FAQ. Physical knot duration remains 0.25 s.
            near=round(n*.99);fresh=round(n*.01);assert near+fresh==n
            samples,noise=smppi_samples(self.mean,self.increment,
                self.rng.normal(0,p['sigma'],(near,h,2)),self.rng.normal(0,p['sigma'],(fresh,h,2)))
            costs=self.cost(observation,goal,samples)
            self.mean,self.increment,_=smppi_update(self.mean,self.increment,samples,noise,costs,
                p['sigma'],p['temperature'],p['gamma'],p['smooth_weight'])
            self.command=self.mean[0].copy();best=float(costs.min());iterations=1
        else:raise ValueError(self.method)
        self.plan_records.append(dict(step=self.step,best_cost=best,iterations=iterations,rollout_steps=self.rollout_steps-before))
        self.plans+=1
