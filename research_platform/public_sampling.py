"""Task-adapted NumPy PS, MPPI and iCEM; sources in README.md (Public baselines and attribution).
The plant/model and noisy information interface match the envelope benchmark.
This is an algorithm port, not execution of the authors' full software stacks.
"""
import numpy as np
import mujoco
from mujoco import rollout
from research_platform.polygon_push import make_polygon_model, PolygonGeometry


def bound(u):
    """Common Euclidean speed bound, including sampled and averaged actions."""
    return u * np.minimum(1., .15 / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-15))


def mppi_update(mean, samples, costs, sigma, temperature):
    noise = samples - mean
    total = costs + temperature * np.sum(mean[None] * noise / sigma**2, axis=(1, 2))
    w = np.exp(-(total - total.min()) / temperature)
    w /= w.sum()
    return bound(mean + np.einsum('n,nhd->hd', w, noise)), w


def colored_noise(rng, n, horizon, beta=1.):
    """Timmer-Koenig Gaussian power-law noise, expected unit variance.
    Same spectral construction as colorednoise.powerlaw_psd_gaussian used by iCEM.
    No per-realization normalization (which would change Gaussian sampling).
    """
    f = np.fft.rfftfreq(horizon)
    f = np.maximum(f, 1. / horizon)
    scale = f ** (-beta / 2.)
    w = scale[1:].copy()
    w[-1] *= (1 + (horizon % 2)) / 2.
    norm = 2 * np.sqrt(np.sum(w*w)) / horizon
    re = rng.normal(size=(n, 2, len(f))) * scale
    im = rng.normal(size=re.shape) * scale
    re[..., 0] *= np.sqrt(2); im[..., 0] = 0
    if horizon % 2 == 0:
        re[..., -1] *= np.sqrt(2); im[..., -1] = 0
    return (np.fft.irfft(re + 1j*im, n=horizon, axis=-1) / norm).transpose(0, 2, 1)


class PublicSampler:
    def __init__(self, vertices, method, sampling_seed=0, settings=None):
        self.settings = dict(samples=128, knots=12, hold=5, sigma=.08,
                             temperature=1., contact_weight=200., iterations=3,
                             elites=10, decay=1.5, alpha=.1, beta=1., reuse=.3)
        self.settings.update(settings or {})
        self.method = method
        self.rng = np.random.default_rng(sampling_seed)
        self.model = make_polygon_model(vertices, .3)
        self.data = mujoco.MjData(self.model)
        self.geometry = PolygonGeometry(vertices)
        self.engine = rollout.Rollout(nthread=0)
        self.mean = np.zeros((self.settings['knots'], 2))
        self.elite = None
        self.step = 0; self.plans = 0; self.rollout_steps = 0
        self.plan_records = []; self.command = np.zeros(2)
        self.lock = -1; self.phase = 'sampling'

    def cost(self, observation, goal, controls):
        s = np.r_[0., observation.qpos, observation.qvel][None]
        physics = np.repeat(controls, self.settings['hold']*20, axis=1)
        states, _ = self.engine.rollout(self.model, self.data, s, physics)
        states = states[:, 19::20]
        self.rollout_steps += states.shape[0]*states.shape[1]
        # FULLPHYSICS: time, object xyz, quaternion wxyz, pusher xy, qvel.
        quat = states[..., 4:8]
        yaw = np.arctan2(2*(quat[..., 0]*quat[..., 3]+quat[..., 1]*quat[..., 2]),
                         1-2*(quat[..., 2]**2+quat[..., 3]**2))
        delta = yaw-goal[2]; delta = np.arctan2(np.sin(delta), np.cos(delta))
        v = 200*np.sum((states[..., 1:3]-goal[:2])**2, axis=-1)+2*delta**2
        # Smooth contact acquisition shaping: distance to the nominal polygon,
        # minus the pusher radius. It is zero at or inside predicted contact.
        dp = states[..., 8:10]-states[..., 1:3]
        co, si = np.cos(yaw), np.sin(yaw)
        local = np.stack([co*dp[..., 0]+si*dp[..., 1], -si*dp[..., 0]+co*dp[..., 1]], axis=-1)
        a = self.geometry.vertices; edge = np.roll(a,-1,axis=0)-a
        rel = local[..., None, :]-a
        t = np.clip(np.sum(rel*edge, axis=-1)/np.sum(edge*edge, axis=-1),0,1)
        distance = np.min(np.linalg.norm(rel-t[..., None]*edge, axis=-1),axis=-1)
        outside = np.max(local @ self.geometry.normal.T-self.geometry.bound, axis=-1)>0
        gap = np.where(outside,np.maximum(distance-.02,0),0)
        shaped = self.settings['contact_weight']*gap**2
        cost = np.sum(v+shaped,axis=1)+60*v[:,-1]
        return np.where(np.isfinite(cost),cost,1e12)

    def plan(self, observation, goal):
        p = self.settings; h = p['knots']; n = p['samples']
        if self.plans:
            self.mean = np.concatenate([self.mean[1:], self.mean[-1:]])
        before = self.rollout_steps
        if self.method in ('ps','mppi'):
            samples = bound(self.mean + self.rng.normal(0,p['sigma'],(n,h,2)))
            if self.method == 'ps':
                samples = np.concatenate([self.mean[None],samples])
            costs = self.cost(observation,goal,samples)
            if self.method == 'ps': self.mean = samples[np.argmin(costs)].copy()
            else: self.mean,_ = mppi_update(self.mean,samples,costs,p['sigma'],p['temperature'])
            self.command = self.mean[0].copy()
        elif self.method == 'icem':
            std = np.full_like(self.mean,p['sigma']); old_elite = self.elite
            for i in range(p['iterations']):
                if i: n = max(2*p['elites'],int(n/p['decay']))
                samples = bound(self.mean + std*colored_noise(self.rng,n,h,p['beta']))
                if i==p['iterations']-1: samples[0]=self.mean
                if i==0 and old_elite is not None:
                    keep = old_elite[:int(len(old_elite)*p['reuse'])]
                    tail = bound(self.mean + std*colored_noise(self.rng,len(keep),h,p['beta']))[:,-1:]
                    samples = np.concatenate([samples,np.concatenate([keep[:,1:],tail],axis=1)])
                costs = self.cost(observation,goal,samples)
                if i:
                    k = int(len(self.elite)*p['reuse'])
                    samples = np.concatenate([samples,self.elite[:k]])
                    costs = np.r_[costs,elite_costs[:k]]
                ix = np.argsort(costs)[:p['elites']]
                self.elite = samples[ix].copy(); elite_costs=costs[ix]
                self.mean = p['alpha']*self.mean+(1-p['alpha'])*self.elite.mean(axis=0)
                std = p['alpha']*std+(1-p['alpha'])*self.elite.std(axis=0)
            self.command = self.elite[0,0].copy()
        else: raise ValueError(self.method)
        self.plan_records.append(dict(step=self.step, best_cost=float(costs.min()),
                                      rollout_steps=self.rollout_steps-before))
        self.plans += 1

    def action(self, observation, goal):
        if self.step % self.settings['hold']==0: self.plan(observation,np.asarray(goal))
        self.step += 1
        return self.command.copy()
