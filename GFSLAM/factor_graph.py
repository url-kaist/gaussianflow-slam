import torch
import lietorch
import numpy as np

import os, matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops

from cuda_timer import CudaTimer
from functools import partial
from utils import flow_utils
import time

if torch.__version__.startswith("2"):
    autocast = partial(torch.autocast, device_type="cuda")
else:
    autocast = torch.cuda.amp.autocast


class FactorGraph:
    def __init__(self, video, update_op, device="cuda", corr_impl="volume", max_factors=-1, upsample=True):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl
        self.upsample = True #upsample

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps) # (buffer, video.ht//8 , video.wd//8)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        ## For dealing flow
        self.flow_ups = torch.zeros([1, 0, video.ht, video.wd, 2], device=device, dtype=torch.float)
        self.weight_ups = torch.zeros([1, 0, video.ht, video.wd, 2], device=device, dtype=torch.float)
        # self.flow_confs = torch.zeros([1, 0, video.ht, video.wd], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.flow_ups_inac = torch.zeros([1, 0, video.ht, video.wd, 2], device='cpu', dtype=torch.float)
        self.weight_ups_inac = torch.zeros([1, 0, video.ht, video.wd, 2], device='cpu', dtype=torch.float)
        # self.flow_confs_inac = torch.zeros([1, 0, video.ht, video.wd], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        """remove duplicate edges"""
        # filter if (ii, jj) insersect with any active edge
        if len(self.ii) > 0:
            mask = ((ii[:, None] == self.ii) & (jj[:, None] == self.jj)).any(dim=-1)
            ii = ii[~mask]
            jj = jj[~mask]

        # filter if (ii, jj) intersect with any inactive edge
        if len(self.ii_inac) > 0:
            mask = ((ii[:, None] == self.ii_inac) & (jj[:, None] == self.jj_inac)).any(
                dim=-1
            )
            ii = ii[~mask]
            jj = jj[~mask]

        return ii, jj

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0,2,3,4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """add edges to factor graph"""

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if (
            self.max_factors > 0
            and self.ii.shape[0] + ii.shape[0] > self.max_factors
            and self.corr is not None
            and remove
        ):

            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii, 0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj, c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)
        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

        ## For dealing flow
        zero_flow_up = torch.zeros([1, target.shape[1], self.video.ht, self.video.wd, 2], device=self.device, dtype=torch.float)
        self.flow_ups = torch.cat([self.flow_ups, zero_flow_up], 1) 
        self.weight_ups = torch.cat([self.weight_ups, torch.zeros_like(zero_flow_up)], 1)

        # one_flow_confs = torch.ones([1, target.shape[1], self.video.ht, self.video.wd], device=self.device, dtype=torch.float)
        # self.flow_confs = torch.cat([self.flow_confs, one_flow_confs], 1)

    @autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            # self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            # self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            # self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            # self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)
            # self.flow_ups_inac = torch.cat([self.flow_ups_inac, self.flow_ups[:, mask]], 1)
            # self.weight_ups_inac = torch.cat([self.weight_ups_inac, self.weight_ups[:, mask]], 1)

            idx = torch.nonzero(mask, as_tuple=True)[0].to(self.device)
            self.ii_inac     = torch.cat([self.ii_inac, self.ii.index_select(0, idx)], 0)
            self.jj_inac     = torch.cat([self.jj_inac, self.jj.index_select(0, idx)], 0)
            self.target_inac = torch.cat([self.target_inac, self.target.index_select(1, idx)], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight.index_select(1, idx)], 1)
            if self.flow_ups_inac.device.type != 'cpu':
                self.flow_ups_inac = self.flow_ups_inac.to('cpu', dtype=torch.float16)
            if self.weight_ups_inac.device.type != 'cpu':
                self.weight_ups_inac = self.weight_ups_inac.to('cpu', dtype=torch.float16)
            add_flow_ups   = self.flow_ups.index_select(1, idx).detach().to('cpu', dtype=torch.float16, non_blocking=True)
            add_weight_ups = self.weight_ups.index_select(1, idx).detach().to('cpu', dtype=torch.float16, non_blocking=True)
            self.flow_ups_inac   = torch.cat([self.flow_ups_inac,   add_flow_ups],   1)
            self.weight_ups_inac = torch.cat([self.weight_ups_inac, add_weight_ups], 1)
            # self.flow_confs_inac = torch.cat([self.flow_confs_inac, self.flow_confs[:, mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]

        ## For debugging flow
        self.flow_ups = self.flow_ups[:,~mask] 
        self.weight_ups = self.weight_ups[:,~mask]
        # self.flow_confs = self.flow_confs[:,~mask]


    @autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """

        t = self.video.counter.value
        # with self.video.get_lock():
        self.video.images[ix : t - 1] = self.video.images[ix + 1 : t].clone()
        self.video.poses[ix : t - 1] = self.video.poses[ix + 1 : t].clone()
        self.video.disps[ix : t - 1] = self.video.disps[ix + 1 : t].clone()
        self.video.disps_sens[ix : t - 1] = self.video.disps_sens[ix + 1 : t].clone()
        # disps_up was missed in legacy rm_keyframe — fine for sync mode
        # because disps_up was consumed before rm_keyframe ran. fast_mode
        # async insert reads disps_up at queue-pop time, which can be
        # AFTER rm_keyframe → without this shift, the slot returns stale
        # data from a retired KF, hurting create_new_gaussians depth
        # init and reducing inserted point count.
        self.video.disps_up[ix : t - 1] = self.video.disps_up[ix + 1 : t].clone()
        self.video.intrinsics[ix : t - 1] = self.video.intrinsics[ix + 1 : t].clone()

        self.video.nets[ix : t - 1] = self.video.nets[ix + 1 : t].clone()
        self.video.inps[ix : t - 1] = self.video.inps[ix + 1 : t].clone()
        self.video.fmaps[ix : t - 1] = self.video.fmaps[ix + 1 : t].clone()
        self.video.tstamp[ix: t - 1] = self.video.tstamp[ix + 1 : t].clone()

        self.video.gs_viewpoints[ix : t - 1] = self.video.gs_viewpoints[ix + 1 : t]
        self.video.gs_viewpoints[t - 1] = None

        # ---- Layer 1 (UID): keep kf_uids / uid_to_slot consistent with shift ----
        # The KF that lived at slot `ix` is retired (its uid is invalidated);
        # every UID at slot > ix moves down by one. Mapping reads uid_to_slot
        # and treats `None` as "this KF is gone, drop the task".
        retired_uid = self.video.kf_uids[ix]
        if retired_uid >= 0:
            self.video.uid_to_slot.pop(retired_uid, None)
        # shift kf_uids[ix+1:t] → kf_uids[ix:t-1] in python
        self.video.kf_uids[ix : t - 1] = self.video.kf_uids[ix + 1 : t]
        self.video.kf_uids[t - 1] = -1
        # rebuild uid_to_slot for the affected range; cheaper than tracking
        # individual moves and avoids ordering bugs (all uids at slot >= ix
        # got their slot decremented).
        for new_slot in range(ix, t - 1):
            uid = self.video.kf_uids[new_slot]
            if uid >= 0:
                self.video.uid_to_slot[uid] = new_slot

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:, ~m]
            self.weight_inac = self.weight_inac[:, ~m]
            idx = torch.nonzero(~m, as_tuple=True)[0].to('cpu')
            self.flow_ups_inac = torch.index_select(self.flow_ups_inac, 1, idx)
            self.weight_ups_inac = torch.index_select(self.weight_ups_inac, 1, idx)

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)

    @autocast(enabled=True)
    def update_by_gs(self, coords1_by_gsflow, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, process_dba=False, selected_mask=None):
        ## coords1: (1, E, ht, wd, 2)
        with autocast(enabled=False):
            if selected_mask is not None:
                coords1, mask = self.video.reproject(self.ii, self.jj)
                idx = torch.nonzero(selected_mask, as_tuple=True)[0]
                src = torch.index_select(coords1_by_gsflow, 1, idx)
                coords1.index_copy_(1, idx, src)
                del idx, src
                # coords1[:, selected_mask, ...] = coords1_by_gsflow[:, selected_mask, ...]
            else:
                # coords1, mask = self.video.reproject(self.ii, self.jj)
                coords1 = coords1_by_gsflow

            # motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            # motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
                
            B, E, ht, wd, _ = coords1.shape
            motn = torch.empty((B, E, 4, ht, wd), device=coords1.device, dtype=coords1.dtype)
            tmp = (coords1 - self.coords0).permute(0,1,4,2,3).contiguous()  # [B,E,2,ht,wd]
            motn[:,:,0:2,:,:] = tmp
            tmp = (self.target - coords1).permute(0,1,4,2,3).contiguous()   # [B,E,2,ht,wd]
            motn[:,:,2:4,:,:] = tmp
            del tmp
            motn.clamp_(-64.0, 64.0)
        
        corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]
            self.damping[torch.unique(self.ii)] = damping

            if use_inactive:
                # # m = (self.ii_inac >= t0) & (self.jj_inac >= t0)
                # ii = torch.cat([self.ii_inac[m], self.ii], 0)
                # jj = torch.cat([self.jj_inac[m], self.jj], 0)
                # target = torch.cat([self.target_inac[:,m], self.target], 1)
                # weight = torch.cat([self.weight_inac[:,m], self.weight], 1)
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                midx = torch.nonzero(m, as_tuple=True)[0]
                ii_inac_sel = torch.index_select(self.ii_inac, 0, midx)
                jj_inac_sel = torch.index_select(self.jj_inac, 0, midx)
                target_inac_sel = torch.index_select(self.target_inac, 1, midx)
                weight_inac_sel = torch.index_select(self.weight_inac, 1, midx)
                ii = torch.cat([ii_inac_sel, self.ii], 0)
                jj = torch.cat([jj_inac_sel, self.jj], 0)
                target = torch.cat([target_inac_sel, self.target], 1)
                weight = torch.cat([weight_inac_sel, self.weight], 1)
                del midx, ii_inac_sel, jj_inac_sel, target_inac_sel, weight_inac_sel
            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight
            
            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP
            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            if self.upsample:
                self.video.upsample(torch.unique(self.ii), upmask)

                # For upsampling flow
                self.flow_ups = flow_utils.upsample_flow(
                    self.target, self.coords0, upmask, self.ii
                ).unsqueeze(0)
                self.weight_ups = flow_utils.upsample_weight(
                    self.weight, upmask, self.ii
                ).unsqueeze(0)
            
            if process_dba:
                self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                itrs=itrs, lm=1e-4, ep=0.1, motion_only=False)
                self.age += 1
            
    
    @autocast(enabled=True)
    def solve_poses_cuda(self, coords1, itrs=2, use_inactive=False, EP=1e-7):
        with autocast(enabled=False):
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

        corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)
        
        t0 = max(1, self.ii.min().item()+1)
        t1 = None

        with autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]
            self.damping[torch.unique(self.ii)] = damping

            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)

            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight


            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            coords1_gsflow = coords1.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            if self.upsample:
                self.video.upsample(torch.unique(self.ii), upmask)

                # For debugging flow (upsampling)
                self.flow_ups = flow_utils.upsample_flow(
                    self.target, self.coords0, upmask, self.ii
                ).unsqueeze(0)
                self.weight_ups = flow_utils.upsample_weight(
                    self.weight, upmask, self.ii
                ).unsqueeze(0)

            # dense bundle adjustment
            self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                itrs=itrs, lm=1e-4, ep=0.1, motion_only=True)
            
            self.video.ba_gsflow(coords1_gsflow, target, weight, damping, ii, jj, t0, t1, 
                itrs=itrs, lm=1e-4, ep=0.1, motion_only=True)
    


    @autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False):
        """ run update operator on factor graph """

        # motion features
        with autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

        
        # correlation features
        corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)

            ht, wd = self.coords0.shape[0:2]
            self.damping[torch.unique(self.ii)] = damping

            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)

            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight


            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()

            if self.upsample:
                self.video.upsample(torch.unique(self.ii), upmask)

                # For debugging flow (upsampling)
                self.flow_ups = flow_utils.upsample_flow(
                    self.target, self.coords0, upmask, self.ii
                ).unsqueeze(0)
                self.weight_ups = flow_utils.upsample_weight(
                    self.weight, upmask, self.ii
                ).unsqueeze(0)

            # dense bundle adjustment
            self.video.ba(target, weight, damping, ii, jj, t0, t1, 
                itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
        
        self.age += 1

    @autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in range(steps):
            with CudaTimer("backend", enabled=False):
                with autocast(enabled=False):
                    coords1, mask = self.video.reproject(self.ii, self.jj)
                    motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                    motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

                s = 8
                for i in range(self.ii.min(), self.jj.max()+1, s):
                    v = (self.ii >= i) & (self.ii < i + s)
                    iis = self.ii[v]
                    jjs = self.jj[v]

                    if v.count_nonzero().item() == 0:
                        continue

                    ht, wd = self.coords0.shape[0:2]

                    with autocast(enabled=True):
                        corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                        net, delta, weight, damping, upmask = \
                            self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)

                        if self.upsample:
                            self.video.upsample(torch.unique(iis), upmask)

                    self.net[:,v] = net
                    self.target[:,v] = coords1[:,v] + delta.float()
                    self.weight[:,v] = weight.float()
                    self.damping[torch.unique(iis)] = damping

                damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP

                if use_inactive:
                    ii = torch.cat([self.ii_inac, self.ii], 0)
                    jj = torch.cat([self.jj_inac, self.jj], 0)
                    target = torch.cat([self.target_inac, self.target], 1)
                    weight = torch.cat([self.weight_inac, self.weight], 1)

                else:
                    ii, jj, target, weight = self.ii, self.jj, self.target, self.weight

                damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP
                target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
                weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
                
                self.age += 1

                # dense bundle adjustment
                self.video.ba(target, weight, damping, ii, jj, 1, t, 
                    itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

                self.video.dirty[:t] = True

    def add_neighborhood_factors(self, t0, t1, r=3):
        """add edges between neighboring frames within radius r"""

        ii, jj = torch.meshgrid(
            torch.arange(t0, t1, device=self.device),
            torch.arange(t0, t1, device=self.device),
            indexing="ij",
        )

        c = 1 if self.video.stereo else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    def add_proximity_factors(
        self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False, min_gap_sample=3,
    ):
        """add edges to the factor graph based on distance"""

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx, indexing="ij")
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta).cpu()
        # d[ii - rad < jj] = np.inf 
        d[ii - min_gap_sample < jj] = np.inf 
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):
                    if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1 - t0) * (t - t1) + (j1 - t1)] = np.inf

        es = []
        for i in range(t0, t):
            if self.video.stereo:
                es.append((i, i))
                d[(i - t0) * (t - t1) + (i - t1)] = np.inf

            for j in range(max(i - rad - 1, 0), i):
                es.append((i, j))
                es.append((j, i))
                d[(i - t0) * (t - t1) + (j - t1)] = np.inf

        ix = torch.argsort(d) #descending=True
        for k in ix:
            if d[k] > thresh:
                continue

            if self.max_factors > 0:
                if len(es) > self.max_factors:
                    break

            i = ii[k]
            j = jj[k]

            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms + 1):
                for dj in range(-nms, nms + 1):
                    if abs(di) + abs(dj) <= max(min(abs(i - j) - 2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1 - t0) * (t - t1) + (j1 - t1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)

    def update_flowconfs(self, flowconfs, mask=None):
        if mask is not None:
            self.flow_confs[:, mask, ...] = flowconfs.unsqueeze(0)
        else:
            self.flow_confs = flowconfs.unsqueeze(0)
