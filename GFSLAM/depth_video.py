import numpy as np
import torch
import lietorch
import gfslam_backends
from lietorch import SE3, SO3

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from gfslam_net import cvx_upsample
import geom.projective_ops as pops

import copy

###### 3DGS ######
from utils.camera_utils import Camera
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov

class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):
                
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        self.max_uid = 1e8

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        # self.red = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=device, dtype=torch.float).share_memory_()
        # self.oldest_window_uids = (torch.ones(buffer, device=device, dtype=torch.int32) * self.max_uid).share_memory_()
        # self.latest_window_uids = (torch.ones(buffer, device=device, dtype=torch.int32) * -1).share_memory_()

        self.gs_viewpoints = [None] * buffer
        # ---- fast_mode async: stable KF identity ----
        # `kf_uids[slot]` is the UID currently living at that slot, or -1 if
        # the slot is empty / tombstoned. `uid_to_slot[uid]` resolves a UID
        # to its current slot (after any rm_keyframe shift). Only valid for
        # `slot < counter.value`; mapping callers must hold the video lock
        # while they read both. `_next_uid` monotonically increments.
        self._next_uid = 0
        self.kf_uids = [-1] * buffer
        self.uid_to_slot = {}
        self.fake_gt_pose = np.eye(4)
        self.device = device

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device=device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device=device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device=device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device)
        self.skip_exposure_copy = True
        
    def to(self, device="cuda"):
        self.tstamp = self.tstamp.to(device=device)
        self.images = self.images.to(device=device)
        self.dirty = self.dirty.to(device=device)
        # self.red = self.red.to(device=device)
        self.poses = self.poses.to(device=device)
        self.disps = self.disps.to(device=device)
        self.disps_sens = self.disps_sens.to(device=device)
        self.disps_up = self.disps_up.to(device=device)
        self.intrinsics = self.intrinsics.to(device=device)
        self.oldest_window_id = self.oldest_window_id.to(device=device)

        self.fmaps = self.fmaps.to(device=device)
        self.nets = self.nets.to(device=device)
        self.inps = self.inps.to(device=device)

        return self

    def __del__(self):
        # delete all tensors
        del self.tstamp
        del self.images
        del self.dirty
        # del self.red
        del self.poses
        del self.disps
        del self.disps_sens
        del self.disps_up
        del self.intrinsics
        del self.fmaps
        del self.nets
        del self.inps
        
        # del self.oldest_window_uids
        # del self.latest_window_uids
        del self.gs_viewpoints

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8].cuda()
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6 and item[6] is not None:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

        ### Set gs_viewpoint
        if self.counter.value > 0 and not hasattr(self, 'projection_matrix') and item[5] is not None:
            self.fx, self.fy, self.cx, self.cy = item[5].cpu().numpy() * 8
            self.projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, 
                                                          fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, 
                                                          W=self.wd, H=self.ht).transpose(0, 1).cuda()
            self.fovx = focal2fov(self.fx, self.wd)
            self.fovy = focal2fov(self.fy, self.ht)
            
        key_viewpoint = Camera(item[0], item[1] / 255.0, 1.0 / item[3] if item[3] is not None else None, 
                               self.fake_gt_pose, self.projection_matrix,
                               self.fx, self.fy, self.cx, self.cy,
                               self.fovx, self.fovy, self.ht, self.wd, self.device)
        if item[2] is not None:
            Tcw_tensor = SE3(item[2]).matrix()
            key_viewpoint.update_RT(Tcw_tensor[:3, :3], Tcw_tensor[:3, 3])
        if self.counter.value > 1 and not self.skip_exposure_copy:
            prev_viewpoint = self.gs_viewpoints[self.counter.value - 2]
            key_viewpoint.exposure_a.data.copy_(prev_viewpoint.exposure_a.data)
            key_viewpoint.exposure_b.data.copy_(prev_viewpoint.exposure_b.data)
        self.gs_viewpoints[index] = key_viewpoint

        # Assign a stable UID to this KF slot. Mapping queue entries reference
        # KFs by uid; rm_keyframe later compacts uids alongside the slot
        # tensors so `uid_to_slot[uid]` always points at the current slot
        # (or returns None if the KF was retired).
        if isinstance(index, int):
            uid = self._next_uid
            self._next_uid += 1
            self.kf_uids[index] = uid
            self.uid_to_slot[uid] = index

    def shift(self, ix, n=1):
        with self.get_lock():
            self.tstamp[ix+n:self.counter.value+n] = self.tstamp[ix:self.counter.value].clone()
            self.images[ix+n:self.counter.value+n] = self.images[ix:self.counter.value].clone()
            self.dirty[ix+n:self.counter.value+n] = self.dirty[ix:self.counter.value].clone()
            self.poses[ix+n:self.counter.value+n] = self.poses[ix:self.counter.value].clone()
            self.disps[ix+n:self.counter.value+n] = self.disps[ix:self.counter.value].clone()
            self.disps_up[ix+n:self.counter.value+n] = self.disps_up[ix:self.counter.value].clone()
            self.intrinsics[ix+n:self.counter.value+n] = self.intrinsics[ix:self.counter.value].clone()
            self.fmaps[ix+n:self.counter.value+n] = self.fmaps[ix:self.counter.value].clone()
            self.nets[ix+n:self.counter.value+n] = self.nets[ix:self.counter.value].clone()
            self.inps[ix+n:self.counter.value+n] = self.inps[ix:self.counter.value].clone()
            self.gs_viewpoints[ix+n:self.counter.value+n] = self.gs_viewpoints[ix:self.counter.value]            
            # self.gs_viewpoints[ix+n:self.counter.value+n] = [
            #     copy.deepcopy(vp) if vp is not None else None
            #     for vp in self.gs_viewpoints[ix:self.counter.value]
            # ]
            self.counter.value += n

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """
        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.dirty[:self.counter.value] = True

    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = gfslam_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = gfslam_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = gfslam_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d
    
    def distance_covis(self, ii=None):
        """ frame distance metric based on covisibility """
        ii = torch.as_tensor(ii)
        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        poses = self.poses[:self.counter.value].clone()
        d = gfslam_backends.covis_distance(poses, self.disps, self.intrinsics[0], ii)
        d = d * (1. / self.disps[ii].median())
        return d
    
    def set_sliding_uids(self, indices):
        min_uid = int(self.tstamp[indices].min().item())
        max_uid = int(self.tstamp[indices].max().item())
        # mask_min = (self.oldest_window_uids[indices] > min_uid)
        # mask_max = (self.latest_window_uids[indices] < max_uid)
        # self.oldest_window_uids[indices[mask_min]] = min_uid
        # self.latest_window_uids[indices[mask_max]] = max_uid    
    
    def ba_gsflow(self, coords1, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=True):
        with self.get_lock():
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            gfslam_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

            self.disps.clamp_(min=0.001)
