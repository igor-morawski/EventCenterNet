import numpy as np

module_imports =  ['from {}prophesee.src.io.psee_loader import PSEELoader',
    'from {}prophesee.src.io.box_loading import reformat_boxes']
for module_import in module_imports:
    try:
        exec(module_import.format("."))
    except ImportError: 
        exec(module_import.format(""))

import os.path as op
import torch

def us_to_s(us):
    return us/1000000

def us_to_ms(us):
    return us/1000

def s_to_us(us):
    return us*1000000

def ms_to_us(us):
    return us*1000

def calc_floor_ceil_delta(x): 
    # XXX confirm that this is consistent with the original implementation
    # github.com/alexzzhu/EventGAN
    x_fl = torch.floor(x.float() + 1e-8).long()
    x_ce = torch.ceil(x.float() - 1e-8).long()
    x_ce_fake = (torch.floor(x.float()) + 1).long()

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]

def create_update(x, y, t, dt, p, vol_size):
    # github.com/alexzzhu/EventGAN
    assert (x>=0).byte().all() and (x<vol_size[2]).byte().all()
    assert (y>=0).byte().all() and (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() and (t<vol_size[0] // 2).byte().all()

    vol_mul = torch.where(p < 0,
                          torch.ones(p.shape, dtype=torch.long) * vol_size[0] // 2,
                          torch.zeros(p.shape, dtype=torch.long))

    inds = (vol_size[1]*vol_size[2]) * (t + vol_mul)\
         + (vol_size[2])*y\
         + x

    vals = dt

    return inds, vals

def events_to_voxel(events, bins, height, width, device=torch.device('cuda:0')):
    # github.com/alexzzhu/EventGAN
    vol_size = [2*bins, height, width]
    npts = events.shape[0]
    volume = torch.zeros(*vol_size).cpu() # TODO: confirm
    
    x = torch.Tensor(events['x'].astype('long')).cpu().long()
    y = torch.Tensor(events['y'].astype('long')).cpu().long()
    t = torch.Tensor(events['t'].astype('long')).cpu().long()
    p = torch.Tensor(events['p'].astype('long')).cpu().long()
    
    t_min = t.min()
    t_max = t.max()
    t_scaled = (t-t_min) * ((vol_size[0] // 2-1) / (t_max-t_min))
    
    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    
    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],
                                     p,
                                     vol_size)
    
    volume.view(-1).put_(inds_fl, vals_fl.float(), accumulate=True)
    return volume

def video_segment_to_voxel(video, t0, delta_t, bins):
    current_time = video.current_time
    video.seek_time(t0)
    events = video.load_delta_t(delta_t)
    video.seek_time(current_time)
    return events_to_voxel(events, bins, *video.get_size())

def video_segment_anno_to_bboxes(anno, t0, delta_t):
    current_time = anno.current_time
    anno.seek_time(t0)
    bboxes = anno.load_delta_t(delta_t)
    anno.seek_time(current_time)
    return bboxes