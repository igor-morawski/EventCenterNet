import os
import sys
import time
import argparse
import copy 
import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np

import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from nets.recurrent_hourglass import get_hourglass
from nets.resdcn import get_pose_net

from utils.utils import _tranpose_and_gather_feature_t, _tranpose_and_gather_feature, load_model
from utils.image import transform_preds
from utils.losses import _neg_loss_t, _reg_loss_t
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.post_process import ctdet_decode

from utils.dataloader import CSVDataset, collater

HW = (240, 304)
PROPH_STRUCTURED_ARRAY = [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')]

# Training settings
parser = argparse.ArgumentParser(description='simple_centernet45')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default="/tmp2/igor/EV/Dataset/Automotive/")
parser.add_argument('--log_name', type=str, default='test')
parser.add_argument('--pretrain_name', type=str, default='pretrain')

parser.add_argument('--dataset', type=str, default='CSVDataset', choices=['CSVDataset'])
parser.add_argument('--train_csv_file', type=str)
parser.add_argument('--val_csv_file', type=str)
parser.add_argument('--class_list_file', type=str)
parser.add_argument('--trim_to_shortest', action="store_true")
parser.add_argument('--delta_t', type=int)
parser.add_argument('--frames_per_batch', type=int)
parser.add_argument('--bins', type=int)
parser.add_argument('--arch', type=str, default='small_hourglass')

parser.add_argument('--img_h', type=int, default=240)
parser.add_argument('--img_w', type=int, default=304)
parser.add_argument('--split_ratio', type=float, default=1.0)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_step', type=str, default='90,120')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_epochs', type=int, default=140)

# parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=2)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.pretrain_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_name, 'checkpoint.t7')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]


def main():
  saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
  logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
  summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
  print = logger.info
  print(cfg)

  torch.manual_seed(317)
  torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training

  num_gpus = torch.cuda.device_count() 
  if cfg.dist:
    raise NotImplementedError
    cfg.device = torch.device('cuda:%d' % cfg.local_rank)
    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=num_gpus, rank=cfg.local_rank)
  else:
    cfg.device = torch.device('cuda')

  print('Setting up data...')
  if cfg.dataset == "CSVDataset":
    Dataset = CSVDataset 
    train_cfg = {"csv_file": cfg.train_csv_file, "class_list": cfg.class_list_file,
               "batch_size": cfg.batch_size, "data_root": cfg.data_dir,
               "trim_to_shortest": cfg.trim_to_shortest, "delta_t": cfg.delta_t, 
               "frames_per_batch": cfg.frames_per_batch, "bins": cfg.bins, 
               "hw" : HW}
    val_cfg = copy.deepcopy(train_cfg)
    val_cfg["csv_file"] = cfg.val_csv_file
    val_cfg["val"] = True
    val_cfg["trim_to_shortest"] = False
    val_cfg["batch_size"] = 1
    val_cfg["frames_per_batch"] = cfg.frames_per_batch * cfg.batch_size # XXX ?
  else:
    raise NotImplementedError
  train_dataset = Dataset(**train_cfg)
  # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
  #                                                                 num_replicas=num_gpus,
  #                                                                 rank=cfg.local_rank)
  # XXX
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collater, batch_size=train_cfg["batch_size"], shuffle=False)

  val_dataset = Dataset(**val_cfg)
  val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=collater, batch_size=val_cfg["batch_size"], shuffle=False)
  print('Creating model...')
  if 'hourglass' in cfg.arch:
    model = get_hourglass(cfg.arch, cfg.bins*2, train_dataset.num_classes)
  elif 'resdcn' in cfg.arch:
    model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=train_dataset.num_classes)
  else:
    raise NotImplementedError

  if cfg.dist:
    raise NotImplementedError
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(cfg.device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cfg.local_rank, ],
                                                output_device=cfg.local_rank)
  else:
    model = nn.DataParallel(model).to(cfg.device)

  if os.path.isfile(cfg.pretrain_dir):
    model = load_model(model, cfg.pretrain_dir)

  optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)

  def train(epoch):
    print('\n Epoch: %d' % epoch)
    model.train()
    tic = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
      batch['event'] = batch['event'].to(device=cfg.device, non_blocking=True)

      outputs = model(batch['event'])
      outs, hiddens = outputs
      hmap, regs, w_h_ = zip(*outs)
      # [print(batch[key].size()) for key in ['event','inds_t', 'hmap_t', 'regs_t', 'w_h_t']]
      # print([r.size() for r in regs])
      # print([r.size() for r in hmap])
      # print([r.size() for r in w_h_])
      batch['inds_t'] = batch['inds_t'].to(device=cfg.device, non_blocking=True)
      regs = [_tranpose_and_gather_feature_t(r, batch['inds_t']) for r in regs]
      w_h_ = [_tranpose_and_gather_feature_t(r, batch['inds_t']) for r in w_h_]

      hmap_loss = _neg_loss_t(hmap, batch['hmap_t'].to(device=cfg.device, non_blocking=True))
      reg_loss = _reg_loss_t(regs, batch['regs_t'].to(device=cfg.device, non_blocking=True), batch['ind_masks_t'].to(device=cfg.device, non_blocking=True))
      w_h_loss = _reg_loss_t(w_h_, batch['w_h_t'].to(device=cfg.device, non_blocking=True), batch['ind_masks_t'].to(device=cfg.device, non_blocking=True))
      loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        duration = time.perf_counter() - tic
        tic = time.perf_counter()
        print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
              ' hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f total_loss = %.5f' %
              (hmap_loss.item(), reg_loss.item(), w_h_loss.item(), loss.item()) +
              ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration))

        step = len(train_loader) * epoch + batch_idx
        summary_writer.add_scalar('hmap_loss', hmap_loss.item(), step)
        summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
        summary_writer.add_scalar('w_h_loss', w_h_loss.item(), step)
        summary_writer.add_scalar('total_loss', loss.item(), step)
      return
    return

  def val_map(epoch):
    print('\n Val@Epoch: %d' % epoch)
    model.eval()
    torch.cuda.empty_cache()
    max_per_image = 100

    results = {}
    with torch.no_grad():
      detections = {}
      for inputs in tqdm.tqdm(val_loader):
        assert len(inputs["file_path"]) == 1
        event_file_path = inputs["file_path"][0]
        if event_file_path in list(detections.keys()):
          continue
        outputs = model(inputs['event'].to(cfg.device))
        outs, _ = outputs
        hmap_t, regs_t, w_h_t = zip(*outs)
        for frame_idx, (hmap, regs, w_h_) in enumerate(zip(hmap_t[-1],regs_t[-1],w_h_t[-1])):
          dets = ctdet_decode(hmap, regs, w_h_)
          dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]
          dets[:, :2] = transform_preds(dets[:, 0:2],
                                        inputs['center'][0],
                                        inputs['scale'][0],
                                        (inputs['fmap_w'][0], inputs['fmap_h'][0]))
          dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                          inputs['center'][0],
                                          inputs['scale'][0],
                                          (inputs['fmap_w'][0], inputs['fmap_h'][0]))
          scores = dets[:, 4]
          dets = dets[dets[:, 4].argsort()[::-1]]
          scores = dets[:, 4]
          if len(scores) > max_per_image:
            dets = dets[:max_per_image]
          try:  
            detections[event_file_path].append(dets)
          except KeyError:
            assert inputs["info"][frame_idx]["first_segment"]
            detections[event_file_path] = [dets]

      proph_bboxes = {}
      for key in detections.keys():
        # n_dets = sum([len(d) for d in detections[key]])
        entries = []
        for dets_idx, dets in enumerate(detections[key]):
          t = dets_idx * val_dataset.delta_t
          assert t == int(t)
          t = int(t)
          X = dets[:,0]
          Y = dets[:,1]
          W = dets[:,0] + dets[:,2]
          H = dets[:,1] + dets[:,3]
          Class_confidence = dets[:,4]
          Class_id = dets[:,5]
          track_id = 0
          for x, y, w, h, class_id, class_confidence in zip(X, Y, W, H, Class_id, Class_confidence):
            entries.append((t, x, y, w, h, class_id, class_confidence, track_id))
        proph_bboxes[key] = np.array(entries, dtype=PROPH_STRUCTURED_ARRAY)
    eval_results = val_dataset.run_eval(proph_bboxes)
    
    print(eval_results)
    summary_writer.add_scalar('val_mAP/mAP', eval_results[0], epoch)
  
  print('Starting training...')
  for epoch in range(1, cfg.num_epochs + 1):
    # train_sampler.set_epoch(epoch)
    train(epoch)
    if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
      val_map(epoch)
    print(saver.save(model.module.state_dict(), f'checkpoint{epoch}'))
    lr_scheduler.step(epoch)  # move to here after pytorch1.1.0

  summary_writer.close()


if __name__ == '__main__':
  with DisablePrint(local_rank=cfg.local_rank):
    main()
