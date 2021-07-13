from types import FrameType
import unittest
import torch
import copy
import os.path as op
import random
import collections

from torch.utils import data

from utils.dataloader import CSVDataset, collater
from torch.utils.data import DataLoader


H=240
W=304
TEST_ANN_CSV = op.join("tests", "train_a.csv")
TEST_CLS_CSV = op.join("tests", "classes.csv")
# your own data_root
BINS = 6
DELTA_T = 50000
FRAMES_PER_BATCH = 5
BATCH_SIZE = 2
DATA_ROOT = "/tmp2/igor/EV/Dataset/Automotive/"
INIT_KWARGS = {"csv_file": TEST_ANN_CSV, "class_list": TEST_CLS_CSV,
               "batch_size": BATCH_SIZE, "data_root": DATA_ROOT,
               "trim_to_shortest": True, "delta_t": DELTA_T, "frames_per_batch": FRAMES_PER_BATCH,
               "bins": BINS}

# VALS TO TEST
SHORTEST_TOTAL_TIME = 59999982
SEGMENT_N = SHORTEST_TOTAL_TIME // DELTA_T // FRAMES_PER_BATCH
VAL_FILEPATHS = ['train_a/17-04-06_13-51-53_854500000_914500000_td.dat',
                 'train_a/17-04-14_15-49-57_1281500000_1341500000_td.dat']


class TestCSVDataset(unittest.TestCase):
    def test_init(self):
        dataset = CSVDataset(**INIT_KWARGS)
        self.assertEqual(dataset.shortest_total_time, SHORTEST_TOTAL_TIME)
        self.assertEqual(dataset.segment_n, SEGMENT_N)
        self.assertEqual(dataset.event_names, [op.join(
            DATA_ROOT, val_filepath) for val_filepath in VAL_FILEPATHS])

        no_trimming = copy.deepcopy(INIT_KWARGS)
        no_trimming["trim_to_shortest"] = False
        with self.assertRaises(NotImplementedError):
            dataset = CSVDataset(**no_trimming)

    def test_batch_size_1(self):
        batch_size = 1
        init_1 = copy.deepcopy(INIT_KWARGS)
        init_1['batch_size'] = batch_size
        frames_per_batch = init_1['frames_per_batch']
        delta_t = init_1['delta_t']
        dataset = CSVDataset(**init_1)
        # length
        self.assertEqual(len(dataset), SEGMENT_N * len(VAL_FILEPATHS))
        #b{batch_idx}{segment_idx}
        b00 = dataset[0]
        b01 = dataset[1]
        b10 = dataset[SHORTEST_TOTAL_TIME//frames_per_batch//delta_t]
        b11 = dataset[SHORTEST_TOTAL_TIME//frames_per_batch//delta_t+1]
        b0_1 = dataset[SHORTEST_TOTAL_TIME//frames_per_batch//delta_t-1]
        b1_1 = dataset[len(dataset)-1]
        bs = [b00, b01, b10, b11, b0_1, b1_1]
        firsts = [True, False, True, False, False, False]
        lasts = [False, False, False, False, True, True]
        offsets = [0, 
            frames_per_batch*1*delta_t,
            0, 
            frames_per_batch*1*delta_t, 
            (SHORTEST_TOTAL_TIME//delta_t//frames_per_batch)*frames_per_batch*delta_t-frames_per_batch*delta_t, 
            (SHORTEST_TOTAL_TIME//delta_t//frames_per_batch)*frames_per_batch*delta_t-frames_per_batch*delta_t]
        file_paths = [op.join(
            DATA_ROOT, val_filepath) for val_filepath in VAL_FILEPATHS] * 3
        file_paths[1] = file_paths[-2]
        file_paths[2] = file_paths[-1]
        gens = [bs, firsts, lasts, file_paths, offsets]
        len_gens = [len(g) for g in gens]
        assert len(set(len_gens)) == 1

        for idx, (b, first, last, file_path, offset) in enumerate(zip(*gens)):
            annot = b['annot']
            self.assertEqual(len(annot),frames_per_batch, msg=idx)
            event = b['event']
            self.assertEqual(event.size(),torch.Size([frames_per_batch, BINS*2, H, W]), msg=idx)
            info = b['info']
            self.assertEqual(info['first_segment'],first, msg=idx)
            self.assertEqual(info['last_segment'],last, msg=idx)
            self.assertEqual(b['file_path'],file_path, msg=idx)
            self.assertEqual(info['time_info']['delta_t'],delta_t, msg=idx)
            for frame_idx in range(frames_per_batch):
                self.assertEqual(info['time_info']['t0'][frame_idx],float(frame_idx*delta_t+offset), msg=f"{idx},{frame_idx}")
        
        dataloader = DataLoader(dataset, collate_fn=collater, batch_size=dataset.batch_size, shuffle=False)
        batches = []
        for iter_num, data in enumerate(dataloader):
            self.assertEqual(data['event'].size(),torch.Size([batch_size, frames_per_batch, 2*BINS, H, W]))
            batches.append(data)
            if iter_num >= 1:
                break
        self.assertEqual(len(batches),2)
        batch0, batch1 = batches
        batches_gen = [batch0, batch1]
        sample_idx_gen = [0, 0]

        for idx, (batch, sample_idx, b, first, last, file_path, offset) in enumerate(zip(batches_gen, sample_idx_gen, *gens)):
            annot = b['annot']
            self.assertEqual(len(annot),len(batch['annot'][sample_idx]), msg=idx)
            event = b['event']
            self.assertEqual(event.size(),batch['event'][sample_idx].size(), msg=idx)
            self.assertTrue(torch.equal(event,batch['event'][sample_idx]), msg=idx)
            info = b['info']
            self.assertEqual(info['first_segment'],batch['info'][sample_idx]['first_segment'], msg=idx)
            self.assertEqual(info['last_segment'],batch['info'][sample_idx]['last_segment'], msg=idx)
            self.assertEqual(b['file_path'],batch['file_path'][sample_idx], msg=idx)
            self.assertEqual(info['time_info']['delta_t'],batch['info'][sample_idx]['time_info']['delta_t'], msg=idx)
            for frame_idx in range(frames_per_batch):
                self.assertEqual(info['time_info']['t0'][frame_idx],batch['info'][sample_idx]['time_info']['t0'][frame_idx], msg=f"{idx},{frame_idx}")
            
    def test_batch_size_2(self):
        batch_size = 2 
        init_2 = copy.deepcopy(INIT_KWARGS)
        init_2['batch_size'] = batch_size
        frames_per_batch = init_2['frames_per_batch']
        delta_t = init_2['delta_t']
        dataset = CSVDataset(**init_2)
        # length
        self.assertEqual(len(dataset), SEGMENT_N * len(VAL_FILEPATHS))
        #b{batch_idx}{segment_idx}
        b00 = dataset[0]
        b10 = dataset[1]
        b01 = dataset[2]
        b11 = dataset[3]
        b0_1 = dataset[len(dataset)-2]
        b1_1 = dataset[len(dataset)-1]
        bs = [b00, b10, b01, b11, b0_1, b1_1]
        firsts = [True, True, False, False, False, False]
        lasts = [False, False, False, False, True, True]
        offsets = [0, 0, 
            frames_per_batch*1*delta_t, frames_per_batch*1*delta_t, 
            (SHORTEST_TOTAL_TIME//delta_t//frames_per_batch)*frames_per_batch*delta_t-frames_per_batch*delta_t, 
            (SHORTEST_TOTAL_TIME//delta_t//frames_per_batch)*frames_per_batch*delta_t-frames_per_batch*delta_t]
        file_paths = [op.join(
            DATA_ROOT, val_filepath) for val_filepath in VAL_FILEPATHS] * 3
        gens = [bs, firsts, lasts, file_paths, offsets]
        len_gens = [len(g) for g in gens]
        assert len(set(len_gens)) == 1
        for idx, (b, first, last, file_path, offset) in enumerate(zip(*gens)):
            annot = b['annot']
            self.assertEqual(len(annot),frames_per_batch, msg=idx)
            event = b['event']
            self.assertEqual(event.size(),torch.Size([frames_per_batch, BINS*2, H, W]), msg=idx)
            info = b['info']
            self.assertEqual(info['first_segment'],first, msg=idx)
            self.assertEqual(info['last_segment'],last, msg=idx)
            self.assertEqual(b['file_path'],file_path, msg=idx)
            self.assertEqual(info['time_info']['delta_t'],delta_t, msg=idx)
            for frame_idx in range(frames_per_batch):
                self.assertEqual(info['time_info']['t0'][frame_idx],float(frame_idx*delta_t+offset), msg=f"{idx},{frame_idx}")
            # no collating
        dataloader = DataLoader(dataset, collate_fn=collater, batch_size=dataset.batch_size, shuffle=False)
        batches = []
        for iter_num, data in enumerate(dataloader):
            self.assertEqual(data['event'].size(),torch.Size([batch_size, frames_per_batch, 2*BINS, H, W]))
            batches.append(data)
            if iter_num >= 1:
                break
        self.assertEqual(len(batches),2)
        batch0, batch1 = batches
        batches_gen = [batch0, batch0, batch1, batch1]
        sample_idx_gen = [0, 1, 0, 1]

        for idx, (batch, sample_idx, b, first, last, file_path, offset) in enumerate(zip(batches_gen, sample_idx_gen, *gens)):
            annot = b['annot']
            self.assertEqual(len(annot),len(batch['annot'][sample_idx]), msg=idx)
            event = b['event']
            self.assertEqual(event.size(),batch['event'][sample_idx].size(), msg=idx)
            self.assertTrue(torch.equal(event,batch['event'][sample_idx]), msg=idx)
            info = b['info']
            self.assertEqual(info['first_segment'],batch['info'][sample_idx]['first_segment'], msg=idx)
            self.assertEqual(info['last_segment'],batch['info'][sample_idx]['last_segment'], msg=idx)
            self.assertEqual(b['file_path'],batch['file_path'][sample_idx], msg=idx)
            self.assertEqual(info['time_info']['delta_t'],batch['info'][sample_idx]['time_info']['delta_t'], msg=idx)
            for frame_idx in range(frames_per_batch):
                self.assertEqual(info['time_info']['t0'][frame_idx],batch['info'][sample_idx]['time_info']['t0'][frame_idx], msg=f"{idx},{frame_idx}")
            
