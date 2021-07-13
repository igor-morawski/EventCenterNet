import unittest
import torch
import copy
import os.path as op
import random

from utils.dataloader import CSVDataset, collater


H=240
W=304
TEST_ANN_CSV = op.join("tests", "train_a.csv")
TEST_CLS_CSV = op.join("tests", "classes.csv")
# your own data_root
BINS = 6
DELTA_T = 50000
FRAMES_PER_BATCH = 5
DATA_ROOT = "/tmp2/igor/EV/Dataset/Automotive/"
INIT_KWARGS = {"train_file": TEST_ANN_CSV, "class_list": TEST_CLS_CSV,
               "batch_size": 2, "data_root": DATA_ROOT,
               "trim_to_shortest": True, "delta_t": DELTA_T, "frames_per_batch": FRAMES_PER_BATCH,
               "bins": BINS}

# VALS TO TEST
SHORTEST_TOTAL_TIME = 59999982
SEGMENT_N = SHORTEST_TOTAL_TIME // DELTA_T // FRAMES_PER_BATCH
VAL_FILEPATHS = ['train_a/17-04-06_13-51-53_854500000_914500000_td.dat',
                 'train_a/17-04-14_15-49-57_1281500000_1341500000_td.dat']


class TestAttritubtes(unittest.TestCase):
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
        dataset = CSVDataset(**INIT_KWARGS)
        # length
        self.assertEqual(len(dataset), SEGMENT_N * len(VAL_FILEPATHS))
        # XXX

    def test_batch_size_2(self):
        dataset = CSVDataset(**INIT_KWARGS)
        # length
        self.assertEqual(len(dataset), SEGMENT_N * len(VAL_FILEPATHS))
        #b{batch_idx}{segment_idx}
        b00 = dataset[0]
        b01 = dataset[1]
        b10 = dataset[2]
        b11 = dataset[3]
        b0_1 = dataset[len(dataset)-2]
        b1_1 = dataset[len(dataset)-1]
        bs = [b00, b01, b10, b11, b0_1, b1_1]
        firsts = [True, True, False, False, False, False]
        lasts = [False, False, False, False, True, True]
        for idx, (b, first, last) in enumerate(zip(bs, firsts, lasts)):
            annot = b['annot']
            self.assertEqual(len(annot),FRAMES_PER_BATCH, msg=idx)
            event = b['event']
            self.assertEqual(event.size(),torch.Size([FRAMES_PER_BATCH, BINS*2, H, W]), msg=idx)
            info = b['info']
            self.assertEqual(info['first_segment'],first, msg=idx)
            self.assertEqual(info['last_segment'],last, msg=idx)
            


    def test_batch_size_3(self):
        dataset = CSVDataset(**INIT_KWARGS)
        # length
        self.assertEqual(len(dataset), SEGMENT_N * len(VAL_FILEPATHS))
