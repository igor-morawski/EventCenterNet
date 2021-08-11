import sys
import csv
import os.path as op

import math
from utils.losses import _reg_loss
import numpy as np
import torch

from torch.utils.data import Dataset
from utils.image import draw_umich_gaussian, gaussian_radius, get_affine_transform, affine_transform
try:
    from . import events as ev
except ImportError:
    import events as ev

import cv2 # XXX debug

module_imports = ['from {}prophesee.src.io.psee_loader import PSEELoader as PSEELoader',
                  'from {}prophesee.src.metrics.coco_eval import evaluate_detection as evaluate_detection',
                  'from {}prophesee.src.io.box_filtering import filter_boxes as filter_boxes',
                  'from {}prophesee.src.io.box_loading import reformat_boxes as reformat_boxes']
for module_import in module_imports:
    try:
        exec(module_import.format("."))
    except ImportError:
        exec(module_import.format(""))


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, csv_file, class_list, batch_size, data_root, hw, transform=None, trim_to_shortest=True, delta_t=50000, frames_per_batch=5, bins=5, pad=True, val=False, camera="GEN4"):
        """
        Args:
            csv_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
            batch_size (int): nubmer of samples in each batch
            # TODO
        """
        assert batch_size > 0
        assert isinstance(batch_size, int)
        self.csv_file = csv_file
        assert op.exists(data_root)
        self.data_root = data_root
        self.class_list = class_list
        self.batch_size = batch_size
        assert len(hw) == 2
        self.hw = hw
        self.h, self.w = self.hw
        self.transform = transform
        assert bins > 0
        self.bins = bins
        if self.transform:
            raise NotImplementedError
        assert delta_t > 0
        assert isinstance(delta_t, int)
        self.delta_t = delta_t
        assert frames_per_batch > 0
        assert isinstance(frames_per_batch, int)
        self.frames_per_batch = frames_per_batch
        self.pad = pad
        self.val = val
        self.trim_to_shortest = trim_to_shortest
        self.camera = camera

        if self.val:
            if not self.batch_size == 1:
                raise ValueError("For validation batch size must be 1")
            if self.trim_to_shortest:
                raise ValueError("'trim_to_shortest' must be False")

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(
                    csv.reader(file, delimiter=","))
        except ValueError as e:
            raise(ValueError("invalid CSV class file: {}: {}".format(self.class_list, e)))
        self.num_classes = len(self.classes.keys())

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with events_path, video_path, total_time
        try:
            with self._open_for_csv(self.csv_file) as file:
                self.events_data = self._read_annotations(
                    csv.reader(file, delimiter=","))
        except ValueError as e:
            raise(ValueError(
                "invalid CSV annotations file: {}: {}".format(self.csv_file, e)))
        self.event_names = list(self.events_data.keys())
        self.total_times = [self.events_data[n]["total_time"]
                            for n in self.event_names]
        if self.trim_to_shortest:
            self.shortest_total_time = min(
                [data["total_time"] for data in self.events_data.values()])
            # total frames in each sample
            self.sample_frames = self.shortest_total_time//self.delta_t
            # number of segments in each sample
            self.segment_n = self.sample_frames // self.frames_per_batch
        if not self.val and not self.trim_to_shortest:
            raise NotImplementedError
        if self.val and not self.trim_to_shortest:
            self.sample_frames = 0
            self.segment_n = 0
            # init self._val_len
            self._val_len = sum([data["total_time"] // self.delta_t //
                                self.frames_per_batch for data in self.events_data.values()])
            self._batch2file_idx = []
            self._batch2segment_idx = []
            for file_idx in range(len(self.events_data.keys())):
                total_time = self.total_times[file_idx]
                for segment_idx in range(total_time // self.delta_t // self.frames_per_batch):
                    self._batch2file_idx.append(file_idx)
                    self._batch2segment_idx.append(segment_idx)
            assert len(self._batch2file_idx) == self._val_len
            assert len(self._batch2segment_idx) == self._val_len

        self.down_ratio = 4
        self.event_size = {"h": self.h, "w": self.w}
        self.max_objs = 128
        self.padding = 128
        self.pad_h = self.padding - (self.h % self.padding)
        self.pad_w = self.padding - (self.w % self.padding)
        self.fmap_size = {"h": self.h // self.down_ratio,
                            "w": self.w // self.down_ratio}
        self.padded_fmap_size = {"h": (self.h + self.pad_h) // self.down_ratio,
                            "w": (self.w + self.pad_w) // self.down_ratio}
        self.gaussian_iou = 0.7

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode "rb",
        for python3 this means "r" with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, "rb")
        else:
            return open(path, "r", newline="")

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError(
                    "line {}: format should be \"class_name,class_id\"".format(line)))
            class_id = self._parse(
                class_id, int, "line {}: malformed class ID: {{}}".format(line))

            if class_name in result:
                raise ValueError(
                    "line {}: duplicate class name: \"{}\"".format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        if not self.val:
            file_n = len(self.event_names)
            return (file_n * self.segment_n)
        else:
            return self._val_len

    def __getitem__(self, idx):
        if not self.val:
            file_idx = (idx // (self.batch_size * self.segment_n)) * \
                self.batch_size + (idx % self.batch_size)
            segment_idx = idx % (
                self.batch_size * self.segment_n) // self.batch_size
        else:
            file_idx = self._batch2file_idx[idx]
            segment_idx = self._batch2segment_idx[idx]

        event, event_time_info = self.load_events(
            file_idx, segment_idx, return_time_info=True)
        annot, annot_time_info = self.load_annotations(
            file_idx, segment_idx, return_time_info=True)
        assert event_time_info == annot_time_info
        first_segment = True if not segment_idx else False
        last_segment = True if segment_idx == self.segment_n - 1 else False

        height, width = event.size()[-2:]
        assert self.h == height
        assert self.w == width
        center = np.array([width / 2., height / 2.], dtype=np.float32)
        scale = max(height, width) * 1.0
        trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])
        if self.pad:
            event = torch.nn.functional.pad(
                event, (0, self.pad_w, 0, self.pad_h)) 

        bboxes_t, labels_t = zip(*[(a[:, 0:4], a[:, 4]) for a in annot])
        hmap_t = []
        w_h_t = []
        regs_t = []
        inds_t = []
        ind_masks_t = []
        lengths = self.fmap_size
        if self.pad:
            lengths = self.padded_fmap_size
        for t, (bboxes, labels) in enumerate(zip(bboxes_t, labels_t)):
            hmap =  np.zeros(
                (self.num_classes, self.padded_fmap_size["h"], self.padded_fmap_size["w"]), dtype=np.float32)  # heatmap
            # width and height
            w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)
            regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
            inds = np.zeros((self.max_objs,), dtype=np.int64)
            ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
            # filled=False # XXX
            for k, (bbox, label) in enumerate(zip(bboxes, labels)):
                # break # XXX DEBUG
                # filled = True
                bbox[:2] = affine_transform(bbox[:2], trans_fmap)
                bbox[2:] = affine_transform(bbox[2:], trans_fmap)
                bbox[[0, 2]] = np.clip(
                    bbox[[0, 2]], 0, self.fmap_size["w"] - 1)
                bbox[[1, 3]] = np.clip(
                    bbox[[1, 3]], 0, self.fmap_size["h"] - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 or w > 0:
                    obj_c = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    obj_c_int = obj_c.astype(np.int32)
                    assert int(label) == label  # sanity
                    radius = max(0, int(gaussian_radius(
                        (math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                    draw_umich_gaussian(hmap[int(label)], obj_c_int, radius)
                    w_h_[k] = 1. * w, 1. * h
                    regs[k] = obj_c - obj_c_int  # discretization error
                    inds[k] = obj_c_int[1] * lengths["w"] + obj_c_int[0] 
                    ind_masks[k] = 1
            # if filled: # XXX
                # print(bbox)
                # print(np.where(hmap > 0))
                # print(w_h_[0])
                # print(regs[0])
                # print("po ")
                # inv_tr = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']], inv=True)
                # print(affine_transform(bbox, inv_tr))
                # exit()
                # pass
            hmap_t.append(hmap)
            w_h_t.append(w_h_)
            regs_t.append(regs)
            inds_t.append(inds)
            ind_masks_t.append(ind_masks)
        info = {"first_segment": first_segment, "last_segment": last_segment,
                "time_info": event_time_info, "debug":[any([len(b) for b in bboxes]), bboxes]}  # XXX
        file_path = self.event_names[file_idx]
        sample = {"event": event, "annot": annot, "info": info, "file_path": file_path,
                  "hmap_t": hmap_t, "w_h_t": w_h_t, "regs_t": regs_t, "inds_t": inds_t, "ind_masks_t": ind_masks_t,
                  "center": center, "scale": scale, "fmap_h": self.fmap_size["h"], "fmap_w": self.fmap_size["w"]}
        if self.transform:
            raise NotImplementedError
            sample = self.transform(sample)
        return sample

    def load_events(self, file_idx, segment_idx, return_time_info=False):
        video = PSEELoader(self.event_names[file_idx])
        events = []
        if return_time_info:
            time_info = {"t0": [], "delta_t": self.delta_t}
        for frame_idx in range(self.frames_per_batch):
            t0 = (segment_idx * self.frames_per_batch *
                  self.delta_t) + (frame_idx*self.delta_t)
            if return_time_info:
                time_info["t0"].append(t0)
            event = ev.video_segment_to_voxel(
                video, t0, self.delta_t, bins=self.bins)
            events.append(event)
        segment = torch.stack(events)  # F, C, H, W
        if return_time_info:
            return segment, time_info
        return segment

    def load_annotations(self, file_idx, segment_idx, return_time_info=False):  # here
        anno = PSEELoader(
            self.events_data[self.event_names[file_idx]]["anno_file"])
        annotations_for_frames = []
        if return_time_info:
            time_info = {"t0": [], "delta_t": self.delta_t}
        for frame_idx in range(self.frames_per_batch):
            annotations = np.zeros((0, 5))
            t0 = (segment_idx * self.frames_per_batch *
                  self.delta_t) + (frame_idx*self.delta_t)
            if return_time_info:
                time_info["t0"].append(t0)
            for bbox_info in ev.video_segment_anno_to_bboxes(anno, t0, self.delta_t):
                # break # XXX DEBUG 
                x, y, w, h, class_id = bbox_info["x"], bbox_info["y"], bbox_info["w"], bbox_info["h"], bbox_info["class_id"]
                annotation = np.zeros((1, 5))
                x1 = x
                x2 = x+w
                y1 = y
                y2 = y+h
                annotation[0, 0] = x1
                annotation[0, 1] = y1
                annotation[0, 2] = x2
                annotation[0, 3] = y2
                annotation[0, 4] = class_id
                # print(annotation)
                annotations = np.append(annotations, annotation, axis=0)
            # print("sum:",annotations)
            annotations_for_frames.append(annotations)
        # print("frames:",annotations_for_frames)
        if return_time_info:
            return annotations_for_frames, time_info
        return annotations_for_frames

    def _read_annotations(self, csv_reader, times_only=False):
        """
        Returns a dictionary {file_path : {"total_time" : total_time, "anno_file" : anno_file } }"""
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                events_file, anno_file, total_time = row[:3]
            except ValueError:
                raise_from(ValueError(
                    "line {}: format should be \"events_file,anno_file,total_time\" or \"events_file,,\"".format(line)), None)

            events_file = op.join(self.data_root, events_file)
            anno_file = op.join(self.data_root, anno_file)
            assert events_file not in result

            # If a row contains only an image path, it"s an image without annotations.
            if anno_file == "":
                continue
            assert op.exists(events_file)
            assert op.exists(anno_file)

            total_time = self._parse(
                total_time, int, "line {}: malformed total_time: {{}}".format(line))

            result[events_file] = {
                "total_time": total_time, "anno_file": anno_file}
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def run_eval(self, results, save_dir=None):
        if not self.val:
            raise ValueError(
                f"'val' set to {self.val} when init. dataset object")
        if save_dir:
            raise NotImplementedError

        dt_file_paths = sorted(list(results.keys()))
        gt_file_paths = sorted(list(self.events_data.keys()))
        assert len(dt_file_paths) == len(gt_file_paths)
        assert dt_file_paths == gt_file_paths
        print("There are {} GT bboxes and {} PRED bboxes".format(
            len(gt_file_paths), len(dt_file_paths)))
        result_boxes_list = [results[p] for p in gt_file_paths]
        gt_boxes_list = [np.load(self.events_data[p]["anno_file"])
                         for p in gt_file_paths]

        result_boxes_list = [reformat_boxes(p) for p in result_boxes_list]
        gt_boxes_list = [reformat_boxes(p) for p in gt_boxes_list]

        min_box_diag = 60 if self.camera == 'GEN4' else 30
        min_box_side = 20 if self.camera == 'GEN1' else 10
        filter_boxes_fn = lambda x:filter_boxes(x, int(1e5), min_box_diag, min_box_side)

        gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
        result_boxes_list = map(filter_boxes_fn, result_boxes_list)
        
        #XXX car pedestrian
        return evaluate_detection(gt_boxes_list, result_boxes_list, classes=("car", "pedestrian"), height=self.h, width=self.w,
                                  time_tol=50000)


def collater(data):
    result = {}
    result["event"] = torch.stack([s["event"] for s in data])
    keys = ["annot", "info", "file_path",
            "center", "scale", "fmap_h", "fmap_w"]
    transpose_keys = ["hmap_t", "w_h_t", "regs_t", "inds_t", "ind_masks_t", ]
    for key in keys:
        result[key] = [s[key] for s in data]
    for key in transpose_keys:
        tmp = np.array([s[key] for s in data])
        remainder = list(range(0, len(tmp.shape)))[2:]
        tmp = tmp.transpose([1, 0]+remainder)
        tmp = torch.Tensor(tmp).cpu().contiguous()
        # print(tmp.size())
        if "ind" in key:
            tmp = tmp.long()
        result[key] = tmp
    return result
