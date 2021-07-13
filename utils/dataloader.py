import sys
import csv
import numpy as np
import os.path as op
import torch
from torch.utils.data import Dataset
try:
    from . import events as ev
except ImportError: 
    import events

class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, batch_size, data_root, transform=None, trim_to_shortest=True, delta_t=50000, frames_per_batch=5, bins=5):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
            batch_size (int): nubmer of samples in each batch
        """
        assert batch_size > 0
        assert isinstance(batch_size, int)
        self.train_file = train_file
        assert op.exists(data_root)
        self.data_root = data_root
        self.class_list = class_list
        self.batch_size = batch_size
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

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
        
        self.trim_to_shortest = trim_to_shortest
        
        # csv with events_path, video_path, total_time
        try:
            with self._open_for_csv(self.train_file) as file:
                self.events_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.event_names = list(self.events_data.keys())
        if self.trim_to_shortest:          
            self.shortest_total_time = min([data['total_time'] for data in self.events_data.values()])
        if not self.trim_to_shortest:
            raise NotImplementedError
        self.sample_frames = self.shortest_total_time//self.delta_t # total frames in each sample
        self.segment_n = self.sample_frames // self.frames_per_batch # number of segments in each sample
        


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
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        file_n = len(self.event_names)
        return (file_n * self.segment_n)

    def __getitem__(self, idx):
        file_idx = (idx // (self.batch_size * self.segment_n)) * self.batch_size + (idx % self.batch_size)
        segment_idx = idx % (self.batch_size * self.segment_n) // self.batch_size
        event = self.load_events(file_idx, segment_idx)
        annot = self.load_annotations(file_idx, segment_idx)
        first_segment = True if not segment_idx else False
        last_segment = True if segment_idx == self.segment_n - 1 else False
        info = { "first_segment" : first_segment, "last_segment" : last_segment } # XXX
        file_path = self.event_names[file_idx]
        sample = {'event': event, 'annot': annot, 'info' : info, 'file_path' : file_path}
        if self.transform:
            raise NotImplementedError
            sample = self.transform(sample)
        return sample

    def load_events(self, file_idx, segment_idx):
        video = ev.PSEELoader(self.event_names[file_idx])
        events = []
        for frame_idx in range(self.frames_per_batch):
            t0 = (segment_idx * self.frames_per_batch * self.delta_t) + (frame_idx*self.delta_t)
            event = ev.video_segment_to_voxel(video, t0, self.delta_t, bins=self.bins)
            events.append(event)
        segment = torch.stack(events) # F, C, H, W
        return segment

    def load_annotations(self, file_idx, segment_idx):  # here
        anno = ev.PSEELoader(self.events_data[self.event_names[file_idx]]['anno_file'])
        annotations_for_frames = []
        for frame_idx in range(self.frames_per_batch):
            annotations = np.zeros((0, 5))
            t0 = (segment_idx * self.frames_per_batch * self.delta_t) + (frame_idx*self.delta_t)
            for bbox_info in ev.video_segment_anno_to_bboxes(anno, t0, self.delta_t):
                x, y, w, h, class_id  = bbox_info['x'], bbox_info['y'], bbox_info['w'], bbox_info['h'], bbox_info['class_id']
                annotation  = np.zeros((1, 5))
                # XXX
                x1 = x
                x2 = x+w
                y1 = y
                y2 = y+h
                annotation[0, 0] = x1
                annotation[0, 1] = y1
                annotation[0, 2] = x2
                annotation[0, 3] = y2
                annotation[0, 4]  = class_id
                # print(annotation)
                annotations = np.append(annotations, annotation, axis=0)    
            # print("sum:",annotations)       
            annotations_for_frames.append(annotations)
        # print("frames:",annotations_for_frames)       
        return annotations_for_frames

    def _read_annotations(self, csv_reader, classes):
        '''
        Returns a dictionary {file_path : {'total_time' : total_time, 'anno_file' : anno_file } }'''
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                events_file, anno_file, total_time = row[:3]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'events_file,anno_file,total_time\' or \'events_file,,\''.format(line)), None)

            events_file = op.join(self.data_root, events_file)
            anno_file = op.join(self.data_root, anno_file)
            assert events_file not in result

            # If a row contains only an image path, it's an image without annotations.
            if anno_file == "":
                continue
            assert op.exists(events_file)
            assert op.exists(anno_file)

            total_time = self._parse(total_time, int, 'line {}: malformed total_time: {{}}'.format(line))

            result[events_file] = {'total_time':total_time,'anno_file':anno_file}
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

def collater(data):

    events = [s['event'] for s in data]
    annots = [s['annot'] for s in data]
    infos = [s['info'] for s in data]
    file_paths = [s['file_path'] for s in data]

    # XXX

    return {'event': events, 'annot': annots, 'info': infos, 'file_paths': file_path}

