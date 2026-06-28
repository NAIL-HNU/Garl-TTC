"""
TTC dataset implemented as PyTorch dataset object.
"""
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import torch.utils.data
from typing import List
import math
import random
import logging
import io
import tarfile

try:
    from colorama import Fore, Style
except ImportError:
    class _NoColor:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = RESET_ALL = ''
    Fore = Style = _NoColor()
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from garl_ttc.utils.io import read_txt, read_pkl, read_image
from garl_ttc.utils.events import extract_from_h5_by_timewindow
import traceback

from garl_ttc.datasets.event_representation import get_timevolume_roi_np


empty_logger = logging.getLogger("empty_logger")
empty_logger.setLevel(logging.CRITICAL + 1)
    
class TTCEstimationDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            cfgs, 
            split, 
            logger=None,
            seed=None,
            db_mode='window',  # 'sequence' or 'window' or 'target' or 'case'
            db_size=None,
            seq_ts=0,
            test_ttc_range=None,
            target_asset=None,
            target_id=None,
            hybridtus_start=None,
            hybridtus_end=None,
            case_ids=None
            ):
        super().__init__()

        """
        sync: Can be either 'front' (last event ts), or 'back' (first event ts). Whether the front of the window or
              the back of the window is synced with the images.

        Each sample of this dataset loads one image, events, and labels at a timestamp. The behavior is different for 
        sync='front' and sync='back', and these are visualized below.

        Legend: 
        . = events
        | = image
        L = label

        sync='front'
        -------> time
        .......|
               L

        sync='back'
        -------> time
        |.......
               L
        
        """
        if logger is None:
            self.logger = empty_logger
        else:
            self.logger = logger
        self.db_mode = db_mode
        assert db_mode in ['sequence', 'window', 'target', 'case'], f"Invalid db_mode: {db_mode}"
        
        dataset_cfgs = cfgs['dataset']
        self.root = Path(dataset_cfgs['root']).absolute()
        self.annotation_format = dataset_cfgs.get(split, {}).get(
            'annotation_format',
            dataset_cfgs.get('annotation_format', 'pkl'),
        )
        self.has_labels = True
        self._tar_cache = {}
        self.sync = dataset_cfgs['sync']
        self.window_interval = dataset_cfgs['window_interval']
        self.pixel_diff = dataset_cfgs['event_pixel_diff']
        self.input_feat_size = cfgs['model']['input_feat_size']
        self.datablob_dir = Path(dataset_cfgs.get('data_blob_dir', self.root / 'data_blobs')).absolute()
        self.data_mode = dataset_cfgs['mode']
        self.fy = dataset_cfgs['fy']
        normalize = transforms.Normalize(mean=dataset_cfgs['img_mean'], std=dataset_cfgs['img_std'])
        transform_list = [transforms.ToTensor(), normalize]
        self.pth_trans = transforms.Compose(transform_list) 
        
        assert Path(self.root).exists()
        assert split in ['train', 'test', 'val']
        assert self.sync in ['front', 'back']
        assert self.window_interval > 0
        
        self.split = split
        
        if self.split == 'train':
            self.gt_range = cfgs['training_settings']['gt_range']
            db_size = dataset_cfgs.get('db_sample_size', None)
        
        elif self.split == 'test':
            self.seq_ts = seq_ts
            self.gt_range = test_ttc_range
            db_size = db_size
            self.target_asset = target_asset
            self.target_id = target_id
            self.hybridtus_start = hybridtus_start
            self.hybridtus_end = hybridtus_end
            self.case_ids = case_ids
        else:
            raise NotImplementedError(f"Split [{split}] not implemented")
        
        print(Fore.YELLOW + f"Initializing [{split}] set, db_mode: [{db_mode}], db_size: [{db_size}], TTC range: {self.gt_range}, please wait... " + Style.RESET_ALL)
        self.logger.info(Fore.YELLOW + f"Initializing [{split}] set, db_mode: [{db_mode}], db_size: [{db_size}], TTC range: {self.gt_range}, please wait... " + Style.RESET_ALL)
        self.init_db(split, dataset_cfgs)
        
        print(Fore.GREEN + f"Initialization finished for [{split}] set. TTC range: {self.gt_range}" + Style.RESET_ALL)
        self.logger.info(Fore.GREEN + f"Initialization finished for [{split}] set. TTC range: {self.gt_range}" + Style.RESET_ALL)
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
        if db_size is not None and db_size < self.total_data:
            self.indices = torch.randperm(self.total_data)[:db_size]
        else:
            self.indices = torch.randperm(self.total_data)
        self.logger.info(f"Downsampled dataset size from [{self.total_data}] to [{len(self.indices)}]")
            

    def init_db(self, split, dataset_cfgs):
        """
        Prepare data (e.g., input-output pairs and metadata) that will be used 
        depending on the type of experiment.
        """
        if self.annotation_format == 'parquet':
            return self.init_db_from_parquet(split, dataset_cfgs)

        assets_id_path = dataset_cfgs[split]['asset_path']
        self.asset_list = read_txt(assets_id_path)
        if not len(self.asset_list) > 0:
            self.logger.error(f"No asset_id found in asset file: [{assets_id_path}]")
            raise RuntimeError
        
        track_dir = dataset_cfgs.get(split, {}).get('annotation_dir', dataset_cfgs.get('annotation_dir'))
        if track_dir is None:
            track_dir = Path(self.root) / 'anno' / dataset_cfgs['raw_data_tag'] / dataset_cfgs['anno_tag'] / 'filter_anno' / dataset_cfgs['fillter_tag']
        else:
            track_dir = Path(track_dir)
        if not Path(track_dir).exists():
            self.logger.error(f"Track directory not found: [{track_dir}]")
            raise RuntimeError
        else:
            self.logger.info(f"Loading tracks from: [{track_dir}]")
            
        trackpath_list = []
        for asset in self.asset_list:
            track_path = Path(track_dir) / f"{asset}.pkl"
            if not track_path.is_file():
                self.logger.error(f"Track file not found for asset {asset}: [{track_path}]")
                raise FileNotFoundError(track_path)
            trackpath_list.append(track_path)
    
        db_list = []
        for path in tqdm(trackpath_list):
            track_list = read_pkl(path=path)
            for track in track_list:
                
                if self.db_mode == 'window':
                    instance_list = self.get_instance_list_from_one_track(track=track)
                    
                elif self.db_mode == 'sequence':
                    assert self.gt_range is not None, f"Invalid gt_range: {self.gt_range}"
                    assert self.seq_ts > 0, f"Invalid seq_ts: {self.seq_ts}"
                    instance_list = self.get_test_continue_sequence_from_one_track(track=track)
                    
                elif self.db_mode == 'target':
                    assert self.target_asset is not None and self.target_id is not None, \
                        f"Invalid target asset: {self.target_asset} or target id: {self.target_id}"
                    instance_list = self.get_test_target_from_one_track(
                        track=track, 
                        target_asset=self.target_asset, 
                        target_id=self.target_id,
                        tus_start=self.hybridtus_start,
                        tus_end=self.hybridtus_end,
                        )
                elif self.db_mode == 'case':
                    assert self.case_ids is not None, f"Invalid case_ids: {self.case_ids}"
                    instance_list = self.get_case_list_from_one_track_debug(track=track,
                                                                            case_ids=self.case_ids)
                else:
                    raise ValueError(f"Invalid db_mode: {self.db_mode}")
                if instance_list is None:
                    continue
                else:
                    db_list.extend(instance_list)
        
        self.db = db_list
        self.total_data = len(self.db)
        
        return

    def init_db_from_parquet(self, split, dataset_cfgs):
        split_cfg = dataset_cfgs.get(split, {})
        data_path = split_cfg.get('data_parquet')
        if data_path is None:
            raise RuntimeError(f"Missing dataset.{split}.data_parquet for HF-style annotations")
        data_df = pd.read_parquet(data_path)
        assets_id_path = split_cfg.get('asset_path')
        if assets_id_path is not None and Path(assets_id_path).exists():
            asset_set = set(read_txt(assets_id_path))
            data_df = data_df[data_df['sequence_id'].isin(asset_set)]

        label_path = split_cfg.get('labels_parquet')
        if label_path is not None and Path(label_path).exists():
            label_df = pd.read_parquet(label_path)
            data_df = data_df.merge(label_df, on=['sequence_id', 'sample_token', 'track_id', 'public_track_id', 'timestamp_us'], how='left')
            self.has_labels = 'ttc' in data_df.columns and not data_df['ttc'].isna().any()
        else:
            self.has_labels = False

        if split == 'train' and not self.has_labels:
            raise RuntimeError('HF-style train split requires annotations/train.parquet labels.')

        self.asset_list = sorted(data_df['sequence_id'].unique().tolist())
        self.db = data_df.to_dict('records')
        self.total_data = len(self.db)
        self.logger.info(f"Loaded HF-style {split} rows from [{data_path}], rows={self.total_data}, labels={self.has_labels}")
        return

    def _read_hf_image(self, shard_rel_path, member_path):
        shard_path = Path(self.root) / str(shard_rel_path)
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
        tar = self._tar_cache.get(shard_path)
        if tar is None:
            tar = tarfile.open(shard_path, 'r')
            self._tar_cache[shard_path] = tar
        extracted = tar.extractfile(str(member_path))
        if extracted is None:
            raise FileNotFoundError(f"{member_path} in {shard_path}")
        data = np.frombuffer(extracted.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Image decode failed: {member_path} in {shard_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _parquet_nested_float_list(self, value):
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [self._parquet_nested_float_list(item) for item in value.tolist()]
            return value.astype(float).tolist()
        if isinstance(value, (list, tuple)):
            return [self._parquet_nested_float_list(item) for item in value]
        if value is None:
            return value
        return float(value)

    def _parquet_tracks(self, row):
        tracks = []
        boxes = row['boxes_xyxy']
        max_edge = 0
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            square_box = get_expand_box([xmin, ymin, xmax, ymax])
            max_edge = max(max_edge, square_box[3] - square_box[1])

        frame_ttc = row.get('frame_ttc')
        box3d_fcam = row.get('box3d_Fcam')
        for idx, (timestamp, box) in enumerate(zip(row['frame_timestamps_us'], boxes)):
            track = {
                'public_track_id': row['public_track_id'],
                'timestamp': int(timestamp),
                'box': [int(v) for v in box],
                'max_edge': max_edge,
                'seq_name': row.get('seq_name', ''),
            }
            if self.has_labels:
                if frame_ttc is not None and not isinstance(frame_ttc, float):
                    track['ttc'] = float(frame_ttc[idx])
                else:
                    track['ttc'] = float(row['ttc'])
                track['box3d_h'] = float(row['box3d_h'])
                if box3d_fcam is not None and not isinstance(box3d_fcam, float):
                    track['box3d_Fcam'] = self._parquet_nested_float_list(box3d_fcam[idx])
            tracks.append(track)
        return tracks

    def _parquet_images(self, row):
        return [
            self._read_hf_image(shard, member)
            for shard, member in zip(row['rgb_shard_paths'], row['rgb_member_paths'])
        ]

    def _parquet_events(self, row, image_shape):
        event_file_path = Path(self.root) / str(row['events_path'])
        if not event_file_path.exists():
            raise FileNotFoundError(event_file_path)
        height, width = image_shape[:2]
        t_min_us_list = [int(item[0]) for item in row['event_windows_us']]
        t_max_us_list = [int(item[1]) for item in row['event_windows_us']]
        event_list = extract_from_h5_by_timewindow(
            str(event_file_path),
            t_min_us_list,
            t_max_us_list,
            self.pixel_diff,
            [height, width],
        )
        for ev_buffer in event_list:
            if len(ev_buffer['x']) == 0:
                return None
        return event_list

    def _getitem_parquet(self, idx):
        row = self.db[self.indices[idx]]
        try:
            images = self._parquet_images(row)
            events = self._parquet_events(row, images[-1].shape)
            if events is None:
                raise ValueError(f"Empty events for sample_token={row['sample_token']}")
            rawdata = {
                'tracks': self._parquet_tracks(row),
                'images': images,
                'masks': [None for _ in images],
                'events': events,
            }
            return self.load_data(rawdata, require_labels=self.has_labels, sample_token=row['sample_token'])
        except Exception as e:
            self.logger.error(f"Skipping parquet row at index {idx}, due to error: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def get_instance_list_from_one_track(self, track):
        instance_list = []
        track_id = track.get('track_id')
        valid_ts_list = sorted(track.get('valid_ts_list'))
        track_anno = track.get('anno')
        _, box3d_h, _ = next(iter(track_anno.values())).get('box3d_ego')[3:6]
        window_size = self.window_interval + 2
        for window in sliding_window(valid_ts_list, window_size):
            instance = get_instance_from_time_window(window=window, 
                                                    track_anno=track_anno, 
                                                    track_id=track_id,
                                                    gt_range=self.gt_range,
                                                    box3d_h=box3d_h)
            if instance is None:
                continue
            instance_list.append(instance)
            
        return instance_list or None
        
    def get_test_continue_sequence_from_one_track(self, track):
        instance_list = []
        track_id = track.get('track_id')
        valid_ts_list = sorted(track.get('valid_ts_list'))
        track_anno = track.get('anno')
        _, box3d_h, _ = next(iter(track_anno.values())).get('box3d_ego')[3:6]
        
        # split the track's valid_ts_list into continuous windows
        start_i = 0
        cnt_seqs = []
        for index in range(1, len(valid_ts_list)):
            if valid_ts_list[index] - valid_ts_list[index-1] > 100000:
                cnt_seqs.append(valid_ts_list[start_i:index])
                start_i = index
        if len(cnt_seqs) == 0:
            cnt_seqs.append(valid_ts_list)
        
        for cnt_seq in cnt_seqs:
            if self.db_mode == 'sequence':
                if len(cnt_seq) < self.seq_ts*10 + 2:
                    continue

                instance = get_instance_from_continue_sequence(
                    cnt_sequence=cnt_seq,
                    win_ts=self.seq_ts,
                    track_anno=track_anno,
                    track_id=track_id,
                    ttc_range=self.gt_range,
                    box3d_h=box3d_h
                    )
            else:
                raise ValueError(f"Invalid test_mode {self.db_mode}")
            
            if instance is None:
                continue
            instance_list.extend(instance)
            
        return instance_list or None
        
    def get_test_target_from_one_track(self, track, target_asset, target_id, 
                                       tus_start=None, 
                                       tus_end=None):
        instance_list = []
        track_id = track.get('track_id')
        track_asset = track_id.split('_')[0]
        track_id = int(track_id.split('_')[1])
        if track_asset != target_asset or track_id != target_id:
            return None
        
        print(Fore.GREEN + f"Found target track: {track_id} in asset: {track_asset}" + Style.RESET_ALL)
        
        valid_ts_list = sorted(track.get('valid_ts_list'))
        track_anno = track.get('anno')
        _, box3d_h, _ = next(iter(track_anno.values())).get('box3d_ego')[3:6]
        
        # split the track's valid_ts_list into continuous windows
        start_i = 0
        cnt_seqs = []
        for index in range(1, len(valid_ts_list)):
            if valid_ts_list[index] - valid_ts_list[index-1] > 100000:
                cnt_seqs.append(valid_ts_list[start_i:index])
                start_i = index
        if len(cnt_seqs) == 0:
            cnt_seqs.append(valid_ts_list)
        
        for cnt_seq in cnt_seqs:
            window_annos = []
            for tus in cnt_seq:
                zytname = get_zytname_from_tus(int(tus))
                anno = track_anno.get(zytname)
                
                if tus_start is not None and anno['meta']['corr_exposure_start_timestamp_us'] < tus_start:
                    continue
                if tus_end is not None and anno['meta']['corr_exposure_start_timestamp_us'] > tus_end:
                    continue
                
                anno.update(track_id=track_id)
                anno.update(box3d_h=box3d_h)
                window_annos.append(anno)
            
            if not window_annos:
                continue
            instance_list.append(window_annos)
                
        return instance_list or None
        
    def get_case_list_from_one_track_debug(self, track, case_ids):
        instance_list = []
        track_id = track.get('track_id')
        valid_ts_list = sorted(track.get('valid_ts_list'))
        track_anno = track.get('anno')
        _, box3d_h, _ = next(iter(track_anno.values())).get('box3d_ego')[3:6]
        window_size = self.window_interval + 2
        for window in sliding_window(valid_ts_list, window_size):
            instance = get_instance_from_time_window(window=window, 
                                                    track_anno=track_anno, 
                                                    track_id=track_id,
                                                    gt_range=self.gt_range,
                                                    box3d_h=box3d_h)
            if instance is None:
                continue
            
            case_id_list = [f"{item['public_track_id']}_{item['timestamp']}" for item in instance]
            match_case_id = list(set(case_id_list) & set(case_ids))
            if len(match_case_id) > 0:
                instance_list.append(instance)
                print(Fore.GREEN + f"Found target track: {match_case_id} in asset: {track_id}" + Style.RESET_ALL)
            
        return instance_list or None
        
        
    def get_events(self, record):
        try:
            t_us_list = []
            for item in record:
                t_us_list.append(int(item.get('meta').get('corr_exposure_start_timestamp_us')))
            asset_id = item.get('public_track_id').split('_')[0]
            rel_path = item.get('meta').get('image_path')
            image_path = Path(self.datablob_dir) / asset_id / rel_path

            event_file_path = Path(self.datablob_dir) / asset_id / "event_blobs/events.h5"
            assert event_file_path.exists()
            
            # Always read the first and last event intervals, independent of sync mode.
            t_min_us_list = [t_us_list[0], t_us_list[-2]]
            t_max_us_list = [t_us_list[1], t_us_list[-1]] 
            height, width, _ = read_image(image_path).shape   
            event_list = extract_from_h5_by_timewindow(str(event_file_path), t_min_us_list, t_max_us_list, self.pixel_diff, [height, width])
            for ev_buffer in event_list:
                if len(ev_buffer['x']) == 0:
                    err_msg = f"Empty event list for asset_id: {asset_id}, image_path='{image_path}'  t_min_us_list={t_min_us_list} t_max_us_list={t_max_us_list}"
                    print(Fore.RED + err_msg + Style.RESET_ALL)
                    self.logger.error(err_msg)
                    return None
        
        except Exception as e:
            self.logger.error(f"get_events error: {e}")
            self.logger.error(traceback.format_exc()) 
        else:
            return event_list
        
    def _select_sync_records(self, record):
        if self.sync == 'front':
            return record[1:-self.window_interval] + record[-1:]
        return record[:-self.window_interval - 1] + record[-2:-1]

    def get_images(self, record):
        subrecord = self._select_sync_records(record)

        try:
            image_list = []
            for item in subrecord:
                asset_id = item.get('public_track_id').split('_')[0]
                rel_path = item.get('meta').get('image_path')
                image_path = Path(self.datablob_dir) / asset_id / rel_path
                
                if not Path(image_path).exists():
                    err_msg = f"Image not found: {image_path}"
                    print(Fore.RED + err_msg + Style.RESET_ALL)
                    self.logger.error(err_msg)
                    return None
                
                image = cv2.imread(str(image_path))
                if image is None:
                    err_msg = f"Image read failed: {image_path}"
                    print(Fore.RED + err_msg + Style.RESET_ALL)
                    self.logger.error(err_msg)
                    return None
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_list.append(image)
        except Exception as e:
            self.logger.error(f"get_images error: {e}")
            self.logger.error(traceback.format_exc()) 
        else:
            return image_list

    def get_masks(self, record):
        subrecord = self._select_sync_records(record)

        mask_list = []
        for item in subrecord:
            asset_id = item.get('public_track_id').split('_')[0]
            rel_path = item.get('meta').get('image_path')
            image_path = Path(self.datablob_dir) / asset_id / rel_path
            mask_path = Path(self.datablob_dir) / asset_id / rel_path.replace('.png', '.npy')
            
            if Path(mask_path).exists():
                mask_record = np.load(mask_path, allow_pickle=True).item()
                mask = mask_record['mask']
            else:
                err_msg = f"Mask not found: {mask_path}"
                if self.split == 'train':
                    self.logger.error(err_msg)
                mask = None
            
            mask_list.append(mask)
        return mask_list

    def get_tracks(self, record):
        
        try:
            subrecord = self._select_sync_records(record)
            
            max_edge = 0
            for item in subrecord:
                xmin, ymin, xmax, ymax = map(int, item['box'])
                square_box = get_expand_box([xmin, ymin, xmax, ymax])
                max_edge = max(max_edge, square_box[3] - square_box[1])
            
            track_list = []
            for item in subrecord:
                item.update({'max_edge': max_edge})
                track_list.append(item)
        except Exception as e:
            self.logger.error(f"get_tracks error: {e}")
            self.logger.error(traceback.format_exc()) 
        else:
            return track_list
    
    def get_collate_fn(self):
        if self.annotation_format == 'parquet':
            return parquet_collate_fn if self.has_labels else inference_collate_fn
        if self.split=='train':
            return my_collate_fn
        elif self.split=='test':
            return sequence_collate_fn
        else:
            return None
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.annotation_format == 'parquet':
            return self._getitem_parquet(idx)
        
        if self.split == 'train':

            try:         
                record = self.db[self.indices[idx]]
                rawdata= {}
                rawdata['tracks'] = self.get_tracks(record)
                rawdata['images'] = self.get_images(record)
                rawdata['masks'] = self.get_masks(record)
                rawdata['events'] = self.get_events(record)
                
                if rawdata['images'] is None or rawdata['events'] is None:
                    raise ValueError(f"Invalid data indices: {idx}")
                return self.load_data(rawdata)
            except Exception as e:
                self.logger.error(f"Skipping data at index {idx}, due to error: {e}")
                self.logger.error(traceback.format_exc())
                return None
            
        elif self.split == 'test':
            sequence_annos = self.db[self.indices[idx]]
            window_size = self.window_interval + 2  
            test_batch = []
            try:
                for window_annos in sliding_window(sequence_annos, window_size):
                    record = get_record_from_window_annos(window_annos, gt_range=None)
                    
                    if record is not None:
                        rawdata= {}
                        rawdata['tracks'] = self.get_tracks(record)
                        rawdata['images'] = self.get_images(record)
                        rawdata['masks'] = self.get_masks(record)
                        rawdata['events'] = self.get_events(record)

                        if rawdata['images'] is None or rawdata['events'] is None:
                            raise ValueError(f"Invalid data indices: {idx}")

                        test_item = self.load_data(rawdata)
                        test_batch.append(test_item)
                
                return test_batch
            
            except Exception as e:
                self.logger.error(f"Skipping data at index {idx}, due to error: {e}")
                self.logger.error(traceback.format_exc())
                return None
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def load_data(self, rawdata, require_labels=True, sample_token=None):
        data_tensor = preprocess_data(rawdata, self.input_feat_size, self.pth_trans, 
                                      data_mode=self.data_mode,
                                      fy=self.fy,
                                      logger=self.logger,
                                      require_labels=require_labels,
                                      sample_token=sample_token)
        return data_tensor
    
        
def preprocess_data(
    rawdata, 
    target_size: List[int],
    transforms,
    fy,
    data_mode='image_event',
    super_resolution_fact=2,
    logger=None,
    require_labels=True,
    sample_token=None,
):

    img_tensor_list = [] 
    evt_tensor_list = []
    ttc_tensor_list = []
    mask_tensor_list = []
    dimension_tensor_list = []
    visible_height_tensor_list = []
    case_id = []
    seq_name = []    
    use_mask_supervison = True
    try:
        for image, event, track, mask in zip(rawdata['images'], rawdata['events'], rawdata['tracks'], rawdata['masks']):
            
            if len(event['x']) == 0:
                continue
            
            case_id.append(f"{track['public_track_id']}_{track['timestamp']}")
            seq_name.append(track.get('seq_name', []))
            
            xmin, ymin, xmax, ymax = map(int, track['box'])
            cx = int((xmin + xmax) / 2.0)
            cy = int((ymin + ymax) / 2.0)
            
            max_edge = track['max_edge']
            square_box = get_square_box(cx=cx, cy=cy, max_edge=max_edge)
            
            image = Image.fromarray(image) 
            
            img_tensor_list.append(
                transforms(image).unsqueeze(0)
                )
            
            if mask is None:
                use_mask_supervison = False
                mask_tensor = -torch.ones(img_tensor_list[-1].shape) 
            else:
                mask_tensor = torch.from_numpy(mask)[None,...]
            
            mask_tensor_list.append(mask_tensor)
            
            x_warp = np.ascontiguousarray(event['x'], dtype=np.int16)
            y_warp = np.ascontiguousarray(event['y'], dtype=np.int16)
            tus_warp = np.ascontiguousarray(event['t'], dtype=np.int64)
            
            box_warp = np.ascontiguousarray(square_box, dtype=np.int16)
            evt_img_roi, _ = get_timevolume_roi_np(expand_box=box_warp,
                                                    x=x_warp,
                                                    y=y_warp,
                                                    tus=tus_warp)
            evt_img_roi = torch.from_numpy(evt_img_roi[None,...])
            
            _, _, evt_h, evt_w = evt_img_roi.shape
            evt_tensor_list.append(
                get_target_roi_from_feature_torch(
                    input_image=evt_img_roi,
                    expand_box=[0, 0, evt_w, evt_h],
                    target_size=target_size
                    )
                )
            
            if require_labels:
                ttc = torch.tensor([track['ttc']], dtype=torch.float32)
                ttc_tensor_list.append(ttc)
                
                box3d_h = track['box3d_h']
                cam_corners = np.array(track['box3d_Fcam'])
                min_depth = cam_corners[:,2].min()
                assert evt_h == max_edge, f"Event height [{evt_h}] is not equal to max_edge [{max_edge}]"
                scaling = target_size[0] / evt_h 
                visible_height = fy * box3d_h / min_depth * scaling
                visible_height_tensor_list.append(visible_height)
                dimension_tensor_list.append(box3d_h)
        
        
        img_tensor_resize = get_target_roi_from_feature_torch(
            input_image=torch.concat(img_tensor_list, dim=1),
            expand_box=square_box,
            target_size=target_size)
        
        super_resolved_target_size = [val * super_resolution_fact for val in target_size]
        mask_tensor_resize = get_target_roi_from_feature_torch(
            input_image=torch.concat(mask_tensor_list, dim=1).to(torch.float32),
            expand_box=square_box,
            target_size=super_resolved_target_size).to(torch.int64)
        
        if mask_tensor_resize[0,0].sum() < 512 or \
            mask_tensor_resize[0,1].sum() < 512:
            use_mask_supervison = False
            
        evt_tensor_resize = torch.concat(evt_tensor_list, dim=1)
        
        if data_mode == "image_event":
            data_tensor_resize = torch.concat([img_tensor_resize, evt_tensor_resize], dim=1)
            
        elif data_mode == "image_only":
            data_tensor_resize = img_tensor_resize
            
        elif data_mode == "event_only":
            data_tensor_resize = evt_tensor_resize
            
        else:
            raise NotImplementedError
        
        del x_warp, y_warp, tus_warp, box_warp
        del rawdata, image, event, track,  img_tensor_list, evt_tensor_list, img_tensor_resize, evt_tensor_resize, evt_img_roi
        ret = {
            'data': data_tensor_resize,
            'case_id': case_id,
            'seq_name': seq_name,
            'mask_target': mask_tensor_resize,
            'use_mask_supervison': use_mask_supervison
            }
        if require_labels:
            ret.update({
                'target': ttc_tensor_list[-1],
                'visible_height': torch.tensor(visible_height_tensor_list,  dtype=torch.float32),
                'dimension': torch.tensor(dimension_tensor_list, dtype=torch.float32),
            })
        if sample_token is not None:
            ret['sample_token'] = sample_token
        return ret
    

    except Exception as e:
        logger.error(f"Process data error: {e}")
        logger.error(traceback.format_exc())
        return None
        
def my_collate_fn(
        batch,
        tensor_keys=['data', 'target', 'mask_target'],
        list_keys=['case_id', 'use_mask_supervison'],
        optional_keys=['visible_height', 'dimension']
        ):
    batch = [d for d in batch if d is not None]
    if not batch:
        return None
    ret = {}
    for key in tensor_keys:
        ret[key] = torch.cat([d[key] for d in batch], dim=0)
    for key in list_keys:
        ret[key] = [d[key] for d in batch]
    for key in optional_keys:
        if key in batch[0]:
            ret[key] = torch.stack([d[key] for d in batch])
    del batch
    return ret

def parquet_collate_fn(
        batch,
        tensor_keys=['data', 'target', 'mask_target'],
        list_keys=['case_id', 'seq_name', 'sample_token', 'use_mask_supervison'],
        optional_keys=['visible_height', 'dimension']
        ):
    batch = [d for d in batch if d is not None]
    if not batch:
        return None
    ret = {}
    for key in tensor_keys:
        if key in batch[0]:
            ret[key] = torch.cat([d[key] for d in batch], dim=0)
    for key in list_keys:
        if key in batch[0]:
            ret[key] = [d[key] for d in batch]
    for key in optional_keys:
        if key in batch[0]:
            ret[key] = torch.stack([d[key] for d in batch])
    del batch
    return ret

def inference_collate_fn(
        batch,
        tensor_keys=['data'],
        list_keys=['case_id', 'seq_name', 'sample_token']
        ):
    batch = [d for d in batch if d is not None]
    if not batch:
        return None
    ret = {}
    for key in tensor_keys:
        ret[key] = torch.cat([d[key] for d in batch], dim=0)
    for key in list_keys:
        if key in batch[0]:
            ret[key] = [d[key] for d in batch]
    del batch
    return ret

def sequence_collate_fn(
        batch,
        tensor_keys=['data', 'target'],
        list_keys=['case_id', 'seq_name'],
        optional_keys=['visible_height', 'dimension']
        ):
    ret = {}
    sequences = [d for d in batch if d is not None]
    if not sequences:
        return None
    batch = [item for sequence in sequences for item in sequence if item is not None]
    if not batch:
        return None
    for key in tensor_keys:
        ret[key] = torch.cat([d[key] for d in batch], dim=0)
    for key in list_keys:
        ret[key] = [d[key] for d in batch]
    for key in optional_keys:
        if key in batch[0]:
            ret[key] = torch.stack([d[key] for d in batch])
    del batch
    return ret

def sliding_window(lst, window_size, step_size=1):
    for i in range(0, len(lst) - window_size + 1, step_size):
        yield lst[i:i + window_size]
        
        
def get_zytname_from_tus(timestamp_us: int) -> str:
    ts_s_part = timestamp_us // int(1e6)
    ts_us_part = int((timestamp_us % int(1e6)) * 1e3)
    return f"{ts_s_part:012d}_{ts_us_part:012d}"


def get_instance_from_time_window(window, track_anno, track_id, gt_range=None, box3d_h=None):
    instance = []
    diffs = [abs(window[i] - window[i+1]) for i in range(len(window)-1)]
    if max(diffs) > 100000:
        return None
    if box3d_h is None or box3d_h < 0 or box3d_h > 10:
        err_msg = f"Invalid box3d_h: {box3d_h}"
        print(Fore.RED + err_msg + Style.RESET_ALL)
        raise ValueError(err_msg)
    for tus in window:
        zytname = get_zytname_from_tus(int(tus))
        anno = track_anno.get(zytname)
        anno.update(track_id=track_id)
        anno.update(box3d_h=box3d_h)
        instance.append(anno)
        if gt_range is not None:
            if anno.get('ttc') < gt_range[0] or anno.get('ttc') > gt_range[1]:
                return None
        
    return instance


def get_instance_from_continue_sequence(cnt_sequence, win_ts, track_anno, track_id, ttc_range=None, box3d_h=None):
    instance = []
    window_size = win_ts * 10 + 1
    if box3d_h is None or box3d_h < 0 or box3d_h > 10:
        err_msg = f"Invalid box3d_h: {box3d_h}"
        print(Fore.RED + err_msg + Style.RESET_ALL)
        raise ValueError(err_msg)
    for window in sliding_window(cnt_sequence, window_size=window_size, step_size=2*10):
        if len(window) < window_size:
            continue
        out_range_count = 0
        window_annos = []
        
        t_us_list = []
        for name_tus in window:
            zytname = get_zytname_from_tus(int(name_tus))
            anno = track_anno.get(zytname)
            t_us_list.append(int(anno.get('meta').get('corr_exposure_start_timestamp_us')))
        t_min_us_list = t_us_list[:-1]
        seq_name = f"{track_id}_{t_min_us_list[0]}_{t_min_us_list[-1]}"
    
        for tus in window:
            zytname = get_zytname_from_tus(int(tus))
            anno = track_anno.get(zytname)
            anno.update(track_id=track_id)
            anno.update(box3d_h=box3d_h)
            anno.update(seq_name=seq_name)
            window_annos.append(anno)
            
            ttc = anno.get('ttc')
            if ttc<ttc_range[0] or ttc>ttc_range[1] or ttc==0:
                out_range_count += 1

        if out_range_count > 0.05 * window_size:
            continue
        else:
            instance.append(window_annos)
    
    return instance or None

    
def get_record_from_window_annos(window_annos, gt_range=None):
    record = []
    diffs = [abs(window_annos[i]['timestamp'] - window_annos[i+1]['timestamp']) \
        for i in range(len(window_annos)-1)]
    if max(diffs) > 100000:
        return None
    box3d_h = window_annos[0].get('box3d_h')

    for anno in window_annos:
        assert anno.get('box3d_h') == box3d_h
        record.append(anno)
        if gt_range is not None:
            if anno.get('ttc') < gt_range[0] or anno.get('ttc') > gt_range[1]:
                return None
        
    return record

def get_target_roi_from_feature_torch(input_image, expand_box, target_size):
    
    _, _, img_h, img_w = input_image.shape  # (B, C, H, W)
    target_height, target_width = target_size
    xmin, ymin, xmax, ymax = expand_box

    xs = torch.linspace(xmin, xmax, steps=target_width)
    ys = torch.linspace(ymin, ymax, steps=target_height)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    coords = torch.cat([x[...,None], y[...,None]], dim=2)[None,...]

    coords[:,:,:,0] = coords[:,:,:,0] / img_w * 2 -1
    coords[:,:,:,1] = coords[:,:,:,1] / img_h * 2 -1     

    resize_image = F.grid_sample(input_image, 
                                 coords, 
                                 mode='bilinear', 
                                 padding_mode='zeros', 
                                 align_corners=True)
    return resize_image


def get_expand_box(vd_box: List[int]) -> List[int]:
    """
    Expand the box to a square box.
    """
    
    xmin, ymin, xmax, ymax = vd_box
    
    cx, cy = int((xmin + xmax) / 2.0), int((ymin + ymax) / 2.0)
    
    width, height = xmax - xmin, ymax - ymin
    max_edge = max(width, height)
    
    expand_x1 = math.ceil(cx - max_edge / 2.0)
    expand_y1 = math.ceil(cy - max_edge / 2.0)
    expand_x2 = math.ceil(cx + max_edge / 2.0)
    expand_y2 = math.ceil(cy + max_edge / 2.0)
    
    assert expand_x1 <= expand_x2 or expand_y1 <= expand_y2

    return [expand_x1, expand_y1, expand_x2, expand_y2]

    
def get_square_box(cx: int, cy: int, max_edge: int) -> List[int]:
    """
    Expand the box to a square box.
    """
    
    square_x1 = math.ceil(cx - max_edge / 2.0)
    square_y1 = math.ceil(cy - max_edge / 2.0)
    square_x2 = math.ceil(cx + max_edge / 2.0)
    square_y2 = math.ceil(cy + max_edge / 2.0)
    
    assert square_x1 <= square_x2 or square_y1 <= square_y2

    return [square_x1, square_y1, square_x2, square_y2]
