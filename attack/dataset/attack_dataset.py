import json
import os
from loguru import logger
from .example import Example
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset

class AttackDataset:
    def __init__(self, path, subset_slice=None):
        if isinstance(path, list):
            self.data = path
        elif os.path.exists(path):
            # logger.warning('You chose to load a dataset from a local file. If it is not a json/jsonl file, please make sure you correctly set \'local_file_type\' parameter when you instantiate the AttackDataset.')

            if path.endswith('.json'):
                with open(path) as f:
                    _data = json.load(f)
            elif path.endswith('.jsonl'):
                _data = pd.read_json(path, lines=True).to_dict('records')
            elif path.endswith('.csv'):
                _data = pd.read_csv(path).to_dict('records')
            elif path.endswith('.xlsx'):
                _data = pd.read_excel(path).to_dict('records')
            elif path.endswith('.parquet'):
                _data = pd.read_parquet(path).to_dict('records')

            self.data = [Example(**d) for d in _data]

        elif 'thu-coai/AISafetyLab_Datasets' in path:
            file_name = path.split('/')[-1]
            if '_' in file_name:
                data_name = file_name.split('_')[0]
                split = file_name.split('_')[1]
            else:
                data_name = file_name
                split = 'test'

            _data = load_dataset("thu-coai/AISafetyLab_Datasets", data_name, split=split)
            self.data = [Example(**d) for d in _data]

        else:
            logger.error('Please make sure the path is a valid local file path or a huggingface dataset path. For local files, we support the following types: json, jsonl, csv, xlsx, parquet. For huggingface datasets, we now only support the datasets in the "thu-coai/AISafetyLab_Datasets" repository. Please open an issue if you want to use other datasets.')
            raise NotImplementedError(
                'Please make sure the path is a valid local file path or a huggingface dataset path. For local files, we support the following types: json, jsonl, csv, xlsx, parquet. For huggingface datasets, we now only support the datasets in the "thu-coai/AISafetyLab_Datasets" repository. Please open an issue if you want to use other datasets.'
            )

        # logger.info(f'subset slice:{subset_slice}')
        if subset_slice is not None:
            if isinstance(subset_slice, int):
                self.data = self.data[:subset_slice]
            elif isinstance(subset_slice, list):
                _slice = slice(*subset_slice)
                self.data = self.data[_slice]
            else:
                self.data = self.data[subset_slice]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class AugmentDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, augment_target, shuffle):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        self.effective_dataset_size = len(self.dataset)
        self.aufgment_target = augment_target

        self.process_fn = lambda s: s.replace("Sure, here is", "Sure, here's")
        self.process_fn2 = lambda s: s.replace("Sure, h", "H")

    def __iter__(self):
        for batch in super(AugmentDataLoader, self).__iter__():
            if self.aufgment_target:
                targets = []
                for target in batch["target"]:
                    if np.random.random() < 0.5:
                        target = self.process_fn(target)
                    if np.random.random() < 0.5:
                        target = self.process_fn2(target)
                    targets.append(target)
                batch["target"] = targets
            yield batch

def get_dataloader(data_pth, batch_size, shuffle, augment_target):
    dataset = AttackDataset(path=data_pth)
    dataloader = AugmentDataLoader(
        dataset, augment_target=augment_target, shuffle=shuffle, batch_size=batch_size
    )
    return dataloader
