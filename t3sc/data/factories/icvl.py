import logging
import os
import torch
import h5py
import numpy as np
import re
from datasets import load_dataset
from t3sc.data.normalizers import GlobalMinMax
from .base_factory import DatasetFactory
from .utils import touch
from t3sc.data.splits import (
    icvl_train,
    icvl_val,
    icvl_test,
    icvl_crops,
    icvl_rgb,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ICVL(DatasetFactory):
    NAME = "ICVL"
    IMG_SHAPE = (31, 1024, 1024)
    CROPS = icvl_crops
    RGB = icvl_rgb

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.split == 0

        self.f_train = icvl_train
        self.f_val = icvl_val
        self.f_test = icvl_test
        
        # Load dataset from Hugging Face
        self.dataset = load_dataset("danaroth/icvl")
        
        # Extract filenames from dataset
        self.file_names = self._extract_filenames()

    def _extract_filenames(self):
        """Extracts and cleans filenames from the dataset."""
        file_names = []
        for i in range(len(self.dataset["train"])):
            image_path = self.dataset["train"][i]["image"].filename
            file_name = os.path.splitext(os.path.basename(image_path))[0]
            clean_name = re.sub(r"-200x215$", "", file_name)  # Remove size suffix
            file_names.append(clean_name)
        return file_names

    @classmethod
    def download(cls, path_data):
        """Simulates dataset download using Hugging Face."""
        path_dataset = os.path.join(path_data, cls.NAME)
        path_raw = os.path.join(path_dataset, "raw")
        path_dl_complete = os.path.join(path_raw, ".download_complete")
        
        if os.path.exists(path_dl_complete):
            logger.info("Dataset already downloaded")
            return
        
        logger.info("Downloading dataset from Hugging Face")
        load_dataset("danaroth/icvl")
        os.makedirs(path_raw, exist_ok=True)
        touch(path_dl_complete)
        logger.info("Dataset download complete")

    def preprocess(self):
        path_source = os.path.join(self.path_data, self.NAME, "raw")
        path_dest = os.path.join(self.path_data, self.NAME, "clean")
        path_complete = os.path.join(path_dest, ".complete")
        if os.path.exists(path_complete):
            return

        os.makedirs(path_dest, exist_ok=True)

        normalizer = GlobalMinMax()
        icvl_all = list(set(self.f_train + self.f_test + self.f_val))
        
        for i, fn in enumerate(self.file_names):
            path_out = os.path.join(path_dest, f"{fn}.pth")
            if os.path.exists(path_out):
                continue
            
            logger.info(f"Processing {fn}")
            img = np.array(self.dataset["train"][i]["image"], dtype=np.float32)
            img_torch = torch.tensor(img, dtype=torch.float32)
            img_torch = normalizer.transform(img_torch).clone()
            
            torch.save(img_torch, path_out)
            logger.info(f"Saved image {i + 1}/{len(self.file_names)} to {path_out}")
        
        touch(path_complete)
        logger.info("Dataset preprocessing complete")
