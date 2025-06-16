import cv2
import glob
import numpy as np
import os
import pandas as pd
import re
from torch.utils.data import Dataset
from itertools import compress
from annotator.hed import HEDdetector
from annotator.util import HWC3

class MyDataset(Dataset):
    def __init__(self, image_dir, data_dir, hint_dir):

        self.image_dir = image_dir
        self.data_dir = data_dir
        self.hint_dir = hint_dir
       
        df = pd.DataFrame() 
        for f in self.data_dir: 
            df = pd.concat([df, pd.read_csv(f)])
        #print(f"Reading data from: {self.data_dir}")
        #df = pd.read_csv(self.data_dir)
        #df = df.head(10)
        print("you are using Yuzhou script")
        self.description = df
        
        #self.image_list_xtile = self.description['longitude'].to_list()
        #self.image_list_ytile = self.description['latitude'].to_list()
        self.image_list_id = self.description['Image_Name'].to_list()
        
        #self.image_city = self.description['city'].to_list()
 
        self.description = self.description['description'].to_list()
        
    def __len__(self):
        return len(self.image_list_id)
    


    def __getitem__(self, item):

        #xtile = self.image_list_xtile[item]
        #ytile = self.image_list_ytile[item]
        imgid = self.image_list_id[item]
        #city = self.image_city[item]

        if '_' in str(imgid):
            img_name = self.image_dir + "/" + str(imgid) + ".jpg"
            #hint_name = "fml/Streetview images/hint/blank_white.png"
            hint_name = self.hint_dir + "/" + str(imgid) + '_road_segmentation' + ".png"
        else:
            img_name = self.image_dir + "/" + str(imgid) + ".jpg"
            hint_name = self.hint_dir + "/" + str(imgid) + '_road_segmentation' + ".png"
            #hint_name = "fml/Streetview images/hint/blank_white.png"
        target = cv2.imread(img_name)
        if target is None:
            raise FileNotFoundError(f"Target image not found or failed to load: {img_name}")
        source = cv2.imread(hint_name, cv2.IMREAD_UNCHANGED)
        if source is None:
            print(f"Hint image not loaded: {hint_name}. Using blank_white.png as default.")
            hint_name = os.path.join(self.hint_dir, f"{imgid}_road_segmentation.png")
            #hint_name = "fml/Streetview images/hint/blank_white.png"
            source = cv2.imread(hint_name, cv2.IMREAD_UNCHANGED)
        #if source is None:
            #print(hint_name)
        source = cv2.resize(source, (640, 640))
        #print(f"Source shape: {source.shape}")
        #print(f"Target shape: {target.shape}")
        # convert 4-channel source image to 3-channel
        #make mask of where the transparent bits are
        #trans_mask = source[:,:,3] == 0

        #replace areas of transparency with white and not transparent
        #source[trans_mask] = [255, 255, 255, 255]

        #new image without alpha channel...
        source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
        
        #OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        prompt = self.description[item]
        
        
        return dict(jpg=target, txt=prompt, hint=source, filename=imgid)

print("run successed")
