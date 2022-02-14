# #### BEGIN LICENSE BLOCK #####
# MIT License
#    
# Copyright (c) 2021 Bin.Li (ornot2008@yahoo.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# #### END LICENSE BLOCK #####
# /

import os
import random
import sys

import cv2

current_dir= os.path.dirname(os.path.realpath(__file__))
work_folder=current_dir[:current_dir.find('unittest')]
sys.path.append(work_folder+'src/dataset')

import unittest

from torch.utils.tensorboard import SummaryWriter

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

from deeplesion import DataType, load_deeplesion_instances


class TestDeepLesion(unittest.TestCase):

    def setUp(self) -> None:
        self.writer = SummaryWriter(work_folder+'log/')
        deep_lesion_path = "/home/yan/data/deeplesion"
        self.dataset_dicts = load_deeplesion_instances(deep_lesion_path, DataType.Train)
    
    def test_deeplesion_dataset(self):
        deeplesion_metadata = MetadataCatalog.get("Deeplesion_Train") 
        d = self.dataset_dicts[111]       
        
        img = cv2.imread(d["file_name"])
        img= cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        visualizer = Visualizer(img[:, :, ::-1], metadata=deeplesion_metadata, scale=1.5)
        out = visualizer.draw_dataset_dict(d)
        img = out.get_image()[:, :, ::-1]
        self.writer.add_image('train',img,d['image_id'],dataformats='HWC')


if __name__ == "__main__":
    print("Lesion Tracker Testing....")
    unittest.main()