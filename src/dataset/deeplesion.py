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
from enum import Enum

import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


__all__ = ["load_deeplesion_instances", "register_deeplesion"]

class DataType(Enum):
    Train = 1
    Val = 2
    Test = 3


DEEPLESION_NAMES= { DataType.Train: "DeepLesion"+DataType.Train.name,
                    DataType.Val:    "DeepLesion"+DataType.Val.name,
                    DataType.Test:   "DeepLesion"+DataType.Test.name
                }

CLASS_NAMES = [ "others",
                "bone",  
                "abdomen", 
                "mediastinum", 
                "liver", 
                "lung", 
                "kidney",
                "soft tissue", 
                "pelvis"]




def load_deeplesion_instances(img_dir: str, data_type: DataType):
    dataset_dicts = {}
    csv_file = os.path.join(img_dir, "DL_info.csv")
    patient_df = pd.read_csv(csv_file)

    for idx, row in patient_df.iterrows():
        dt = row["Train_Val_Test"]
        if dt != data_type.value:
            continue
        img_path = os.path.join(img_dir,
                                'Images_png',
                                '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(
                                    **row),
                                '{Key_slice_index:03d}.png'.format(**row))

        if img_path in dataset_dicts:
            record = dataset_dicts[img_path]
        else:
            record = {}
            record["file_name"] = img_path
            record["image_id"] = idx
            record["width"] = int(row["Image_size"].split(",")[0])
            record["height"] = int(row["Image_size"].split(",")[1])
            record['dicom_windows'] = row["DICOM_windows"]
            record["annotations"] = []

        record["annotations"].append(_create_bbox(row))

        dataset_dicts[img_path] = record

    return list(dataset_dicts.values())


def register_deeplesion(img_path:str):
    for data_type in [DataType.Train, DataType.Val, DataType.Test]:
        dataset_name = DEEPLESION_NAMES[data_type]
        DatasetCatalog.register(dataset_name,lambda d=data_type: load_deeplesion_instances(img_path, d))
        MetadataCatalog.get(dataset_name).set(thing_classes=CLASS_NAMES,img_path=img_path)

def _create_measuremnt(row):
    measurement_coordicats = row["Measurement_coordinates"].split(",")
    diameters = row["Lesion_diameters_Pixel_"].split(",")
    long_axis = [
        float(measurement_coordicats[0]),
        float(measurement_coordicats[1]),
        float(measurement_coordicats[2]),
        float(measurement_coordicats[3]),
        float(diameters[0]),
    ]

    short_axis = [
        float(measurement_coordicats[4]),
        float(measurement_coordicats[5]),
        float(measurement_coordicats[6]),
        float(measurement_coordicats[7]),
        float(diameters[1]),
    ]

    return long_axis, short_axis


def _create_bbox(row):
    bounding_box = row["Bounding_boxes"].split(",")
    return {
        "bbox": [
            float(bounding_box[0]),
            float(bounding_box[1]),
            float(bounding_box[2]),
            float(bounding_box[3])
        ],
        "bbox_mode": BoxMode.XYXY_ABS,
        "category_id": int(row["Coarse_lesion_type"]) if int(row["Coarse_lesion_type"]) > 0 else 0,
        "long_axis": _create_measuremnt(row)[0],
        "short_axis": _create_measuremnt(row)[1],
    }
