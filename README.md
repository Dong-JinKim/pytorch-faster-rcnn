# About this fork
This is a fork of `ruotianluo/pytorch-faster-rcnn` with a simplified script to extract boxes, scores, and features from any set of images and dump them in a directory. Here I will just describe the additions I made to the original repository. For installation instructions please refer to the [ORIGINAL_README.md](ORIGINAL_README.md). 

# What's new?
The following python script has been added:
```
tools/extract_boxes_scores_features.py
```

The python script allows you to easily extract detected bounding boxes, object class scores, NMS keep ids, and the last layer features that may then be used in a downstream application. The script takes in a single argument `im_in_out_json`. This is the path to a json file that specifies the paths to the images on which you want to run the object detector and the directory where you want the outputs to be saved. 

# Structure of the `im_in_out_json` file

I created this repository for a project on HICO-Det dataset. So here's a sample of what the .json file actually looked like:
```
[
    {
        "in_path": "/home/ssd/hico_det_clean_20160224/images/train2015/HICO_train2015_00000001.jpg",
        "out_dir": "/home/ssd/hico_det_processed_20160224/faster_rcnn_boxes",
        "prefix": "HICO_train2015_00000001_"
    },
    {
        "in_path": "/home/ssd/hico_det_clean_20160224/images/train2015/HICO_train2015_00000002.jpg",
        "out_dir": "/home/ssd/hico_det_processed_20160224/faster_rcnn_boxes",
        "prefix": "HICO_train2015_00000002_"
    },
    {
        "in_path": "/home/ssd/hico_det_clean_20160224/images/train2015/HICO_train2015_00000003.jpg",
        "out_dir": "/home/ssd/hico_det_processed_20160224/faster_rcnn_boxes",
        "prefix": "HICO_train2015_00000003_"
    },
    ...
]
```

Essentially this is a list of dictionaries saved to a json file. Each dictionary specifies the following:
- `in_path`: path to the input image
- `out_dir`: directory where the extracted boxes, scores, nms ids, and features will be saved
- `prefix`: this specifies any prefix to be added to the filename while writing extracted data to `out_dir`

# Sample output
When executed, for the first image we would see the following files written to `out_dir`:
- `HICO_train2015_00000001_scores.npy` (\<prefix>scores.npy)
- `HICO_train2015_00000001_boxes.npy` (\<prefix>boxes.npy)
- `HICO_train2015_00000001_fc7.npy` (\<prefix>fc7.npy)
- `HICO_train2015_00000001_nms_keep_indices.npy` (\<prefix>nms_keep_indices.npy)

# How to Run
Assuming you have a trained model checkpoint and `im_in_out.json` file available:
- Update variable `saved_model_path` in `extract_boxes_scores_features.py` file to point to the checkpoint location
- Make sure the correct network architecture is being instantiated in line 133. Defaults to Resnet-152 
- Run extraction as follows:
    ```
    python -m tools.extract_boxes_scores_features --im_in_out_json <path to im_in_out.json>
    ```