# COOPER and F-COOPER
Official code of COOPER and F-COOPER 

## Environment
(Please follow SECOND for KITTI object detection)
python 3.6, pytorch 1.0.0. Tested in Ubuntu 16.04.

It is recommend to use Anaconda package manager.

```bash
conda install shapely fire pybind11 pyqtgraph tensorboardX protobuf
```
Follow instructions in https://github.com/facebookresearch/SparseConvNet to install SparseConvNet.

Install Boost geometry:

```bash
sudo apt-get install libboost-all-dev
```

Add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

Add  ~/F-COOPER path to PYTHONPATH


## Prepare dataset

* Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

* Create kitti infos:

```bash
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

* Create reduced point cloud:

```bash
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```

* Create groundtruth-database infos:

```bash
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
```

* Download T&J dataset

Tom and Jerry Dataset (in our COOPER and F-COOPER papers) [T&J](https://drive.google.com/file/d/1xmQppUjvaGHbNOTkB_pwVy2HN85I-YHF/view?usp=sharing) to overwrite LiDAR frames in velodyne and velodyne_reduced folders
Modify the data information in ~/F-COOPER/COOPER/data/ImageSets/.

* Modify config file
```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```

## Train

```bash
python ./pytorch/train.py train --config_path=./configs/car.config --model_dir=/path/to/model_dir
```
You can download pretrained models in [Car_detection](https://drive.google.com/file/d/17OPH4YKlvGwuDumdVoz_5sGQ8nviCojr/view?usp=sharing). The car model is corresponding to car.config.

## Evaluate

```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.config --model_dir=/path/to/model_dir
```

## Viewer

run ```python ./kittiviewer/viewer.py```, check following picture to view:
![GuidePic](https://raw.githubusercontent.com/Aug583/F-COOPER/master/images/result.png)

## Citation

If you use related work, please cite our papers:


    @misc{1905.05265,
        Author = {Qi Chen and Sihai Tang and Qing Yang and Song Fu},
        Title = {Cooper: Cooperative Perception for Connected Autonomous Vehicles based on 3D Point Clouds},
        Year = {2019},
        Eprint = {arXiv:1905.05265},
    }


