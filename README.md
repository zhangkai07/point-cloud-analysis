# Pytorch code for DQRNet

**Quaternion-Enhanced Dual-Branch Network for Robust 3D Point Cloud Analysis**

### Requirement


python -m pip install -r requirements.txt

### 1、Classification

#### Dataset

Download the ModelNet40 dataset (xyz and labels) from [here](https://modelnet.cs.princeton.edu/download.html) and put it under `data`


#### Train:

`python train.py`

#### Test:

`python test.py`


### 2、Part Segmentation 

#### Dataset

Download the ShapeNetPart dataset (xyz, normals and labels) from [here](http://shapenet.cs.stanford.edu/) and put it under `data`

#### Train:

`python train.py`

#### Test:

`python test.py`


### 3、Indoor scene segmentation 

#### Dataset

Download the S3DIS dataset from [here](https://cvg-data.inf.ethz.ch/s3dis/) and put it under `data`


#### Train:

`python train.py`

#### Test:

`python test.py`


## Reference
 
