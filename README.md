# SOE-Net: A Self-Attention and Orientation Encoding Networkfor Point Cloud based Place Recognition 
This repository is the official implementation for paper 

SOE-Net: A Self-Attention and Orientation Encoding Network for Point Cloud based Place Recognition

Yan Xia, Yusheng Xu, Shuang Li, Rui Wang, Juan Du, Daniel Cremers, Uwe Stilla

Technical University of Munich, Beijing Insitute of Technology, Artisense

#### Introduction

SOE-Net fully explores the relationship between points and incorporates long-range context into point-wise local descriptors. Local information of each point from eight orientations is captured in a PointOE module, whereas long-range feature dependencies among local descriptors are captured with a self-attention unit. Moreover, we propose a novel loss function called Hard Positive Hard Negative quadruplet loss (HPHN quadruplet), that achieves better performance than the commonly used lazy quadruplet loss.

#### Pre-requisites

- Python3.6
- Tensorflow1.4.0
- CUDA-9.0
- Scipy
- Pandas
- Sklearn
- The TF operators under tf_ops folder should be compiled.
- generate pickle files, refer to [PointNetVLAD](https://github.com/mikacuy/pointnetvlad).

#### Training

```python
python train.py
```

#### Evaluation

```python
python evaluate.py
```

#### References

1. Mikaela Angelina Uy and Gim Hee Lee. Pointnetvlad: Deep point cloud based retrieval for large-scale place recognition.In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4470–4479, 2018. 1, 2, 3, 5, 6.
2. Wenxiao Zhang and Chunxia Xiao.  Pcan:  3d attention map learning using contextual information for point cloud based retrieval.  In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 12436–12445, 2019. 2, 3, 6, 10.