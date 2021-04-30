# PointPillars
**高度优化的点云目标检测网络[PointPillars](https://github.com/traveller59/second.pytorch)。主要通过tensorrt对网络推理段进行了优化，通过cuda/c++对前处理后处理进行了优化。做到了真正的事实处理（前处理+后处理小于3ms）。**

## Major Advance
- **训练简单**
  
    本仓库直接使用[**mmlab/OpenPCdet**](https://github.com/open-mmlab/OpenPCDet)进行训练。所以只要你按照[**官方教程**](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md)的教程是非常容易训练自己的数据，也可以直接采用[**官方训练参数**](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md)来进行部署。但是由于需要使用**TensorRT**,需要对官方版本的网络进行一些客制化，我将我的修改版本上传至[**]，

- **部署简单**
   
    本仓库在[**Autoware.ai/core_perception/lidar_point_pillars**](https://github.com/Autoware-AI/core_perception/tree/master/lidar_point_pillars)和[**Apollo/modules/perception/lidar/lib/detection/lidar_point_pillars**](https://github.com/ApolloAuto/apollo/tree/master/modules/perception/lidar/lib/detection/lidar_point_pillars)的基础上,修改了信息传递方式，删除了冗余的东西，增加了**MultiHead**功能。



## Usage
```bash

```

## Result
```bash
------------------------------------------------------------------
>>>>                                                          >>>>
                                                                  
Input filename:   ../model/cbgs_pp_multihead_pfe.trt
                                                                  
>>>>                                                          >>>>
------------------------------------------------------------------
                                                                  
                                                                  
------------------------------------------------------------------
>>>>                                                          >>>>
                                                                  
Input filename:   ../model/cbgs_pp_multihead_backbone.trt
                                                                  
>>>>                                                          >>>>
------------------------------------------------------------------
                                                                  
------------------------------------
Module        Time        
------------------------------------
Preprocess    0.459405 ms
Pfe           4.2454   ms
Scatter       0.007755 ms
Backbone      15.5444  ms
Postprocess   7.21689  ms
Summary       27.4806  ms
------------------------------------
```
## Visualization
```bash
cd tools
python viewer.py
```
<p align="left">
  <img width="1200" alt="fig_method" src=docs/demo.png>
</p>

**左图为本仓库实现的demo，右图为OpenPCdet实现的demo**

