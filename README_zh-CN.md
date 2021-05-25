[**English**](README.md) | **简体中文**

# PointPillars
**高度优化的点云目标检测网络[PointPillars](https://github.com/traveller59/second.pytorch)。主要通过tensorrt对网络推理段进行了优化，通过cuda/c++对前处理后处理进行了优化。做到了真正的实时处理（前处理+后处理小于 1 ms/Head）。**

## Major Advance
- **训练简单**
  
    本仓库直接使用[**mmlab/OpenPCdet**](https://github.com/open-mmlab/OpenPCDet)进行训练。所以只要你按照[**官方教程**](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md)的教程是非常容易训练自己的数据，也可以直接采用**官方训练参数**来进行部署。但是由于需要使用**TensorRT**,需要对官方版本的网络进行一些客制化，我将我的修改版本上传至[*hova88/OpenPCdet*](https://github.com/hova88/OpenPCDet)。

- **部署简单**
   
    本仓库在[**Autoware.ai/core_perception/lidar_point_pillars**](https://github.com/Autoware-AI/core_perception/tree/master/lidar_point_pillars)和[**Apollo/modules/perception/lidar/lib/detection/lidar_point_pillars**](https://github.com/ApolloAuto/apollo/tree/master/modules/perception/lidar/lib/detection/lidar_point_pillars)的基础上,修改了信息传递方式，删除了冗余的东西，增加了**MultiHead**功能。


## Requirements (My Environment)
### For *.onnx and *.trt engine file
* Linux Ubuntu 18.04
* OpenPCdet
* ONNX IR version:  0.0.6
* [onnx2trt](https://github.com/onnx/onnx-tensorrt)
  
### For algorithm: 
* Linux Ubuntu 18.04
* CMake 3.17 (版本太低的话cmakelists.txth会找不到cuda)
* CUDA 10.2
* TensorRT 7.1.3 (7以下是不行的)
* yaml-cpp
* google-test (非必须)

### For visualization
* [open3d](https://github.com/intel-isl/Open3D)


## Usage

0. **下载两个工程,并解决环境问题**
   ```bash
   mkdir workspace && cd workspace
   git clone https://github.com/hova88/PointPillars_MultiHead_40FPS.git --recursive && cd ..
   git clone https://github.com/hova88/OpenPCDet.git 
   ```


1. **获取 Engine File**

    - 1.1 **Pytorch model --> ONNX model :** 具体转换教程，我放在了[**hova88/OpenPCdet**]((https://github.com/hova88/OpenPCDet))的**change log**里面。
        * [cbgs_pp_multihead_pfe.onnx](https://drive.google.com/file/d/1iEXjWBPzVr8YVWDA38eCGqk0wQoIrTKD/view?usp=sharing)
        * [cbgs_pp_multihead_backbone.onnx](https://drive.google.com/file/d/19mW-GXjilcRSHiq-hgSVdu5GOefx-1yR/view?usp=sharing)

    - 1.2 **ONNX model --> TensorRT model :** 安装[onnx2trt](https://github.com/onnx/onnx-tensorrt)之后，就非常简单。注意，想要加速推理速度，一定要用半精度/混合精度，即（-d 16)
        ```bash
            onnx2trt cbgs_pp_multihead_pfe.onnx -o cbgs_pp_multihead_pfe.trt -b 1 -d 16 
            onnx2trt cbgs_pp_multihead_backbone.onnx -o cbgs_pp_multihead_backbone.trt -b 1 -d 16 
        ```

    - 1.3 **engine file --> algorithm :** 在`bootstrap.yaml`, 指明你生成的两组engine file (*.onnx , *.trt)的路径。 
    - 1.4 下载测试点云[nuscenes_10sweeps_points.txt](https://drive.google.com/file/d/1enCbjwe4giwGC-x7Wjns4eHx2njZW2Jl/view?usp=sharing) ，并在`bootstrap.yaml`指明输入（clouds）与输出(boxes)路径。

2. **编译**

    ```bash
    cd PointPillars_MultiHead_40FPS
    mkdir build && cd build
    cmake .. && make -j8 && ./test/test_model
    ```

3. **可视化**

    ```bash
    cd PointPillars_MultiHead_40FPS/tools
    python viewer.py
    ```
**左图为本仓库实现的demo，右图为OpenPCdet实现的demo**
<p align="left">
  <img width="2000" alt="fig_method" src=docs/demo.png>
</p>

## Result

### Use *.onnx engine file
```bash
----------------------------------------------------------------
Input filename:   ../model/cbgs_pp_multihead_pfe.onnx
ONNX IR version:  0.0.6
Opset version:    12
Producer name:    pytorch
Producer version: 1.7
Domain:           
Model version:    0
Doc string:       
----------------------------------------------------------------
WARNING: [TRT]/home/hova/onnx-tensorrt/onnx2trt_utils.cpp:220: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
----------------------------------------------------------------
Input filename:   ../model/cbgs_pp_multihead_backbone.onnx
ONNX IR version:  0.0.6
Opset version:    10
Producer name:    pytorch
Producer version: 1.7
Domain:           
Model version:    0
Doc string:       
----------------------------------------------------------------
------------------------------------
Module        Time        
------------------------------------
Preprocess    0.455046 ms
Pfe           0.319025 ms
Scatter       0.004159 ms
Backbone      33.1782  ms
Postprocess   7.17682  ms
Summary       41.1385  ms
------------------------------------
```

### Use *.trt engine file 
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

# License

GNU General Public License v3.0 or later
See [`COPYING`](LICENSE.md) to see the full text.
