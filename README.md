# PointPillars
在GTX 2070的设备下，使用FP16模式的trt模型，前向推理一次耗时小于 **16** ms
 - 1. 取消了手动生成`anchors`的步骤， 将其直接放置在`backbone`中，简化操作，充分利用半精度特性加速推理,但是真是提速了多少？好像并没有 。 
 - 2. 使用与[OpenPCDent]()完全相同的[配置文件(pp_0.2_20_6cls.yaml)](./pointpillars/cfgs/pp_0.2_20_6cls.yaml)构建网络,简便部署 。（但是目前[后处理](./pointpillars/postprocess.cu)里面对`multihead`的使用是写死的，所以目前还是个“花把式”，不实用。

## Overview


## Usage
```bash
mkdir build && cd build
cmake ..
make -j8 && ./test/test_model
```

## Result
```bash
------------------------------------
Module        Time        
------------------------------------
Preprocess    0.35306  ms
Pfe           0.201473 ms
Scatter       0.00389  ms
Backbone      13.8364  ms
Postprocess   1.03954  ms
Summary       15.439   ms
------------------------------------
```
## Visualization
```bash
cd tools
python viewer.py
```
<p align="left">
  <img width="600" alt="fig_method" src=docs/python.png>
  <img width="600" alt="fig_method" src=docs/src.png>
</p>
上图为PYTHON实现的demo ， 下图为C++实现的demo。
## Warning
- yaml-cpp 编译是要用动态库模式


## 存在及问题以及解决方案
- 问题： 放在xavier上用不了！可能是因为xavier上的 `TensorRT 6.0.1`版本比较低导致
    ```bash
    While parsing node number 64 [Gather]:
    3
    ERROR: /home/nvidia/Opensource/onnx-tensorrt/builtin_op_importers.cpp:703 In function importGather:
    [8] Assertion failed: !(data->getType() == nvinfer1::DataType::kINT32 && nbDims == 1) && "Cannot perform gather on a shape tensor!"
    ERROR: failed to parse onnx file
    ```
   方案：使用`TensorRT 7.1.3`或者把下图黑框部分用c++写出来，不放在网络里面。问题所在截图，图中黑框部分。
<p align="center">
<img width="600" alt="fig_method" src=docs/bug_1.png>
</p>