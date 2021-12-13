
**English** | [**简体中文**](README_zh-CN.md)

# PointPillars
**High performance version of 3D object detection network -[PointPillars](https://github.com/traveller59/second.pytorch), which can achieve the real-time processing (less than 1 ms / head)**
1. The inference part of **PointPillars**(pfe , backbone(multihead)) is optimized by tensorrt
2. The pre- and post- processing are optimized by CUDA / C + recode.

## Major Advance
- **Easy to train**
  
    - this repo directly uses [**mmlab/OpenPCdet**](https://github.com/open-mmlab/OpenPCDet) for training. Therefore, as long as you follow the steps of the [**official tutorial**](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md), it is very easy to train your own data. In addition, you can directly use [**official weight(PointPillar-MultiHead)**](https://drive.google.com/file/d/1p-501mTWsq0G9RzroTWSXreIMyTUUpBM/view?usp=sharing) for deployment. Due to the need to use **Tensorrt**, the official **mmlab/OpenPCdet** still needs to be customized. I will upload my modified version to [**hova88/OpenPCdet**](https://github.com/hova88/OpenPCDet).


- **Easy to deploy**
   
    - On the basis of [**Autoware.ai/core_perception/lidar_point_pillars**](https://github.com/Autoware-AI/core_perception/tree/master/lidar_point_pillars) and [**Apollo/modules/perception/lidar/lib/detection/lidar_point_pillars**](https://github.com/ApolloAuto/apollo/tree/master/modules/perception/lidar/lib/detection/lidar_point_pillars), this repo improves the way of information transmission, removes redundant things and adds **MultiHead** feature in postprocess.



## Requirements (My Environment)
### For *.onnx and *.trt engine file
* Linux Ubuntu 18.04
* OpenPCdet
* ONNX IR version:  0.0.6
* [onnx2trt](https://github.com/onnx/onnx-tensorrt)
  
### For algorithm: 
* Linux Ubuntu 18.04
* CMake 3.17 
* CUDA 10.2
* TensorRT 7.1.3 
* yaml-cpp
* google-test (not necessary)

### For visualization
* [open3d](https://github.com/intel-isl/Open3D)


## Usage

1. **clone thest two repositories, and make sure the dependences is complete**
   ```bash
   mkdir workspace && cd workspace
   git clone https://github.com/hova88/PointPillars_MultiHead_40FPS.git --recursive && cd ..
   git clone https://github.com/hova88/OpenPCDet.git 
   ```


2. **generate engine file**

    - 1.1 **Pytorch model --> ONNX model :** The specific conversion tutorial, i have put in the **change log** of [**hova88/OpenPCdet**]((https://github.com/hova88/OpenPCDet)).
        * [cbgs_pp_multihead_pfe.onnx](https://drive.google.com/file/d/1gQWtBZ4vfrSmv2nToSIarr-d7KkEWqxw/view?usp=sharing)
        * [cbgs_pp_multihead_backbone.onnx](https://drive.google.com/file/d/1dvUkjvhE0GEWvf6GchSGg8-lwukk7bTw/view?usp=sharing)

    - 1.2 **ONNX model --> TensorRT model :** after install the [onnx2trt](https://github.com/onnx/onnx-tensorrt), things become very simple. Note that if you want to further improve the the inference speed, you must use half precision or mixed precision(like ,-d 16)
        ```bash
            onnx2trt cbgs_pp_multihead_pfe.onnx -o cbgs_pp_multihead_pfe.trt -b 1 -d 16 
            onnx2trt cbgs_pp_multihead_backbone.onnx -o cbgs_pp_multihead_backbone.trt -b 1 -d 16 
        ```

    - 1.3 **engine file --> algorithm :** Specified the path of engine files(*.onnx , *.trt) in`bootstrap.yaml`.
  
    - 1.4 Download the test pointcloud [nuscenes_10sweeps_points.txt](https://drive.google.com/file/d/1KD0LT0kzcpGUysUu__dfnfYnHUW62iwN/view?usp=sharing), and specified the path in `bootstrap.yaml`.

3. **Compiler**

    ```bash
    cd PointPillars_MultiHead_40FPS
    mkdir build && cd build
    cmake .. && make -j8 && ./test/test_model
    ```

4. **Visualization**

    ```bash
    cd PointPillars_MultiHead_40FPS/tools
    python viewer.py
    ```
**Left figure shows the results of this repo, Right figure shows the official result of [**mmlab/OpenPCdet**](https://github.com/open-mmlab/OpenPCDet).**
<p align="left">
  <img width="2000" alt="fig_method" src=docs/demo.png>
</p>

## Result

### Use *.trt engine file  on NVIDIA GeForce RTX 3080 Ti 

**with the ScoreThreshold = 0.1**
```bash
 |￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣> 
 | ../model/cbgs_pp_multihead_pfe.trt >
 |＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿> 
             (\__/) ||                 
             (•ㅅ•) ||                 
             / 　 づ                                                         
                                                                  
 |￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣> 
 | ../model/cbgs_pp_multihead_backbone.trt >
 |＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿> 
             (\__/) ||                 ****
             (•ㅅ•) ||                 
             / 　 づ     
                                                                  
------------------------------------
Module        Time        
------------------------------------
Preprocess    0.571069 ms
Pfe           3.26203  ms
Scatter       0.384075 ms
Backbone      2.92882  ms
Postprocess   8.82032  ms
Summary       15.9707  ms
------------------------------------
```


**with the ScoreThreshold = 0.4**
```bash
 |￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣> 
 | ../model/cbgs_pp_multihead_pfe.trt >
 |＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿> 
             (\__/) ||                 
             (•ㅅ•) ||                 
             / 　 づ                                                         
                                                                  
 |￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣> 
 | ../model/cbgs_pp_multihead_backbone.trt >
 |＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿> 
             (\__/) ||                 ****
             (•ㅅ•) ||                 
             / 　 づ     
                                                                  
------------------------------------
Module        Time        
------------------------------------
Preprocess    0.337111 ms
Pfe           2.81834  ms
Scatter       0.161953 ms
Backbone      3.64112  ms
Postprocess   4.34731  ms
Summary       11.3101  ms
------------------------------------
```

### Runtime logs
- 1. [ScoreThreshold = 0.1](runtime_log_0.1.txt)
- 2. [ScoreThreshold = 0.2](runtime_log_0.2.txt)
- 3. [ScoreThreshold = 0.3](runtime_log_0.3.txt)
- 4. [ScoreThreshold = 0.4](runtime_log_0.4.txt)
# License

GNU General Public License v3.0 or later
See [`COPYING`](LICENSE.md) to see the full text.
