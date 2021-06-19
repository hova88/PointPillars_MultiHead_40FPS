/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @author Kosuke Murakami
 * @date 2019/02/26
 */

/**
* @author Yan haixu
* Contact: just github.com/hova88
* @date 2021/04/30
*/


// headers in CUDA
#include <thrust/sort.h>

// headers in local files
#include "common.h"
#include "postprocess.h"
#include <stdio.h>


// sigmoid_filter_warp
__device__ void box_decode_warp(int head_offset, const float* box_pred, 
    int tid , int num_anchors_per_head , int counter, float* filtered_box) 
{
    filtered_box[blockIdx.z * num_anchors_per_head * 7  + counter * 7 + 0] = box_pred[ head_offset + tid * 9 + 0];
    filtered_box[blockIdx.z * num_anchors_per_head * 7  + counter * 7 + 1] = box_pred[ head_offset + tid * 9 + 1];
    filtered_box[blockIdx.z * num_anchors_per_head * 7  + counter * 7 + 2] = box_pred[ head_offset + tid * 9 + 2];
    filtered_box[blockIdx.z * num_anchors_per_head * 7  + counter * 7 + 3] = box_pred[ head_offset + tid * 9 + 3];
    filtered_box[blockIdx.z * num_anchors_per_head * 7  + counter * 7 + 4] = box_pred[ head_offset + tid * 9 + 4];
    filtered_box[blockIdx.z * num_anchors_per_head * 7  + counter * 7 + 5] = box_pred[ head_offset + tid * 9 + 5];
    filtered_box[blockIdx.z * num_anchors_per_head * 7  + counter * 7 + 6] = box_pred[ head_offset + tid * 9 + 6];
}


__global__ void sigmoid_filter_kernel(

    float* cls_pred_0,
    float* cls_pred_12,
    float* cls_pred_34,
    float* cls_pred_5,
    float* cls_pred_67,
    float* cls_pred_89,

    const float* box_pred_0,

    const float* box_pred_1,
    const float* box_pred_2,

    const float* box_pred_3,
    const float* box_pred_4,

    const float* box_pred_5,

    const float* box_pred_6,
    const float* box_pred_7,

    const float* box_pred_8,
    const float* box_pred_9,

    float* filtered_box, 
    float* filtered_score, 
    int* filter_count,

    const float score_threshold) {   

    // cls_pred_34 
    // 32768*2 , 2

    int num_anchors_per_head = gridDim.x * gridDim.y * blockDim.x;
    // 16 * 4 * 512 = 32768
    extern __shared__ float cls_score[];
    cls_score[threadIdx.x + blockDim.x] = -1.0f;

    int tid = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y *  blockDim.x + threadIdx.x; 


    if ( blockIdx.z == 0) cls_score[ threadIdx.x ] = 1 / (1 + expf(-cls_pred_0[ tid ]));
    if ( blockIdx.z == 1) {
        cls_score[ threadIdx.x ] = 1 / (1 + expf(-cls_pred_12[ tid * 2 ]));
        cls_score[ threadIdx.x + blockDim.x] = 1 / (1 + expf(-cls_pred_12[ (num_anchors_per_head + tid) * 2]));}
    if ( blockIdx.z == 2) {
        cls_score[ threadIdx.x ] = 1 / (1 + expf(-cls_pred_12[ tid * 2 + 1]));
        cls_score[ threadIdx.x + blockDim.x] = 1 / (1 + expf(-cls_pred_12[ (num_anchors_per_head + tid) * 2 + 1]));}

    if ( blockIdx.z == 3) {
        cls_score[ threadIdx.x ] = 1 / (1 + expf(-cls_pred_34[ tid * 2 ]));
        cls_score[ threadIdx.x + blockDim.x] = 1 / (1 + expf(-cls_pred_34[ (num_anchors_per_head + tid) * 2]));}
    if ( blockIdx.z == 4) {
        cls_score[ threadIdx.x ] = 1 / (1 + expf(-cls_pred_34[ tid * 2 + 1 ]));
        cls_score[ threadIdx.x + blockDim.x] = 1 / (1 + expf(-cls_pred_34[ (num_anchors_per_head + tid) * 2 + 1]));}

    if ( blockIdx.z == 5) cls_score[ threadIdx.x ] = 1 / (1 + expf(-cls_pred_5[ tid ]));

    if ( blockIdx.z == 6) {
        cls_score[ threadIdx.x ] = 1 / (1 + expf(-cls_pred_67[ tid * 2 ]));
        cls_score[ threadIdx.x + blockDim.x] = 1 / (1 + expf(-cls_pred_67[ (num_anchors_per_head + tid) * 2]));}
    if ( blockIdx.z == 7) {
        cls_score[ threadIdx.x ] = 1 / (1 + expf(-cls_pred_67[ tid * 2 + 1 ]));
        cls_score[ threadIdx.x + blockDim.x] = 1 / (1 + expf(-cls_pred_67[ (num_anchors_per_head + tid) * 2 + 1]));}

    if ( blockIdx.z == 8) {
        cls_score[ threadIdx.x ] = 1 / (1 + expf(-cls_pred_89[ tid * 2 ]));
        cls_score[ threadIdx.x + blockDim.x] = 1 / (1 + expf(-cls_pred_89[ (num_anchors_per_head + tid) * 2]));}
    if ( blockIdx.z == 9) {
        cls_score[ threadIdx.x ] = 1 / (1 + expf(-cls_pred_89[ tid * 2 + 1 ]));
        cls_score[ threadIdx.x + blockDim.x] = 1 / (1 + expf(-cls_pred_89[ (num_anchors_per_head + tid) * 2 + 1]));}
    
    __syncthreads();
    
    if( cls_score[ threadIdx.x ] > score_threshold) 
    {
        int counter = atomicAdd(&filter_count[blockIdx.z], 1);
        if ( blockIdx.z == 0) {
            box_decode_warp(0 ,box_pred_0 , tid , num_anchors_per_head , counter , filtered_box);
            filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
        }else
        if ( blockIdx.z == 1) {
            box_decode_warp(0 ,box_pred_1 , tid , num_anchors_per_head , counter , filtered_box);
            filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
        }else
        if ( blockIdx.z == 2) {
            box_decode_warp(0 ,box_pred_1 , tid , num_anchors_per_head , counter , filtered_box);
            filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
        }else
        if ( blockIdx.z == 3) {
            box_decode_warp(0 ,box_pred_3 , tid , num_anchors_per_head , counter , filtered_box);
            filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
        }else 
        if (blockIdx.z == 4) {
            box_decode_warp(0 ,box_pred_3 , tid , num_anchors_per_head , counter , filtered_box);
            filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];            
        }else
        if ( blockIdx.z == 5) {
            box_decode_warp(0 ,box_pred_5 , tid , num_anchors_per_head , counter , filtered_box);
            filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
        }else
        if ( blockIdx.z == 6) {
            box_decode_warp(0 ,box_pred_6 , tid , num_anchors_per_head , counter , filtered_box);
            filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
        }else
        if ( blockIdx.z == 7) {
            box_decode_warp(0 ,box_pred_6 , tid , num_anchors_per_head , counter , filtered_box);
            filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
        }else
        if ( blockIdx.z == 8) {

            box_decode_warp(0 ,box_pred_8 , tid , num_anchors_per_head , counter , filtered_box);
            filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
        }else
        if ( blockIdx.z == 9) {
            box_decode_warp(0 ,box_pred_8 , tid , num_anchors_per_head , counter , filtered_box);
            filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
        }
    }
    __syncthreads();  
    if( cls_score[ threadIdx.x + blockDim.x ] > score_threshold)  {     
            int counter = atomicAdd(&filter_count[blockIdx.z], 1);
            // printf("counter : %d \n" , counter);
            if (blockIdx.z == 1) {
                box_decode_warp(0 ,box_pred_2 , tid , num_anchors_per_head , counter , filtered_box);
                filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
            }else 
            if (blockIdx.z == 2) {
                box_decode_warp(0 ,box_pred_2 , tid , num_anchors_per_head , counter , filtered_box);
                filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
            }else 
            if (blockIdx.z == 3) {
                box_decode_warp(0 ,box_pred_4 , tid , num_anchors_per_head , counter , filtered_box);
                filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
            }else 
            if (blockIdx.z == 4) {
                box_decode_warp(0 ,box_pred_4 , tid , num_anchors_per_head , counter , filtered_box);
                filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
            }else 
            if (blockIdx.z == 6) {
                box_decode_warp(0 ,box_pred_7 , tid , num_anchors_per_head , counter , filtered_box);
                filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
            }else 
            if (blockIdx.z == 7) {
                box_decode_warp(0 ,box_pred_7 , tid , num_anchors_per_head , counter , filtered_box);
                filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
            }else 
            if (blockIdx.z == 8) {
                box_decode_warp(0 ,box_pred_9 , tid , num_anchors_per_head , counter , filtered_box);
                filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
            }else 
            if (blockIdx.z == 9) {
                box_decode_warp(0 ,box_pred_9 , tid , num_anchors_per_head , counter , filtered_box);
                filtered_score[blockIdx.z * num_anchors_per_head + counter] = cls_score[ threadIdx.x ];
            }
    }
}

__global__ void sort_boxes_by_indexes_kernel(float* filtered_box, float* filtered_scores, int* indexes, int filter_count,
    float* sorted_filtered_boxes, float* sorted_filtered_scores,
    const int num_output_box_feature)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < filter_count)  {

        int sort_index = indexes[tid];
        sorted_filtered_boxes[tid * num_output_box_feature + 0] = filtered_box[sort_index * num_output_box_feature + 0];
        sorted_filtered_boxes[tid * num_output_box_feature + 1] = filtered_box[sort_index * num_output_box_feature + 1];
        sorted_filtered_boxes[tid * num_output_box_feature + 2] = filtered_box[sort_index * num_output_box_feature + 2];
        sorted_filtered_boxes[tid * num_output_box_feature + 3] = filtered_box[sort_index * num_output_box_feature + 3];
        sorted_filtered_boxes[tid * num_output_box_feature + 4] = filtered_box[sort_index * num_output_box_feature + 4];
        sorted_filtered_boxes[tid * num_output_box_feature + 5] = filtered_box[sort_index * num_output_box_feature + 5];
        sorted_filtered_boxes[tid * num_output_box_feature + 6] = filtered_box[sort_index * num_output_box_feature + 6];

        // sorted_filtered_dir[tid] = filtered_dir[sort_index];
        sorted_filtered_scores[tid] = filtered_scores[sort_index];
    }
}



PostprocessCuda::PostprocessCuda(const int num_threads, const float float_min, const float float_max,
    const int num_class,const int num_anchor_per_cls,
    const std::vector<std::vector<int>> multihead_label_mapping,
    const float score_threshold,  const float nms_overlap_threshold, 
    const int nms_pre_maxsize, const int nms_post_maxsize,
    const int num_box_corners, 
    const int num_input_box_feature,
    const int num_output_box_feature)
: num_threads_(num_threads),
  float_min_(float_min),
  float_max_(float_max),
  num_class_(num_class),
  num_anchor_per_cls_(num_anchor_per_cls),
  multihead_label_mapping_(multihead_label_mapping),
  score_threshold_(score_threshold),
  nms_overlap_threshold_(nms_overlap_threshold),
  nms_pre_maxsize_(nms_pre_maxsize),
  nms_post_maxsize_(nms_post_maxsize),
  num_box_corners_(num_box_corners),
  num_input_box_feature_(num_input_box_feature),
  num_output_box_feature_(num_output_box_feature) {
    nms_cuda_ptr_.reset(
    new NmsCuda(num_threads_, num_box_corners_, nms_overlap_threshold_));

}


void PostprocessCuda::DoPostprocessCuda(
    float* cls_pred_0,
    float* cls_pred_12,
    float* cls_pred_34,
    float* cls_pred_5,
    float* cls_pred_67,
    float* cls_pred_89,

    const float* box_preds,
   
    float* dev_filtered_box, 
    float* dev_filtered_score, 
    int* dev_filter_count,
    std::vector<float>& out_detection, std::vector<int>& out_label , std::vector<float>& out_score) {
    // 在此之前，先进行rpn_box_output的concat. 
    // 128x128 的feature map， cls_pred 的shape为（32768，1），（32768,1），（32768,1），（65536,2），（32768，1）
    dim3 gridsize(16, 4 , 10);  //16 *  4  * 512  = 32768 代表一个head的anchors
    sigmoid_filter_kernel<<< gridsize, 512 , 512 * 2 * sizeof(float)>>>(
        cls_pred_0,
        cls_pred_12, 
        cls_pred_34, 
        cls_pred_5, 
        cls_pred_67, 
        cls_pred_89,

        &box_preds[0 * 32768 * 9],
        &box_preds[1 * 32768 * 9],
        &box_preds[2 * 32768 * 9],
        &box_preds[3 * 32768 * 9],
        &box_preds[4 * 32768 * 9],
        &box_preds[5 * 32768 * 9],
        &box_preds[6 * 32768 * 9],
        &box_preds[7 * 32768 * 9],
        &box_preds[8 * 32768 * 9],
        &box_preds[9 * 32768 * 9],

        dev_filtered_box, 
        dev_filtered_score,  
        dev_filter_count, 
    
        score_threshold_);
    cudaDeviceSynchronize();
    
    int host_filter_count[num_class_] = {0};
    GPU_CHECK(cudaMemcpy(host_filter_count, dev_filter_count, num_class_ * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < num_class_; ++ i) {
        if(host_filter_count[i] <= 0) continue;

        int* dev_indexes;
        float* dev_sorted_filtered_box;
        float* dev_sorted_filtered_scores;
        GPU_CHECK(cudaMalloc((void**)&dev_indexes, host_filter_count[i] * sizeof(int)));
        GPU_CHECK(cudaMalloc((void**)&dev_sorted_filtered_box, host_filter_count[i] * num_output_box_feature_ * sizeof(float)));
        GPU_CHECK(cudaMalloc((void**)&dev_sorted_filtered_scores, host_filter_count[i]*sizeof(float)));
        // GPU_CHECK(cudaMalloc((void**)&dev_sorted_box_for_nms, NUM_BOX_CORNERS_*host_filter_count[i]*sizeof(float)));
        thrust::sequence(thrust::device, dev_indexes, dev_indexes + host_filter_count[i]);
        thrust::sort_by_key(thrust::device, 
                            &dev_filtered_score[i * num_anchor_per_cls_], 
                            &dev_filtered_score[i * num_anchor_per_cls_ + host_filter_count[i]],
                            dev_indexes, 
                            thrust::greater<float>());

        const int num_blocks = DIVUP(host_filter_count[i], num_threads_);

        sort_boxes_by_indexes_kernel<<<num_blocks, num_threads_>>>(
            &dev_filtered_box[i * num_anchor_per_cls_ * num_output_box_feature_], 
            &dev_filtered_score[i * num_anchor_per_cls_], 
            dev_indexes, 
            host_filter_count[i],
            dev_sorted_filtered_box, 
            dev_sorted_filtered_scores,
            num_output_box_feature_);

        int num_box_for_nms = min(nms_pre_maxsize_, host_filter_count[i]);
        long* keep_inds = new long[num_box_for_nms];  // index of kept box
        memset(keep_inds, 0, num_box_for_nms * sizeof(int));
        int num_out = 0;
        nms_cuda_ptr_->DoNmsCuda(num_box_for_nms, dev_sorted_filtered_box, keep_inds, &num_out);

        num_out = min(num_out, nms_post_maxsize_);

        float* host_filtered_box = new float[host_filter_count[i] * num_output_box_feature_]();
        float* host_filtered_scores = new float[host_filter_count[i]]();


        cudaMemcpy(host_filtered_box, dev_sorted_filtered_box, host_filter_count[i] * num_output_box_feature_ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_filtered_scores, dev_sorted_filtered_scores, host_filter_count[i] * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_out; ++i)  {
            out_detection.emplace_back(host_filtered_box[keep_inds[i] * num_output_box_feature_ + 0]);
            out_detection.emplace_back(host_filtered_box[keep_inds[i] * num_output_box_feature_ + 1]);
            out_detection.emplace_back(host_filtered_box[keep_inds[i] * num_output_box_feature_ + 2]);
            out_detection.emplace_back(host_filtered_box[keep_inds[i] * num_output_box_feature_ + 3]);
            out_detection.emplace_back(host_filtered_box[keep_inds[i] * num_output_box_feature_ + 4]);
            out_detection.emplace_back(host_filtered_box[keep_inds[i] * num_output_box_feature_ + 5]);
            out_detection.emplace_back(host_filtered_box[keep_inds[i] * num_output_box_feature_ + 6]);
            out_score.emplace_back(host_filtered_scores[keep_inds[i]]);
            out_label.emplace_back(i);

        }
        delete[] keep_inds;
        delete[] host_filtered_scores;
        delete[] host_filtered_box;

        GPU_CHECK(cudaFree(dev_indexes));
        GPU_CHECK(cudaFree(dev_sorted_filtered_box));
    }
}
