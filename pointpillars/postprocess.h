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

#pragma once
#include <memory>
#include <vector>
#include "nms.h"

class PostprocessCuda {
 private:
    // initializer list

    const int num_threads_;
    const float float_min_;
    const float float_max_;
    const int num_class_;
    const int num_anchor_per_cls_;
    const float score_threshold_;
    const float nms_overlap_threshold_;
    const int nms_pre_maxsize_;
    const int nms_post_maxsize_;
    const int num_box_corners_;
    const int num_input_box_feature_;
    const int num_output_box_feature_;
    const std::vector<std::vector<int>> multihead_label_mapping_;
    // end initializer list

    std::unique_ptr<NmsCuda> nms_cuda_ptr_;
  public:
  /**
   * @brief Constructor
   * @param[in] num_threads Number of threads when launching cuda kernel
   * @param[in] float_min The lowest float value
   * @param[in] float_max The maximum float value
   * @param[in] num_class Number of classes 
   * @param[in] num_anchor_per_cls Number anchor per category
   * @param[in] multihead_label_mapping 
   * @param[in] score_threshold Score threshold for filtering output
   * @param[in] nms_overlap_threshold IOU threshold for NMS
   * @param[in] nms_pre_maxsize Maximum number of boxes into NMS
   * @param[in] nms_post_maxsize Maximum number of boxes after NMS
   * @param[in] num_box_corners Number of box's corner
   * @param[in] num_output_box_feature Number of output box's feature
   * @details Captital variables never change after the compile, non-capital
   * variables could be changed through rosparam
   */
  PostprocessCuda(const int num_threads, 
                  const float float_min, const float float_max,
                  const int num_class, const int num_anchor_per_cls, 
                  const std::vector<std::vector<int>> multihead_label_mapping,
                  const float score_threshold,  
                  const float nms_overlap_threshold, 
                  const int nms_pre_maxsize, 
                  const int nms_post_maxsize,
                  const int num_box_corners,
                  const int num_input_box_feature, 
                  const int num_output_box_feature);
  ~PostprocessCuda(){}

  /**
   * @brief Postprocessing for the network output
   * @param[in] rpn_box_output Box predictions from the network output
   * @param[in] rpn_cls_output Class predictions from the network output
   * @param[in] rpn_dir_output Direction predictions from the network output
   * @param[in] dev_filtered_box Filtered box predictions
   * @param[in] dev_filtered_score Filtered score predictions
   * @param[in] dev_filter_count The number of filtered output
   * @param[out] out_detection Output bounding boxes
   * @param[out] out_label Output labels of objects
   * @details dev_* represents device memory allocated variables
   */
  void DoPostprocessCuda(
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
    std::vector<float>& out_detection, std::vector<int>& out_label , std::vector<float>& out_score);
};