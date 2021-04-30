
/**
* @author Yan haixu
* Contact: just github.com/hova88
* @date 2021/04/30
*/

class NmsCuda {
 private:
  const int num_threads_;
  const int num_box_corners_;
  const float nms_overlap_threshold_;

 public:
  /**
   * @brief Constructor
   * @param[in] num_threads Number of threads when launching cuda kernel
   * @param[in] num_box_corners Number of corners for 2D box
   * @param[in] nms_overlap_threshold IOU threshold for NMS
   * @details Captital variables never change after the compile, Non-captital
   * variables could be chaned through rosparam
   */
  NmsCuda(const int num_threads, const int num_box_corners,
          const float nms_overlap_threshold);

  /**
   * @brief GPU Non-Maximum Suppresion for network output
   * @param[in] host_filter_count Number of filtered output
   * @param[in] dev_sorted_box_for_nms Bounding box output sorted by score
   * @param[out] out_keep_inds Indexes of selected bounding box
   * @param[out] out_num_to_keep Number of kept bounding boxes
   * @details NMS in GPU and postprocessing for selecting box in CPU
   */
  void DoNmsCuda(const int host_filter_count, float* dev_sorted_box_for_nms,
                 long* out_keep_inds, int* out_num_to_keep);
};