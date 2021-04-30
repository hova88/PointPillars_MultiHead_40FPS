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

// headers in STL
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
// headers in CUDA
#include "cuda_runtime_api.h"

using namespace std;
// using MACRO to allocate memory inside CUDA kernel
#define NUM_3D_BOX_CORNERS_MACRO 8

#define NUM_2D_BOX_CORNERS_MACRO 4

#define NUM_THREADS_MACRO 64

// need to be changed when num_threads_ is changed

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define GPU_CHECK(ans)                    \
  {                                       \
    GPUAssert((ans), __FILE__, __LINE__); \
  }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
};

template <typename T>
void HOST_SAVE(T *array, int size, string filename, string root = "../test/result", string postfix = ".txt")
{
  string filepath = root + "/" + filename + postfix;
  if (postfix == ".bin")
  {
    fstream file(filepath, ios::out | ios::binary);
    file.write(reinterpret_cast<char *>(array), sizeof(size * sizeof(T)));
    file.close();
    std::cout << "|>>>|  Data has been written in " << filepath << "  |<<<|" << std::endl;
    return;
  }
  else if (postfix == ".txt")
  {
    ofstream file(filepath, ios::out);
    for (int i = 0; i < size; ++i)
      file << array[i] << " ";
    file.close();
    std::cout << "|>>>|  Data has been written in " << filepath << "  |<<<|" << std::endl;
    return;
  }
};

template <typename T>
void DEVICE_SAVE(T *array, int size, string filename, string root = "../test/result", string postfix = ".txt")
{
  T *temp_ = new T[size];
  cudaMemcpy(temp_, array, size * sizeof(T), cudaMemcpyDeviceToHost);
  HOST_SAVE<T>(temp_, size, filename, root, postfix);
  delete[] temp_;
};


// int TXTtoArrary( float* &points_array , string file_name , int num_feature = 4)
// {
//   ifstream InFile;
//   InFile.open(file_name.data());
//   assert(InFile.is_open());

//   vector<float> temp_points;
//   string c;

//   while (!InFile.eof())
//   {
//       InFile >> c;

//       temp_points.push_back(atof(c.c_str()));
//   }
//   points_array = new float[temp_points.size()];
//   for (int i = 0 ; i < temp_points.size() ; ++i) {
//     points_array[i] = temp_points[i];
//   }

//   InFile.close();  
//   return temp_points.size() / num_feature;
// };
