// headers in STL
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
// headers in 3rd-part
#include "../pointpillars/pointpillars.h"
#include "gtest/gtest.h"
using namespace std;

int Txt2Arrary( float* &points_array , string file_name , int num_feature = 4)
{
  ifstream InFile;
  InFile.open(file_name.data());
  assert(InFile.is_open());

  vector<float> temp_points;
  string c;

  while (!InFile.eof())
  {
      InFile >> c;

      temp_points.push_back(atof(c.c_str()));
  }
  points_array = new float[temp_points.size()];
  for (int i = 0 ; i < temp_points.size() ; ++i) {
    points_array[i] = temp_points[i];
  }

  InFile.close();  
  return temp_points.size() / num_feature;
  // printf("Done");
};

void Boxes2Txt( std::vector<float> boxes , string file_name , int num_feature = 7)
{
    ofstream ofFile;
    ofFile.open(file_name , std::ios::out );  
    if (ofFile.is_open()) {
        for (int i = 0 ; i < boxes.size() / num_feature ; ++i) {
            for (int j = 0 ; j < num_feature ; ++j) {
                ofFile << boxes.at(i * num_feature + j) << " ";
            }
            ofFile << "\n";
        }
    }
    ofFile.close();
    return ;
};

TEST(PointPillars, __points_array__) {

  std::string file_name ="../test/testdata/raw_cloud.txt";
  float* points_array;
  int in_num_points;
  in_num_points = Txt2Arrary(points_array,file_name,5);
  EXPECT_FLOAT_EQ(63624, in_num_points);
  EXPECT_FLOAT_EQ(-0.04419935122132301, points_array[0]);
  EXPECT_FLOAT_EQ(-0.3708479702472687, points_array[1]);
  EXPECT_FLOAT_EQ(1.5506199598312378, points_array[2]);
  EXPECT_FLOAT_EQ(0.0, points_array[3]);
  EXPECT_FLOAT_EQ(0.0, points_array[4]);
};

TEST(PointPillars, __build_model__) {

  const std::string DB_CONF = "../bootstrap.yaml";
  YAML::Node config = YAML::LoadFile(DB_CONF);

  std::string pfe_file,backbone_file; 
  if(config["UseOnnx"].as<bool>()) {
    pfe_file = config["PfeOnnx"].as<std::string>();
    backbone_file = config["BackboneOnnx"].as<std::string>();
  }else {
    pfe_file = config["PfeTrt"].as<std::string>();
    backbone_file = config["BackboneTrt"].as<std::string>();
  }
  std::cout << backbone_file << std::endl;
  const std::string pp_config = config["ModelConfig"].as<std::string>();
  PointPillars pp(
    config["ScoreThreshold"].as<float>(),
    config["NmsOverlapThreshold"].as<float>(),
    config["UseOnnx"].as<bool>(),
    pfe_file,
    backbone_file,
    pp_config
  );
  std::string file_name = config["InputFile"].as<std::string>();
  float* points_array;
  int in_num_points;
  in_num_points = Txt2Arrary(points_array,file_name,5);
 
  std::vector<float> out_detections;
  std::vector<int> out_labels;
  std::vector<float> out_scores;
  
  cudaDeviceSynchronize();
  pp.DoInference(points_array, in_num_points, &out_detections, &out_labels , &out_scores);
  cudaDeviceSynchronize();
  int BoxFeature = 7;
  int num_objects = out_detections.size() / BoxFeature;

  std::string boxes_file_name = config["OutputFile"].as<std::string>();
  Boxes2Txt(out_detections , boxes_file_name );
  EXPECT_EQ(num_objects,59);

};
