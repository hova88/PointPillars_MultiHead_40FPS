"""
author: hova88
date: 2021/03/16
"""
import numpy as np 
import numpy as np
from visual_tools import draw_clouds_with_boxes
import open3d as o3d
import yaml

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


def dataloader(cloud_path , boxes_path):
    cloud = np.loadtxt(cloud_path).reshape(-1,5)
    boxes = np.loadtxt(boxes_path).reshape(-1,7)
    return cloud , boxes 

if __name__ == "__main__":
    import yaml
    with open("../bootstrap.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cloud ,boxes = dataloader(config['InputFile'], config['OutputFile'])
    draw_clouds_with_boxes(cloud ,boxes)