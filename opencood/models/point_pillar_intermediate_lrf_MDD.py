# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.mdd_modules.radar_cond_diff_denoise import Cond_Diff_Denoise

def vis_feature_denoise(lidar_batch_dict):
    channel_image = np.sum(lidar_batch_dict['de_spatial_features'].detach().cpu().numpy(), axis=(0,1))[:,352:552]
    channel_image /= channel_image.max()
    channel_image *= 255
    channel_image = np.clip(channel_image, 0, 255).astype("uint8")
    plt.imsave(f'/mnt/16THDD/hx/OpenCOOD/vis/gt.png', channel_image, cmap='jet')

    channel_image = np.sum(lidar_batch_dict['spatial_features'].detach().cpu().numpy(), axis=(0,1))[:,352:552]
    channel_image /= channel_image.max()
    channel_image *= 255
    channel_image = np.clip(channel_image, 0, 255).astype("uint8")
    plt.imsave(f'/mnt/16THDD/hx/OpenCOOD/vis/denoise.png', channel_image, cmap='jet')

    channel_image = np.sum(lidar_batch_dict['bf_spatial_features'].detach().cpu().numpy(), axis=(0,1))[:,352:552]
    channel_image /= channel_image.max()
    channel_image *= 255
    channel_image = np.clip(channel_image, 0, 255).astype("uint8")
    plt.imsave(f'/mnt/16THDD/hx/OpenCOOD/vis/raw.png', channel_image, cmap='jet')

    channel_image = np.sum(lidar_batch_dict['ra_spatial_features'].detach().cpu().numpy(), axis=(0,1))[:,352:552]
    channel_image /= channel_image.max()
    channel_image *= 255
    channel_image = np.clip(channel_image, 0, 255).astype("uint8")
    plt.imsave(f'/mnt/16THDD/hx/OpenCOOD/vis/radar.png', channel_image, cmap='jet')

class PointPillarIntermediateLRFMDD(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediateLRFMDD, self).__init__()

        # PIllar VFE
        
        self.lidar_pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.mdd = Cond_Diff_Denoise(args['mdd_block'], 32)
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.radar_pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 128)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)
        

    def forward(self, data_dict):
        
        lidar_voxel_features = data_dict['processed_lidar']['voxel_features']
        lidar_voxel_coords = data_dict['processed_lidar']['voxel_coords']
        lidar_voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        lidar_batch_dict = {'voxel_features': lidar_voxel_features,
                      'voxel_coords': lidar_voxel_coords,
                      'voxel_num_points': lidar_voxel_num_points,
                      'record_len': record_len
                      }

        if data_dict['train']:
            de_lidar_voxel_features = data_dict['processed_de_lidar']['voxel_features']
            de_lidar_voxel_coords = data_dict['processed_de_lidar']['voxel_coords']
            de_lidar_voxel_num_points = data_dict['processed_de_lidar']['voxel_num_points']
            record_len = data_dict['record_len']

            de_lidar_batch_dict = {'voxel_features': de_lidar_voxel_features,
                        'voxel_coords': de_lidar_voxel_coords,
                        'voxel_num_points': de_lidar_voxel_num_points,
                        'record_len': record_len
                        }
            with torch.no_grad():
                de_lidar_batch_dict = self.lidar_pillar_vfe(de_lidar_batch_dict)
                de_lidar_batch_dict = self.scatter(de_lidar_batch_dict)
            lidar_batch_dict['de_spatial_features'] = de_lidar_batch_dict['spatial_features']


        radar_voxel_features = data_dict['processed_radar']['voxel_features']
        radar_voxel_coords = data_dict['processed_radar']['voxel_coords']
        radar_voxel_num_points = data_dict['processed_radar']['voxel_num_points']
        record_len = data_dict['record_len']

        radar_batch_dict = {'voxel_features': radar_voxel_features,
                      'voxel_coords': radar_voxel_coords,
                      'voxel_num_points': radar_voxel_num_points,
                      'record_len': record_len}
        

        radar_batch_dict = self.radar_pillar_vfe(radar_batch_dict)
        radar_batch_dict = self.scatter(radar_batch_dict)

        lidar_batch_dict = self.lidar_pillar_vfe(lidar_batch_dict)
        lidar_batch_dict = self.scatter(lidar_batch_dict)
        
        lidar_batch_dict['ra_spatial_features'] = radar_batch_dict['spatial_features']
        lidar_batch_dict['bf_spatial_features'] = lidar_batch_dict['spatial_features']
        lidar_batch_dict['train'] = data_dict['train']
        
   
        lidar_batch_dict = self.mdd(lidar_batch_dict)
        lidar_batch_dict['spatial_features'] = lidar_batch_dict['pred_feature'] * (lidar_batch_dict['spatial_features'] != 0)
        #vis_feature_denoise(lidar_batch_dict)
        output_dict = {'pred_feature' : lidar_batch_dict['spatial_features']}
        
        if data_dict['train']:
            output_dict.update({'gt_feature' : lidar_batch_dict['de_spatial_features']})
        # return output_dict
        # output_dict = {}
        batch_dict={
            'spatial_features' : \
            torch.cat([lidar_batch_dict['spatial_features'], radar_batch_dict['spatial_features']],dim = 1),
            'record_len': record_len
        } 

        batch_dict = self.backbone(batch_dict)
        
        spatial_features_2d = batch_dict['spatial_features_2d']
        

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)    
        output_dict.update({'psm': psm,
                       'rm': rm,})
        return output_dict