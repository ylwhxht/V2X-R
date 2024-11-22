# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch.nn as nn

from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.spatialcooper_modules.spatial_cooper_fuse import SpatialFusion
import torch



class PointPillarSpatialCooperLRF(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarSpatialCooperLRF, self).__init__()

        # self.modality = 'processed_lidar'
        # if 'use_modality' in args.keys():
        #     self.modality = args['use_modality']

        # self.max_cav = args['max_cav']
        # # PIllar VFE
        # self.pillar_vfe = PillarVFE(args['pillar_vfe'],
        #                             num_point_features=4,
        #                             voxel_size=args['voxel_size'],
        #                             point_cloud_range=args['lidar_range'])
        # self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.lidar_pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.radar_pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 128)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = SpatialFusion()

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        
        # voxel_features = data_dict[self.modality]['voxel_features']
        # voxel_coords = data_dict[self.modality]['voxel_coords']
        # voxel_num_points = data_dict[self.modality]['voxel_num_points']
        # # voxel_features = data_dict['processed_lidar']['voxel_features']
        # # voxel_coords = data_dict['processed_lidar']['voxel_coords']
        # # voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        # record_len = data_dict['record_len']

        # batch_dict = {'voxel_features': voxel_features,
        #               'voxel_coords': voxel_coords,
        #               'voxel_num_points': voxel_num_points,
        #               'record_len': record_len}
        # # n, 4 -> n, c
        # batch_dict = self.pillar_vfe(batch_dict)
        # # n, c -> N, C, H, W
        # batch_dict = self.scatter(batch_dict)
        # batch_dict = self.backbone(batch_dict)
        lidar_voxel_features = data_dict['processed_lidar']['voxel_features']
        lidar_voxel_coords = data_dict['processed_lidar']['voxel_coords']
        lidar_voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        lidar_batch_dict = {'voxel_features': lidar_voxel_features,
                      'voxel_coords': lidar_voxel_coords,
                      'voxel_num_points': lidar_voxel_num_points,
                      'record_len': record_len}

        radar_voxel_features = data_dict['processed_radar']['voxel_features']
        radar_voxel_coords = data_dict['processed_radar']['voxel_coords']
        radar_voxel_num_points = data_dict['processed_radar']['voxel_num_points']
        record_len = data_dict['record_len']

        radar_batch_dict = {'voxel_features': radar_voxel_features,
                      'voxel_coords': radar_voxel_coords,
                      'voxel_num_points': radar_voxel_num_points,
                      'record_len': record_len}

        lidar_batch_dict = self.lidar_pillar_vfe(lidar_batch_dict)
        lidar_batch_dict = self.scatter(lidar_batch_dict)

        radar_batch_dict = self.radar_pillar_vfe(radar_batch_dict)
        radar_batch_dict = self.scatter(radar_batch_dict)

        batch_dict={
            'spatial_features' : \
            torch.cat([lidar_batch_dict['spatial_features'], radar_batch_dict['spatial_features']],dim = 1),
            'record_len': record_len
        }

        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict