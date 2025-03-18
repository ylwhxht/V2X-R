import torch.nn as nn
import torch
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.where2comm_modules.where2comm_fuse import Where2comm
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter

class PointPillarWhere2commLRF(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2commLRF, self).__init__()
        # self.modality = 'processed_lidar'
        # if 'use_modality' in args.keys():
        #     self.modality = args['use_modality']
        self.max_cav = args['max_cav']
        # Pillar VFE
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

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.fusion_net = Where2comm(args['where2comm_fusion'])
        self.multi_scale = args['where2comm_fusion']['multi_scale']

        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
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
        lidar_voxel_features = data_dict['processed_lidar']['voxel_features']
        lidar_voxel_coords = data_dict['processed_lidar']['voxel_coords']
        lidar_voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        lidar_batch_dict = {'voxel_features': lidar_voxel_features,
                      'voxel_coords': lidar_voxel_coords,
                      'voxel_num_points': lidar_voxel_num_points,
                      'record_len': record_len
                      }


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

        batch_dict={
            'spatial_features' : \
            torch.cat([lidar_batch_dict['spatial_features'], radar_batch_dict['spatial_features']],dim = 1),
            'record_len': record_len
        }
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        comm_rates = batch_dict['spatial_features'].count_nonzero().item()
        batch_dict = self.backbone(batch_dict)

        # N, C, H', W': [N, 256, 48, 176]
        spatial_features_2d = batch_dict['spatial_features_2d']
        # Down-sample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        psm_single = self.cls_head(spatial_features_2d)

        # Compressor
        if self.compression:
            # The ego feature is also compressed
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        if self.multi_scale:
            # Bypass communication cost, communicate at high resolution, neither shrink nor compress
            fused_feature, communication_rates = self.fusion_net(batch_dict['spatial_features'],
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix,
                                                                 self.backbone)
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates = self.fusion_net(spatial_features_2d,
                                                                 psm_single,
                                                                 record_len,
                                                                 pairwise_t_matrix)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        output_dict = {}
        output_dict.update({'psm': psm, 'rm': rm, 'com': communication_rates})
        return output_dict
