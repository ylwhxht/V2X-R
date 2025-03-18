from numpy import record
import torch
import torch.nn as nn

from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor

from opencood.models.scope_modules.scope_attn import SCOPE
from opencood.models.scope_modules.temporal_late_fusion import LateFusion
from opencood.models.scope_modules.temporal_fusion_lstm import TemporalFusion_lstm
from opencood.models.common_modules.torch_transformation_utils import warp_affine_simple


def transform_feature(feature_list,matrix_list,downsample_rate,discrete_ratio):
    B = len(feature_list[0])
    _, C, H, W = feature_list[0][0].shape
    new_list =[]
    for i in range(0,len(matrix_list)):
        pairwise_t_matrix = matrix_list[i]
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2
        new_list.append(pairwise_t_matrix)
        
    temporal_list = []
    for b in range(B):
        input = [x[b][0:1,:] for x in feature_list]
        input = torch.cat(input,dim=0)
        
        history_matrix = [m[b,0:1,1:2,:,:] for m in new_list]
        history_matrix = torch.cat(history_matrix[1:],dim=1).squeeze(0)
        history_matrix = torch.cat([new_list[0][b,0,0:1,:,:],history_matrix],dim=0)
        
        history_feature = warp_affine_simple(input,history_matrix,(H,W))
        temporal_list.append(history_feature)
    
    return temporal_list

class PointPillarScope(nn.Module):
    def __init__(self, args):
        super(PointPillarScope, self).__init__()

        self.modality = 'processed_lidar'
        if 'use_modality' in args.keys():
            self.modality = args['use_modality']

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
            
        self.pillar_vfe_2 = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter_2 = PointPillarScatter(args['point_pillar_scatter'])
        if 'resnet' in args['base_bev_backbone']:
            self.backbone_2 = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            self.backbone_2 = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.dcn = False
        # if 'dcn' in args:
        #     self.dcn = True
        #     self.dcn_net = DCNNet(args['dcn'])

        self.fusion_net = SCOPE(args['fusion_args'])
        self.frame = args['fusion_args']['frame']
        self.discrete_ratio = args['fusion_args']['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['fusion_args']['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        self.temporal_fusion = TemporalFusion_lstm(args['fusion_args'])
        self.late_fusion = LateFusion(args['fusion_args']['communication'])
        self.multi_scale = args['fusion_args']['multi_scale']

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
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
        

    def forward(self, data_dict_list):
        batch_dict_list = [] 
        feature_list = []  
        feature_2d_list = []  
        matrix_list = []
        regroup_feature_list = []  
        regroup_feature_list_large = []  
        # print('data_dict_list:',data_dict_list.keys())
        for origin_data in data_dict_list:  
            # data_dict = origin_data['ego'] # train
            data_dict = data_dict_list # inference
            # voxel_features = data_dict['processed_lidar']['voxel_features']
            # voxel_coords = data_dict['processed_lidar']['voxel_coords']
            # voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
            voxel_features = data_dict[self.modality]['voxel_features']
            voxel_coords = data_dict[self.modality]['voxel_coords']
            voxel_num_points = data_dict[self.modality]['voxel_num_points']
            record_len = data_dict['record_len']

            pairwise_t_matrix = data_dict['pairwise_t_matrix']
            batch_dict = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'record_len': record_len}
            batch_dict = self.pillar_vfe(batch_dict)
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)
            spatial_features_2d = batch_dict['spatial_features_2d']
            
            # downsample feature to reduce memory
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
            if self.compression:
                spatial_features_2d = self.naive_compressor(spatial_features_2d)
            # dcn
            if self.dcn:
                spatial_features_2d = self.dcn_net(spatial_features_2d)
                
            batch_dict_list.append(batch_dict)
            spatial_features = batch_dict['spatial_features']
            feature_list.append(spatial_features)
            feature_2d_list.append(spatial_features_2d)
            matrix_list.append(pairwise_t_matrix)  
            regroup_feature_list.append(self.regroup(spatial_features_2d,record_len))  
            regroup_feature_list_large.append(self.regroup(spatial_features,record_len))
        
        pairwise_t_matrix = matrix_list[0].clone().detach()  
        if self.frame > 0: 
            history_feature = transform_feature(regroup_feature_list_large,matrix_list,self.downsample_rate,self.discrete_ratio)
            history_feature_2d = transform_feature(regroup_feature_list,matrix_list,self.downsample_rate,self.discrete_ratio)
            fusion_list = []
            for b in range(len(history_feature)):
                fusion_list.append(self.temporal_fusion(history_feature_2d[b]))
            temporal_output = torch.cat(fusion_list,dim=0)  # B,C,H,W
            psm_temporal = self.cls_head(temporal_output)
            rm_temporal = self.reg_head(temporal_output)        
        
        spatial_features = feature_list[0]
        spatial_features_2d = feature_2d_list[0]
        batch_dict = batch_dict_list[0]
        record_len = batch_dict['record_len']
        
        
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)

        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix, 
                                            self.backbone,
                                            [self.shrink_conv, self.cls_head, self.reg_head])
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else: 
            fused_feature, communication_rates, result_dict = self.fusion_net(spatial_features_2d,
                                            psm_single,
                                            record_len,
                                            pairwise_t_matrix)
        
        psm_cross = self.cls_head(fused_feature)
        rm_cross = self.reg_head(fused_feature)

        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []
        for b in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[b][0:1])
            psm_single_i.append(split_psm_single[b][1:2])
            rm_single_v.append(split_rm_single[b][0:1])
            rm_single_i.append(split_rm_single[b][1:2])
        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)
        
        ego_feature_list = [x[0:1,:] for x in regroup_feature_list[0]]
        ego_feature = torch.cat(ego_feature_list,dim=0)
        final_feature = self.late_fusion([temporal_output,ego_feature,fused_feature],psm_temporal,psm_single_v,psm_cross)
        # print('fused_feature:{},final_feature:{}'.format(fused_feature.shape,final_feature.shape))
        
        psm = self.cls_head(final_feature)
        rm = self.reg_head(final_feature)

        output_dict = {'psm': psm,
                    'rm': rm
                    }
        output_dict.update(result_dict)
        # print("communication rate:",communication_rates)
        
        output_dict.update({'psm_single_v': psm_single_v,
                    'psm_single_i': psm_single_i,
                    'rm_single_v': rm_single_v,
                    'rm_single_i': rm_single_i,
                    'comm_rate': communication_rates
                    })
        
        return output_dict
