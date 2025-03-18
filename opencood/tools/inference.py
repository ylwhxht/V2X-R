# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time

import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
# from opencood.tools import train_utils as train_utils
from opencood.tools import inference_utils as inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import simple_vis
from tqdm import tqdm
from PIL import Image
import numpy as np

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis', type=bool, default=False,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--save_vis_n', type=int, default=10,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--eval_epoch', type=int, default=20,
                        help='Set the checkpoint')
    parser.add_argument('--eval_best_epoch', type=bool, default=False,
                        help='Set the checkpoint')
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Communication confidence threshold')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'intermediate_with_comm', 'no']

    hypes = yaml_utils.load_yaml(None, opt)

    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre

    if 'opv2v' or 'V2XR' in opt.model_dir:
        from opencood.utils import eval_utils_opv2v as eval_utils
        left_hand = True

    elif 'dair' in opt.model_dir:
        from opencood.utils import eval_utils_where2comm as eval_utils
        hypes['validate_dir'] = hypes['test_dir']
        left_hand = False

    else:
        print(f"The path should contain one of the following strings [opv2v|dair] .")
        return 
    
    print(f"Left hand visualizing: {left_hand}")

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False
                             )

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    epoch_id, model = train_utils.load_model(saved_path, model, opt.eval_epoch, start_from_best=opt.eval_best_epoch)
    
    model.zero_grad()
    model.eval()

    # Create the dictionary for evaluation
    #result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
    #               0.5: {'tp': [], 'fp': [], 'gt': 0},
    #               0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    total_comm_rates = []
    # total_box = []
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            # _batch_data = batch_data[0]
            batch_data = train_utils.to_device(batch_data, device)
            # print(_batch_data.keys())
            # _batch_data = train_utils.to_device(_batch_data, device)
            # if 'scope' in hypes['name'] or 'how2comm' in hypes['name']:
            #     batch_data= _batch_data
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, output_dict = inference_utils.inference_late_fusion(batch_data, model, opencood_dataset)
                comm = 0
                for key in output_dict:
                    comm += output_dict[key]['comm_rates']
                total_comm_rates.append(comm)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(batch_data, model, opencood_dataset)
            elif opt.fusion_method == 'no':
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)
            
            elif opt.fusion_method == 'intermediate_with_comm':
                pred_box_tensor, pred_score, gt_box_tensor, comm_rates, mask, each_mask = inference_utils.inference_intermediate_fusion_withcomm(batch_data, model, opencood_dataset)
                total_comm_rates.append(comm_rates)
            else:
                raise NotImplementedError('Only early, late and intermediate, no, intermediate_with_comm fusion modes are supported.')
            if pred_box_tensor is None:
                continue

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
                                       
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor, gt_box_tensor, batch_data['ego']['origin_lidar'][0], i, npy_save_path)

            # if opt.save_vis_n and opt.save_vis_n >i:
            if opt.save_vis:
                """
                vis_save_path = os.path.join(opt.model_dir, 'vis_lidar')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(opt.model_dir, 'vis_lidar/ori_%05d.png' % i)
                simple_vis.visualize(None, gt_box_tensor, batch_data['ego']['origin_lidar'][0], hypes['postprocess']['gt_range'],
                            vis_save_path, method='3d', vis_gt_box=False, vis_pred_box=False, left_hand=False)

                vis_save_path = os.path.join(opt.model_dir, 'vis_image')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                image = batch_data['ego']['image_inputs']['ori_imgs'][0][0].cpu().numpy()
                vis_save_path = os.path.join(opt.model_dir, 'vis_image/camera0_%05d.png' % i)
                pil_image = Image.fromarray(image.astype(np.uint8))
                pil_image.save(vis_save_path)
                
                
                vis_save_path = os.path.join(opt.model_dir, 'vis_depth')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(opt.model_dir, 'vis_depth/depth_%05d.png' % i)
                depth_map = batch_data['ego']['image_inputs']['depth_map'][0][0].cpu().numpy()
                # 3, 360, 480 / 1, 360, 480
                print(np.max(depth_map), np.min(depth_map))
                _max, _min = np.max(depth_map), np.min(depth_map)
                depth_map = (depth_map - _min) / (_max - _min)
                depth_mask = (depth_map >= 1).astype(np.float32)
                depth_map = depth_map[0]
                H, W = depth_map.shape
                draw_depth = np.ones((3,H,W))
                draw_depth[0] = draw_depth[0] * (depth_map*(1-depth_mask)*255+depth_mask*255)
                draw_depth[1] = draw_depth[1] * (depth_map*(1-depth_mask)*140+depth_mask*255)
                draw_depth[2] = draw_depth[2] * (depth_map*(1-depth_mask)*0+depth_mask*255)
                draw_depth = draw_depth.transpose(1,2,0)

                draw_depth[draw_depth==0] = 255
                pil_depth = Image.fromarray(draw_depth.astype(np.uint8))
                pil_depth.save(vis_save_path)
                """
                
                vis_save_path = os.path.join(opt.model_dir, 'vis_3d')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(opt.model_dir, 'vis_3d/3d_%05d.png' % i)
                simple_vis.visualize(pred_box_tensor, gt_box_tensor, batch_data['ego']['origin_lidar'][0], 
                                     hypes['preprocess']['cav_lidar_range'], # hypes['postprocess']['gt_range'], 
                                     vis_save_path, method='3d', left_hand=left_hand, vis_pred_box=True)
                
                vis_save_path = os.path.join(opt.model_dir, 'vis_bev')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(opt.model_dir, 'vis_bev/bev_%05d.png' % i)
                simple_vis.visualize(pred_box_tensor, gt_box_tensor, batch_data['ego']['origin_lidar'][0],
                                     hypes['preprocess']['cav_lidar_range'], # hypes['postprocess']['gt_range'], 
                                     vis_save_path, method='bev', left_hand=left_hand, vis_pred_box=True)
                # if opt.fusion_method == 'intermediate_with_comm':
                #     vis_save_path = os.path.join(opt.model_dir, 'vis_mask')
                #     if not os.path.exists(vis_save_path):
                #         os.makedirs(vis_save_path)
                #     vis_save_path = os.path.join(opt.model_dir, 'vis_mask/%05d.png' % i)

                #     # 1, H, W
                #     mask = mask[0][0].cpu().numpy()
                #     H, W = mask.shape
                #     draw_map = np.zeros((H*W, 3))

                #     mask1d = mask.reshape(-1)

                #     for index, value in enumerate(mask1d):
                #         draw_map[index][0] = 255 if value==2 or value==4 else 0
                #         draw_map[index][1] = 255 if value==1 or value==4 else 125 if value==2 else 0
                #         draw_map[index][2] = 255 if value==3 or value==4 else 0

                #     # 1:fusion cell, green = [0,255,0]
                #     # 2:LiDAR cell, orange = [255,125,0]
                #     # 3:camera cell, blue = [0,0,255]
                #     # 4:other cell, white = [255, 255, 255] black=[0,0,0]
                #     draw_map = draw_map.reshape(H, W, 3)
                #     pil_depth = Image.fromarray(draw_map.astype(np.uint8))
                #     pil_depth.save(vis_save_path)


                #     vis_save_path = os.path.join(opt.model_dir, 'vis_emask')
                #     if not os.path.exists(vis_save_path):
                #         os.makedirs(vis_save_path)

                #     each_mask = each_mask[:,0]
                #     draw_emap = np.zeros((2, H*W, 3))
                #     emask1d = each_mask.reshape(2, H*W)

                #     for j in range(2):
                #         for index, value in enumerate(emask1d[j]):
                #             draw_emap[j][index][0] = 255 if value else 0
                #             draw_emap[j][index][1] = 255 if value else 0
                #             draw_emap[j][index][2] = 255 if value else 0
                #         vis_save_path = os.path.join(opt.model_dir, 'vis_emask/{}_{}.png'.format(i, j))

                #         # 1:fusion cell, green = [0,255,0]
                #         # 2:LiDAR cell, orange = [255,125,0]
                #         # 3:camera cell, blue = [0,0,255]
                #         # 4:other cell, white = [255, 255, 255] black=[0,0,0]
                #         emap = draw_emap[j].reshape(H, W, 3)
                #         pil_mask = Image.fromarray(emap.astype(np.uint8))
                #         pil_mask.save(vis_save_path)
                """
                
                if opt.fusion_method == 'intermediate_with_comm':
                    vis_save_path = os.path.join(opt.model_dir, 'vis_mask')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(opt.model_dir, 'vis_mask/%05d.png' % i)

                    # 1, H, W
                    mask = mask[0][0].cpu().numpy()
                    H, W = mask.shape
                    draw_map = np.zeros((H*W, 3))

                    mask1d = mask.reshape(-1)

                    for index, value in enumerate(mask1d):
                        draw_map[index][0] = 255 if value==2 or value==4 else 0
                        draw_map[index][1] = 255 if value==1 or value==4 else 125 if value==2 else 0
                        draw_map[index][2] = 255 if value==3 or value==4 else 0

                    # 1:fusion cell, green = [0,255,0]
                    # 2:LiDAR cell, orange = [255,125,0]
                    # 3:camera cell, blue = [0,0,255]
                    # 4:other cell, white = [255, 255, 255] black=[0,0,0]
                    draw_map = draw_map.reshape(H, W, 3)
                    pil_depth = Image.fromarray(draw_map.astype(np.uint8))
                    pil_depth.save(vis_save_path)


                    vis_save_path = os.path.join(opt.model_dir, 'vis_emask')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)

                    each_mask = each_mask[:,0]
                    draw_emap = np.zeros((2, H*W, 3))
                    emask1d = each_mask.reshape(2, H*W)

                    for j in range(2):
                        for index, value in enumerate(emask1d[j]):
                            draw_emap[j][index][0] = 255 if value else 0
                            draw_emap[j][index][1] = 255 if value else 0
                            draw_emap[j][index][2] = 255 if value else 0
                        vis_save_path = os.path.join(opt.model_dir, 'vis_emask/{}_{}.png'.format(i, j))

                        # 1:fusion cell, green = [0,255,0]
                        # 2:LiDAR cell, orange = [255,125,0]
                        # 3:camera cell, blue = [0,0,255]
                        # 4:other cell, white = [255, 255, 255] black=[0,0,0]
                        emap = draw_emap[j].reshape(H, W, 3)
                        pil_mask = Image.fromarray(emap.astype(np.uint8))
                        pil_mask.save(vis_save_path)
                """
                pass
            
    # print('total_box: ', sum(total_box)/len(total_box))

    if len(total_comm_rates) > 0:
        comm_rates = (sum(total_comm_rates)/len(total_comm_rates))
        if not isinstance(comm_rates, float):
            comm_rates = comm_rates.item()
    else:
        comm_rates = 0
    ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat, opt.model_dir)
    import math
    #comm_rates_base2=math.log(comm_rates,2)
    with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
        msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} \n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates)
        if opt.comm_thre is not None:
            msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} | comm_thre: {:.04f}\n'.format(epoch_id, ap_30, ap_50, ap_70, comm_rates, opt.comm_thre)
        f.write(msg)
        print(msg)


if __name__ == '__main__':
    main()
