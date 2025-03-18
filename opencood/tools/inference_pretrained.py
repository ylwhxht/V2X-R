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
from opencood.tools import inference_utils as inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import simple_vis
from tqdm import tqdm
from PIL import Image
import numpy as np

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the .pth model file')
    parser.add_argument('--hypes_yaml', type=str, required=True,
                        help='Path to the training configuration yaml file')
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
    parser.add_argument('--eval_epoch', type=int, default=None,
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

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)

    if opt.comm_thre is not None:
        hypes['model']['args']['fusion_args']['communication']['thre'] = opt.comm_thre

    if 'opv2v' or 'V2XR' in opt.model_path:
        from opencood.utils import eval_utils_opv2v as eval_utils
        left_hand = True
    elif 'dair' in opt.model_path:
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
    try:
        state_dict = torch.load(opt.model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.zero_grad()
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    total_comm_rates = []
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            _batch_data = batch_data[0]
            # print(_batch_data.keys())
            _batch_data = train_utils.to_device(_batch_data, device)
            if 'scope' in hypes['name'] or 'how2comm' in hypes['name']:
                batch_data= _batch_data
            # # -------------------
            # batch_data = train_utils.to_device(batch_data, device)
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
                npy_save_path = os.path.dirname(opt.model_path)
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor, gt_box_tensor, batch_data['ego']['origin_lidar'][0], i, npy_save_path)

            if opt.save_vis:
                vis_save_path = os.path.join(os.path.dirname(opt.model_path), 'vis_3d')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(vis_save_path, '3d_%05d.png' % i)
                simple_vis.visualize(pred_box_tensor, gt_box_tensor, batch_data['ego']['origin_lidar'][0], 
                                     hypes['preprocess']['cav_lidar_range'], 
                                     vis_save_path, method='3d', left_hand=left_hand, vis_pred_box=True)
                
                vis_save_path = os.path.join(os.path.dirname(opt.model_path), 'vis_bev')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(vis_save_path, 'bev_%05d.png' % i)
                simple_vis.visualize(pred_box_tensor, gt_box_tensor, batch_data['ego']['origin_lidar'][0],
                                     hypes['preprocess']['cav_lidar_range'], 
                                     vis_save_path, method='bev', left_hand=left_hand, vis_pred_box=True)
            
    if len(total_comm_rates) > 0:
        comm_rates = (sum(total_comm_rates)/len(total_comm_rates))
        if not isinstance(comm_rates, float):
            comm_rates = comm_rates.item()
    else:
        comm_rates = 0
    ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat, os.path.dirname(opt.model_path))
    
    model_name = os.path.basename(opt.model_path).replace('.pth', '')
    result_file_path = os.path.join(os.path.dirname(opt.model_path), 'result.txt')
    with open(result_file_path, 'a+') as f:
        msg = f'Model: {model_name} | AP @0.3: {ap_30:.04f} | AP @0.5: {ap_50:.04f} | AP @0.7: {ap_70:.04f} | comm_rate: {comm_rates:.06f}\n'
        if opt.comm_thre is not None:
            msg = f'Model: {model_name} | AP @0.3: {ap_30:.04f} | AP @0.5: {ap_50:.04f} | AP @0.7: {ap_70:.04f} | comm_rate: {comm_rates:.06f} | comm_thre: {opt.comm_thre:.04f}\n'
        f.write(msg)
        print(msg)

    # Save eval result to yaml file
    dump_dict = {'ap30': [ap_30], 'ap50': [ap_50], 'ap70': [ap_70]}
    yaml_save_path = os.path.join(os.path.dirname(opt.model_path), f'eval_{model_name}.yaml')
    yaml_utils.save_yaml(dump_dict, yaml_save_path)


if __name__ == '__main__':
    main()