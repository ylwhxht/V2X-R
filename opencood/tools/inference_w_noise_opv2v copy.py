"""
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
"""

import argparse
import os
import time
from typing import OrderedDict

import torch
import open3d as o3d
from torch.utils.data import DataLoader
import numpy as np

# import opencood.hypes_yaml.yaml_utils as yaml_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils_opv2v as eval_utils
from opencood.visualization import simple_vis

torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--also_laplace', action='store_true',
                        help="whether to use laplace to simulate noise. Otherwise Gaussian")
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--comm_thre', type=float, default=None,
                        help='Communication confidence threshold')
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--eval_epoch', type=int, default=None, help='eval epoch')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    left_hand = True if 'left_hand' in hypes and hypes['left_hand'] else False

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    eval_epoch, model = train_utils.load_saved_model(saved_path, model, opt.eval_epoch)
    model.eval()

    # add noise to pose.
    # pos_std_list = [0, 0.2, 0.4, 0.6] # 位姿噪声标准差
    # rot_std_list = [0, 0.2, 0.4, 0.6] # 旋转噪声标准差
    # pos_mean_list = [0, 0, 0, 0] # 位姿噪声均值
    # rot_mean_list = [0, 0, 0, 0] # 旋转噪声均值
    pos_std_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 位姿噪声标准差
    rot_std_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # 旋转噪声标准差
    pos_mean_list = [0, 0, 0, 0] # 位姿噪声均值
    rot_mean_list = [0, 0, 0, 0] # 旋转噪声均值
    """
    pos_std_list = [0.6]
    rot_std_list = [0.6]
    pos_mean_list = [0]
    rot_mean_list = [0]
    """
    
    if opt.also_laplace:
        use_laplace_options = [False, True]
    else:
        use_laplace_options = [False]

    for use_laplace in use_laplace_options:
        AP30 = []
        AP50 = []
        AP70 = []
        for (pos_mean, pos_std, rot_mean, rot_std) in zip(pos_mean_list, pos_std_list, rot_mean_list, rot_std_list):
            # setting noise
            np.random.seed(303)
            noise_setting = OrderedDict()
            noise_args = {'pos_std': pos_std,
                          'rot_std': rot_std,
                          'pos_mean': pos_mean,
                          'rot_mean': rot_mean}

            noise_setting['add_noise'] = True
            noise_setting['args'] = noise_args
            suffix = ""
            if use_laplace:
                noise_setting['args']['laplace'] = True
                suffix = "_laplace"
            
            # build dataset for each noise setting
            print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
            hypes['wild_setting'].update({'loc_err': True, 'ryp_std': rot_std, 'xyz_std': pos_std})
            print('wild_setting: ', hypes['wild_setting'])
            
            print('Dataset Building')
            opencood_dataset = build_dataset(hypes, visualize=True, train=False)
            data_loader = DataLoader(opencood_dataset,
                                    batch_size=1,
                                    num_workers=16,
                                    collate_fn=opencood_dataset.collate_batch_test,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)
            
            # Create the dictionary for evaluation
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                           0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                           0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
            
            noise_level = f"{pos_std}_{rot_std}_{pos_mean}_{rot_mean}_" + opt.fusion_method + suffix + opt.note


            for i, batch_data in enumerate(data_loader):
                print(f"{noise_level}_{i}")
                if batch_data is None:
                    continue
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                    
                    if opt.fusion_method == 'late':
                        pred_box_tensor, pred_score, gt_box_tensor = \
                            inference_utils.inference_late_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'early':
                        pred_box_tensor, pred_score, gt_box_tensor = \
                            inference_utils.inference_early_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'intermediate':
                        pred_box_tensor, pred_score, gt_box_tensor = \
                            inference_utils.inference_intermediate_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    else:
                        raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                                'fusion is supported.')

                    #pred_box_tensor = infer_result['pred_box_tensor']
                    #gt_box_tensor = infer_result['gt_box_tensor']
                    #pred_score = infer_result['pred_score']

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
                        inference_utils.save_prediction_gt(pred_box_tensor,
                                                        gt_box_tensor,
                                                        batch_data['ego'][
                                                            'origin_lidar'][0],
                                                        i,
                                                        npy_save_path)

                    if opt.save_vis:
                        vis_save_path = os.path.join(opt.model_dir, 'vis_3d')
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        vis_save_path = os.path.join(opt.model_dir, 'vis_3d/3d_%05d.png' % i)
                        simple_vis.visualize(pred_box_tensor,
                                            gt_box_tensor,
                                            batch_data['ego']['origin_lidar'][0],
                                            hypes['postprocess']['anchor_args']['cav_lidar_range'],
                                            vis_save_path,
                                            method='3d',
                                            left_hand=left_hand,
                                            vis_pred_box=True)
                        
                        vis_save_path = os.path.join(opt.model_dir, 'vis_bev')
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        vis_save_path = os.path.join(opt.model_dir, 'vis_bev/bev_%05d.png' % i)
                        simple_vis.visualize(pred_box_tensor,
                                            gt_box_tensor,
                                            batch_data['ego']['origin_lidar'][0],
                                            hypes['postprocess']['anchor_args']['cav_lidar_range'],
                                            vis_save_path,
                                            method='bev',
                                            left_hand=left_hand,
                                            vis_pred_box=True)

                torch.cuda.empty_cache()

            ap30, ap50, ap70 = eval_final_results(result_stat, opt.model_dir, noise_level, opt.eval_epoch)
            AP30.append(ap30)
            AP50.append(ap50)
            AP70.append(ap70)

            dump_dict = {'ap30': AP30 ,'ap50': AP50, 'ap70': AP70}
            yaml_utils.save_yaml(dump_dict, os.path.join(opt.model_dir, f'AP030507_{opt.note}{suffix}.yaml'))


def eval_final_results(result_stat, save_path, infer_info=None, epoch=None):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    dump_dict.update({'ap30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    if infer_info is None:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f'eval_{infer_info}.yaml'))

    print('The Average Precision at IOU 0.3 is %.2f, '
          'The Average Precision at IOU 0.5 is %.2f, '
          'The Average Precision at IOU 0.7 is %.2f' % (ap_30, ap_50, ap_70))

    return ap_30, ap_50, ap_70


def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = iou_5['fp']
    tp = iou_5['tp']
    assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


if __name__ == '__main__':
    main()
