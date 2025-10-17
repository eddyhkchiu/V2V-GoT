import argparse
import os
import time
import numpy as np

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils
from opencood.utils import box_utils
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.box_utils import project_points_by_matrix_torch

import json
import random

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--save_visible_gt_ids', action='store_true',
                        help='whether to save visiable gt ids'
                             'in npy file')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion', 'no_fusion_keep_all']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False,
                                     isSim=opt.isSim)
    print(hypes['fusion']['core_method'])
    # IntermediateFusionDataset

    print("opencood_dataset.len_record: ", opencood_dataset.len_record)
    # train set has 32 sequences
    # [147, 552, 709, 1953, 2086, 2303, 2425, 2573, 2983, 3298, 3417, 3524, 3648, 3737, 3817, 3962, 4255, 4366, 4549, 4726, 5001, 5287, 5516, 5636, 5804, 6254, 6389, 6532, 6681, 6846, 6997, 7105]
    # test set has 9 sequences
    # [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
    print("opencood_dataset.scenario_database.keys(): ", opencood_dataset.scenario_database.keys())
    # odict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8])
    npy_save_path = os.path.join(opt.model_dir, 'npy')
    print('npy_save_path: ', npy_save_path)
    # ./official_models/cobevt/npy
    #assert False


    # for faster debug
    #transform_and_save_detection_to_ab3dmot_format(opencood_dataset.len_record, npy_save_path, {'ego', '1'})
    #assert False

    # for faster debug
    # only need to generate gt label once
    #transform_and_save_tracking_label_to_ab3dmot_format(opencood_dataset.len_record, npy_save_path)
    #assert False

    # for faster debug
    # generate dataset in llava format
    # full scene text-only late fusion cooperative detection
    #generate_v2v4real_dataset_for_llava(opencood_dataset.len_record, npy_save_path, {'ego', '1'})
    #assert False

    # for faster debug
    #generate_3d_grounding_qa_dataset(opencood_dataset.len_record, npy_save_path, {'ego', '1'})
    #assert False

    # for faster debug
    #generate_3d_grounding_qa_dataset_v2(opencood_dataset.len_record, npy_save_path, {'ego', '1'}, simplified=True, debug=True)
    #assert False

    # not used
    # for faster debug
    #generate_3d_grounding_qa_dataset_v3(opencood_dataset.len_record, npy_save_path, {'ego', '1'})
    #assert False

    # for faster debug
    # q2 v4
    #generate_3d_grounding_qa_dataset_v4(opencood_dataset.len_record, npy_save_path, {'ego', '1'}, downsample_negatives=True, simplified=True, max_num_answer_objects=100)
    #assert False

    # for faster debug
    # q3 v5
    #generate_3d_grounding_qa_dataset_v5(opencood_dataset.len_record, npy_save_path, {'ego', '1'}, downsample_negatives=True, simplified=True, max_num_answer_objects=100)
    #assert False

    # for faster debug
    # q4 v6
    #generate_3d_grounding_qa_dataset_v6(opencood_dataset.len_record, npy_save_path, {'ego', '1'}, downsample_negatives=False, simplified=True, max_num_answer_objects=3)
    #assert False

    # for faster debug
    # q4 v6 double cavs
    #generate_3d_grounding_qa_dataset_v6_double(opencood_dataset.len_record, npy_save_path, {'ego', '1'}, downsample_negatives=False, simplified=True, max_num_answer_objects=100, double_cavs=True)
    #assert False


    # for faster debug
    # q5 v7
    #generate_3d_grounding_qa_dataset_v7(opencood_dataset.len_record, npy_save_path, {'ego', '1'}, downsample_negatives=False, simplified=True, max_num_answer_objects=100, num_future_waypoints=6)
    # assert False

    # for faster debug
    # q5 v7 double cavs
    #generate_3d_grounding_qa_dataset_v7_double(opencood_dataset.len_record, npy_save_path, {'ego', '1'}, downsample_negatives=False, simplified=True, max_num_answer_objects=100, num_future_waypoints=6, double_cavs=True)
    #assert False


    # graph QA
    # nq8 suggested action classification without context
    #generate_3d_grounding_qa_dataset_nq8(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=6, double_cavs=True, with_context=False, context_list=None, output_file=None, context_list_from_gt=None)
    #assert False
    # graph QA
    # nq8 suggested action classification with gt nq7 answer as context
    #generate_3d_grounding_qa_dataset_nq8(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=6, double_cavs=True, with_context=True, context_list=None, output_file=None, context_list_from_gt=None)
    #assert False

    # graph QA
    # nq9 suggested future trajectory without context
    #generate_3d_grounding_qa_dataset_nq9(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=6, double_cavs=True, with_context=False)
    #assert False
    # graph QA
    # nq9 suggested future trajectory with gt nq8 answer as context
    #generate_3d_grounding_qa_dataset_nq9(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=6, double_cavs=True, with_context=True, context_list=None, output_file=None, context_list_from_gt=None)
    #assert False

    # graph QA
    # nq5 prediction by observation without context
    # w6
    #generate_3d_grounding_qa_dataset_nq5(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=6, double_cavs=True)
    # w0
    #generate_3d_grounding_qa_dataset_nq5(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=0, double_cavs=True)
    # w1
    #generate_3d_grounding_qa_dataset_nq5(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=1, double_cavs=True, with_context=False, context_list=None, output_file=None, context_list_from_gt=None)
    #assert False
    # graph QA
    # nq5 prediction by observation with gt nq4 answer as context
    # w1
    #generate_3d_grounding_qa_dataset_nq5(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=1, double_cavs=True, with_context=True, context_list=None, output_file=None, context_list_from_gt=None)
    #assert False

    # graph QA
    # nq4 notable object identification without context
    #generate_3d_grounding_qa_dataset_nq4(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=0, double_cavs=True, with_context=False, context_list=None, output_file=None, context_list_from_gt=None)
    #assert False
    # graph QA
    # nq4 notable object identification with gt nq1 and nq3 as context
    #generate_3d_grounding_qa_dataset_nq4(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=0, double_cavs=True, with_context=True, context_list=None, output_file=None, context_list_from_gt=None)
    #assert False

    # graph QA
    # nq3 invisible notable object without context
    #generate_3d_grounding_qa_dataset_nq3(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=0, double_cavs=True, with_context=False, context_list=None, output_file=None, context_list_from_gt=None)
    #assert False
    # graph QA
    # nq3 invisible notable object with gt nq2 as context
    #generate_3d_grounding_qa_dataset_nq3(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=0, double_cavs=True, with_context=True, context_list=None, output_file=None,  context_list_from_gt=None)
    #assert False


    # graph QA
    # nq1 visible notable object without context
    #generate_3d_grounding_qa_dataset_nq1(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=0, double_cavs=True, with_context=False)
    #assert False


    # graph QA
    # nq2 occluding object without context
    #generate_3d_grounding_qa_dataset_nq2(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=0, double_cavs=True, with_context=False)
    #assert False

    
    # graph QA
    # nq6 prediction by other CAV planning with context
    #generate_3d_grounding_qa_dataset_nq6(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=1, double_cavs=True, with_context=True, context_list=None, output_file=None,  context_list_from_gt=None)
    #assert False

    # nq6 prediction by other CAV planning without context
    #generate_3d_grounding_qa_dataset_nq6(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=1, double_cavs=True, with_context=False, context_list=None, output_file=None,  context_list_from_gt=None)
    #assert False

    # graph QA
    # nq7 all prediction, with NQ5 NQ6 context 
    #generate_3d_grounding_qa_dataset_nq7(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=1, double_cavs=True, with_context=True,  context_list=None, output_file=None, context_list_from_gt=None)
    #assert False

    # nq7 all prediction, without NQ5 NQ6 context 
    #generate_3d_grounding_qa_dataset_nq7(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=1, double_cavs=True, with_context=False, context_list=None, output_file=None, context_list_from_gt=None)
    #assert False


    # this assert prevent generating full inference output
    # when we only want to generate QA dataset
    print('QA data generation finished.')
    assert False


    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                        0.7: {'tp': [], 'fp': [], 'gt': 0}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 10
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(500):
            vis_aabbs_gt.append(o3d.geometry.TriangleMesh())
            vis_aabbs_pred.append(o3d.geometry.TriangleMesh())

    total_time = 0.0

    cav_id_set = set()

    for i, batch_data in enumerate(data_loader):
        print(i)
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
                    infrence_utils.inference_no_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)
            elif opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
                    infrence_utils.inference_late_fusion(batch_data,
                                                         model,
                                                         opencood_dataset)
            elif opt.fusion_method == 'no_fusion_keep_all':
                start_time = time.time()
                pred_box_dict, pred_score_dict, gt_box_tensor, gt_object_id_tensor, pred_feature_dict, pred_early_feature_dict, spatial_features_dict, spatial_features_2d_dict, regression_map_dict, classification_map_dict, visible_gt_object_ids_dict, invisible_gt_object_ids_dict = \
                    infrence_utils.inference_no_fusion_keep_all(batch_data,
                                                                model,
                                                                opencood_dataset)
                end_time = time.time()
                total_time += (end_time - start_time)

                transformation_matrix_dict = {}
                projected_lidar_dict = {}
                lidar_pose_dict = {}
                for cav_id in batch_data.keys():
                  transformation_matrix_dict[cav_id] = batch_data[cav_id]['transformation_matrix']
                  projected_lidar_dict[cav_id] = batch_data[cav_id]['projected_lidar']
                  #print('projected_lidar_dict[cav_id].shape: ', projected_lidar_dict[cav_id].shape)
                  lidar_pose_dict[cav_id] = batch_data[cav_id]['lidar_pose']
                #print('transformation_matrix_dict: ', transformation_matrix_dict)


                #if i == 31:
                #  print('gt_object_id_tensor: ', gt_object_id_tensor)
                #  print('visible_gt_object_ids_dict: ', visible_gt_object_ids_dict)  
                #  print('invisible_gt_object_ids_dict: ', invisible_gt_object_ids_dict)  
                #  assert False 

                # get lidar_pose
                #print('lidar_pose_dict: ', lidar_pose_dict)
                #assert False

                #print('pred_box_dict: ', pred_box_dict)
                #print('pred_score_dict: ', pred_score_dict)
                #print('pred_feature_dict: ', pred_feature_dict)

                # For no_fusion_keep_all,
                # the detection evaluation still use no_fusion approach
                # but we will save all cav's detection results for tracking
                pred_box_tensor = pred_box_dict['ego']
                pred_score = pred_score_dict['ego']

                for cav_id in pred_box_dict.keys():
                  cav_id_set.add(cav_id)

            elif opt.fusion_method == 'early':
                start_time = time.time()
                #pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
                #    infrence_utils.inference_early_fusion(batch_data,
                #                                          model,
                #                                          opencood_dataset)
                # MY_CODE
                pred_box_dict, pred_score_dict, gt_box_tensor, gt_object_id_tensor, pred_feature_dict, pred_early_feature_dict, spatial_features_dict, spatial_features_2d_dict, regression_map_dict, classification_map_dict = \
                    infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
                end_time = time.time()
                total_time += (end_time - start_time)    

                transformation_matrix_dict = {}
                projected_lidar_dict = None
                lidar_pose_dict = None
                for cav_id in batch_data.keys():
                  transformation_matrix_dict[cav_id] = batch_data[cav_id]['transformation_matrix']
                  #projected_lidar_dict[cav_id] = batch_data[cav_id]['projected_lidar']
                  #lidar_pose_dict[cav_id] = batch_data[cav_id]['lidar_pose']    

                pred_box_tensor = pred_box_dict['ego']
                pred_score = pred_score_dict['ego']

                for cav_id in pred_box_dict.keys():
                  cav_id_set.add(cav_id)  

                #print('regression_map_dict["ego"].shape: ', regression_map_dict["ego"].shape)  
                # [1, 14, 50, 176]

            elif opt.fusion_method == 'intermediate':
                start_time = time.time()
                # MY_DEBUG: new return format in dict
                #pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
                pred_box_dict, pred_score_dict, gt_box_tensor, gt_object_id_tensor, pred_feature_dict, pred_early_feature_dict, spatial_features_dict, spatial_features_2d_dict, regression_map_dict, classification_map_dict = \
                    infrence_utils.inference_intermediate_fusion(batch_data,
                                                                 model,
                                                                 opencood_dataset)
                end_time = time.time()
                total_time += (end_time - start_time)
                
                transformation_matrix_dict = {}
                projected_lidar_dict = None
                lidar_pose_dict = None
                for cav_id in batch_data.keys():
                  transformation_matrix_dict[cav_id] = batch_data[cav_id]['transformation_matrix']
                  #projected_lidar_dict[cav_id] = batch_data[cav_id]['projected_lidar']
                  #lidar_pose_dict[cav_id] = batch_data[cav_id]['lidar_pose']

                # This cobevt detection have the same mAP as the v2v4real paper: 0.665
                pred_box_tensor = pred_box_dict['ego']
                pred_score = pred_score_dict['ego']

                for cav_id in pred_box_dict.keys():
                  cav_id_set.add(cav_id)

                #print("batch_data['ego']: ", batch_data['ego'])
                #print("batch_data['ego'].keys(): ", batch_data['ego'].keys())

                #print("batch_data['ego']['record_len']: ", batch_data['ego']['record_len']) # 1
                #print("batch_data['ego']['object_ids']: ", batch_data['ego']['object_ids']) # 1

                if gt_box_tensor.shape[0] != gt_object_id_tensor.shape[0]:
                  # i == 98
                  #print("batch_data['ego']['object_ids']: ", batch_data['ego']['object_ids']) # [1, 2, 11]
                  print('gt_box_tensor.shape[0]: ', gt_box_tensor.shape[0]) # 2
                  print('gt_object_id_tensor: ', gt_object_id_tensor)
                  assert False
                
                #print("batch_data['ego']['scenario_index']: ", batch_data['ego']['scenario_index'])
                #print("batch_data['ego']['timestamp_index']: ", batch_data['ego']['timestamp_index'])

                # num_objects, num_corners_per_box, 3 dim coordinates
                #print('pred_box_tensor.shape: ', pred_box_tensor.shape) # [3, 8, 3]
                #print('pred_box_tensor: ', pred_box_tensor)

                #print('pred_score.shape: ', pred_score.shape) # [3]
                #print('pred_score: ', pred_score)

                #print('gt_box_tensor.shape: ', gt_box_tensor.shape) # [1, 8, 3]
                #print('gt_box_tensor: ', gt_box_tensor)
                #if i == 144:
                #  assert False


            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            # overall calculating
            #print('gt_box_tensor: ', gt_box_tensor)
            #print('len(gt_box_tensor): ', len(gt_box_tensor))
            #print('gt_object_id_tensor: ', gt_object_id_tensor)
            #print('pred_score: ', pred_score)
            #print('pred_box_tensor: ', pred_box_tensor)
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
            # short range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.5,
                                       left_range=0,
                                       right_range=30)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.7,
                                       left_range=0,
                                       right_range=30)

            # middle range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.5,
                                       left_range=30,
                                       right_range=50)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.7,
                                       left_range=30,
                                       right_range=50)

            # right range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.5,
                                       left_range=50,
                                       right_range=100)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.7,
                                       left_range=50,
                                       right_range=100)

            if opt.save_visible_gt_ids:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                infrence_utils.save_visible_gt_ids(i, 
                                                   npy_save_path, 
                                                   visible_gt_object_ids_dict, 
                                                   invisible_gt_object_ids_dict)


            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                npy_co_llm_save_path = os.path.join(opt.model_dir, 'npy', 'co_llm')
                if not os.path.exists(npy_co_llm_save_path):
                    os.makedirs(npy_co_llm_save_path)

                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path,
                                                  npy_co_llm_save_path,
                                                  pred_score,
                                                  gt_object_id_tensor)

                if opt.fusion_method == 'no_fusion_keep_all' or 'intermediate':
                  for cav_id in pred_box_dict.keys():
                    npy_cav_id_save_path = os.path.join(opt.model_dir, 'npy', cav_id)
                    if not os.path.exists(npy_cav_id_save_path):
                        os.makedirs(npy_cav_id_save_path)
                    npy_co_llm_cav_id_save_path = os.path.join(opt.model_dir, 'npy', 'co_llm', cav_id)
                    if not os.path.exists(npy_co_llm_cav_id_save_path):
                        os.makedirs(npy_co_llm_cav_id_save_path)

                    pred_box_tensor = pred_box_dict[cav_id]
                    pred_score = pred_score_dict[cav_id]
                    pred_feature = pred_feature_dict[cav_id]
                    pred_early_feature = pred_early_feature_dict[cav_id]
                    transformation_matrix = transformation_matrix_dict[cav_id]
                    #print('pred_feature.shape: ', pred_feature.shape)
                    spatial_features = spatial_features_dict[cav_id]
                    spatial_features_2d = spatial_features_2d_dict[cav_id]
                    regression_map = regression_map_dict[cav_id]
                    classification_map = classification_map_dict[cav_id]
                    #print('regression_map.shape: ', regression_map.shape)
                    # [1, 14, 50, 88]
                    #print('classification_map.shape: ', classification_map.shape)
                    # [1, 2, 50, 88]

                    #print("batch_data['ego']['projected_lidar'][0].shape: ", batch_data['ego']['projected_lidar'][0].shape)
                    #print("batch_data['1']['projected_lidar'][0].shape: ", batch_data['1']['projected_lidar'][0].shape)
                    #print("batch_data['ego']['lidar_pose']: ", batch_data['ego']['lidar_pose'])
                    #print("batch_data['1']['lidar_pose']: ", batch_data['1']['lidar_pose'])

                    projected_lidar = projected_lidar_dict[cav_id] if projected_lidar_dict is not None else None
                    lidar_pose = lidar_pose_dict[cav_id] if lidar_pose_dict is not None else None


                    infrence_utils.save_prediction_gt(pred_box_tensor,
                                                      gt_box_tensor,
                                                      batch_data['ego'][
                                                          'origin_lidar'][0],
                                                      i,
                                                      npy_cav_id_save_path,
                                                      npy_co_llm_cav_id_save_path,
                                                      pred_score,
                                                      gt_object_id_tensor,
                                                      pred_feature,
                                                      pred_early_feature,
                                                      transformation_matrix,
                                                      spatial_features,
                                                      spatial_features_2d,
                                                      regression_map,
                                                      classification_map,
                                                      projected_lidar[0] if projected_lidar is not None else None, # [N, 4]
                                                      lidar_pose)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        vis_pcd,
                        mode='constant'
                    )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    total_frames = i + 1
    speed = total_frames / total_time
    print('Detection speed: %f frames per second, total frames: %d, total time: %f ' % 
          (speed, total_frames, total_time))

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir)
    eval_utils.eval_final_results(result_stat_short,
                                  opt.model_dir,
                                  "short")
    eval_utils.eval_final_results(result_stat_middle,
                                  opt.model_dir,
                                  "middle")
    eval_utils.eval_final_results(result_stat_long,
                                  opt.model_dir,
                                  "long")


    
    # MY_CODE
    if opt.save_npy:
      # MY_CODE
      # only save the feature map and detection result
      # skip redo ab3dmot format of data for tracking
      #pass
      # we still need to call these functions to save detection results (and maybe gt?)
      transform_and_save_detection_to_ab3dmot_format(opencood_dataset.len_record, npy_save_path, cav_id_set)
      transform_and_save_tracking_label_to_ab3dmot_format(opencood_dataset.len_record, npy_save_path)

    if opt.show_sequence:
        vis.destroy_window()

def transform_and_save_detection_to_ab3dmot_format(len_record, npy_save_path, cav_id_set):
  # opencood_dataset.len_record
  print('len_record: ', len_record)
  # [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]

  max_num_individual_detections = 0

  # add '' to also tranform data in npy_save_path
  # which can be v2v4real's baseline: late fusion, cobevt's detetcion result
  cav_id_set.add('')

  for cav_id in cav_id_set:
    ab3dmot_detection_save_path = os.path.join(npy_save_path, 'ab3dmot_detection', cav_id)
    #print('ab3dmot_detection_save_path: ', ab3dmot_detection_save_path)
    if not os.path.exists(ab3dmot_detection_save_path):
      os.makedirs(ab3dmot_detection_save_path)


    # LLM
    co_llm_detection_save_path = os.path.join(npy_save_path, 'co_llm', cav_id) 
    #print('co_llm_detection_save_path: ', co_llm_detection_save_path)
    if not os.path.exists(co_llm_detection_save_path):
      os.makedirs(co_llm_detection_save_path)


    for scenario_index in range(len(len_record)):
      if scenario_index == 0:
        start_global_timestamp_index = 0
      else:
        start_global_timestamp_index = len_record[scenario_index - 1]
      end_global_timestamp_index = len_record[scenario_index] - 1
      #print('start_global_timestamp_index: ', start_global_timestamp_index)
      #print('end_global_timestamp_index: ', end_global_timestamp_index)
      
      ab3dmot_detection_save_file = os.path.join(ab3dmot_detection_save_path, '%04d.txt' % scenario_index)
      #print('ab3dmot_detection_save_file: ', ab3dmot_detection_save_file)
    
      with open(ab3dmot_detection_save_file, 'w') as f:
        for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1):
          local_timestamp_index = global_timestamp_index - start_global_timestamp_index
          # load v2v4real detection output file
          # reversed operation of 
          # https://github.com/ucla-mobility/V2V4Real/blob/a27925eba5bca69eff241cced4f1d84a224bf6b1/opencood/tools/infrence_utils.py#L116
          #print('global_timestamp_index: ', global_timestamp_index)

          v2v4real_detection_file = os.path.join(npy_save_path, cav_id, '%04d_pred.npy' % global_timestamp_index)
          v2v4real_detection = np.load(v2v4real_detection_file)
          #print('v2v4real_detection.shape: ', v2v4real_detection.shape)
          # (9, 8, 3)
        
          v2v4real_detection_score_file = os.path.join(npy_save_path, cav_id, '%04d_pred_score.npy' % global_timestamp_index)
          v2v4real_detection_score = np.load(v2v4real_detection_score_file)
          #print('v2v4real_detection_score.shape: ', v2v4real_detection_score.shape)
          # (9) 

          if v2v4real_detection.shape[0] > max_num_individual_detections:
            max_num_individual_detections = v2v4real_detection.shape[0]  

        
          # https://github.com/xinshuoweng/AB3DMOT/blob/master/docs/KITTI.md
          # https://github.com/ucla-mobility/V2V4Real/blob/a27925eba5bca69eff241cced4f1d84a224bf6b1/opencood/utils/box_utils.py#L14
          # (N, 8, 3) to (N , [xyz, h, w, l, theta])
          #print('v2v4real_detection: ', v2v4real_detection)
          boxes_3d = box_utils.corner_to_center(v2v4real_detection, order='hwl')
          #print('boxes_3d: ', boxes_3d)
          # (N, [xyz, h, w, l, theta]) to (N, (h, w, l, x, y, z, rot_y))
          boxes_3d = np.concatenate([boxes_3d[:, 3:6], boxes_3d[:, 0:3], boxes_3d[:, 6:7]], axis=1)
          #print('boxes_3d: ', boxes_3d)

          # transform to ab3dmot kitti coordinate system
          # swap y, z
          boxes_3d = np.concatenate([boxes_3d[:, 0:4], boxes_3d[:, 5:6], boxes_3d[:, 4:5], boxes_3d[:, 6:7]], axis=1)
          #print('boxes_3d: ', boxes_3d)


          # For dmstrack, ab3dmot, kitti
          for detection_id in range(v2v4real_detection.shape[0]):
            frame = local_timestamp_index
            type = 2 # type index of Car in ab3dmot KITTI
            box_2d = '0,0,0,0' # ignore 2d box
            score = v2v4real_detection_score[detection_id]
            box_3d = ','.join([str(value) for value in boxes_3d[detection_id]])
            #print('box_3d: ', box_3d)
            alpha = 0 # ignore observation angle
            ab3dmot_detection_string = '%d,%d,%s,%f,%s,%f\n' % (
              local_timestamp_index, type, box_2d, score, box_3d, alpha)
            #print('ab3dmot_detection_string: ', ab3dmot_detection_string)
            f.write(ab3dmot_detection_string)


          # LLM
          # directly save boxes_3d with score as np feature
          co_llm_detection_box_score_save_file = os.path.join(co_llm_detection_save_path, '%04d_detection_box_score.npy' % global_timestamp_index)
          #print('boxes_3d.shape: ', boxes_3d.shape) # (5, 7)
          #print('v2v4real_detection_score.shape: ', v2v4real_detection_score.shape) # (5, )
          box_features = np.concatenate([
            boxes_3d,
            np.expand_dims(v2v4real_detection_score, axis=1)
          ], axis=1)
          #print('box_features.shape: ', box_features.shape) # (5, 8)
          #print('box_features: ', box_features)
          np.save(co_llm_detection_box_score_save_file, box_features)

          co_llm_detection_save_file = os.path.join(co_llm_detection_save_path, '%04d_detection_llm.txt' % global_timestamp_index)
          #print('co_llm_detection_save_file: ', co_llm_detection_save_file)
          with open(co_llm_detection_save_file, 'w') as f_co_llm:
            for detection_id in range(v2v4real_detection.shape[0]):
              # V2V4Real's detection inference and evaluation only has a single Car class
              object_type = 'Car' # type string of Car in ab3dmot label KITTI
              # reduce precision to reduce number of tokens
              score = str(round(v2v4real_detection_score[detection_id], 2))
              # all precision 2
              # box_3d = ','.join([str(round(value, 2)) for value in boxes_3d[detection_id]])
              # precision 1 for first 6 values (h, w, l, x, y, z,), precision 2 for rotation angle and confidence score
              box_3d = ','.join([
                str(round(boxes_3d[detection_id][i], 1)) if i < 6 else str(round(boxes_3d[detection_id][i], 2)) 
                for i in range(len(boxes_3d[detection_id]))
              ])
              #print('box_3d: ', box_3d)
              co_llm_detection_string = '%s,%s,%s\n' % (
                object_type, box_3d, score)
              #print('co_llm_detection_string: ', co_llm_detection_string)
              #assert False
              # coordinate system:
              # https://github.com/eddyhkchiu/DMSTrack/issues/1
              # Right-hand coordinate system,
              # x: forward, y: up, z: right
              # theta: rotation around y-axis
              # (h, w, l, x, y, z, rot_y)
              # Car : [1.6980522, 2.0622315, 3.9799347, -20.500578, -0.978255, -0.10612263, 0.026179945]. 
              f_co_llm.write(co_llm_detection_string)

  print('max_num_individual_detections: ', max_num_individual_detections)
  return



def transform_and_save_tracking_label_to_ab3dmot_format(len_record, npy_save_path):
  # opencood_dataset.len_record
  print('len_record: ', len_record)
  # [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
  
  max_num_gts = 0

  ab3dmot_tracking_label_save_path = os.path.join(npy_save_path, 'ab3dmot_tracking_label')
  if not os.path.exists(ab3dmot_tracking_label_save_path):
    os.makedirs(ab3dmot_tracking_label_save_path)


  # LLM
  co_llm_gt_label_save_path = os.path.join(npy_save_path, 'co_llm') 
  if not os.path.exists(co_llm_gt_label_save_path):
    os.makedirs(co_llm_gt_label_save_path)


  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1
    #print('start_global_timestamp_index: ', start_global_timestamp_index)
    #print('end_global_timestamp_index: ', end_global_timestamp_index)
      
    ab3dmot_tracking_label_save_file = os.path.join(ab3dmot_tracking_label_save_path, '%04d.txt' % scenario_index)
    
    with open(ab3dmot_tracking_label_save_file, 'w') as f:
      # For each frame/teimstamp
      for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1):
        local_timestamp_index = global_timestamp_index - start_global_timestamp_index
        # load v2v4real gt files
        # reversed operation of 
        # https://github.com/ucla-mobility/V2V4Real/blob/a27925eba5bca69eff241cced4f1d84a224bf6b1/opencood/tools/infrence_utils.py#L116
        #print('global_timestamp_index: ', global_timestamp_index)

        v2v4real_gt_file = os.path.join(npy_save_path, '%04d_gt.npy' % global_timestamp_index)
        v2v4real_gt = np.load(v2v4real_gt_file)
        #print('v2v4real_gt.shape: ', v2v4real_gt.shape)
        # (1, 8, 3)
        
        v2v4real_gt_object_id_file = os.path.join(npy_save_path, '%04d_gt_object_id.npy' % global_timestamp_index)
        v2v4real_gt_object_id = np.load(v2v4real_gt_object_id_file)
        #print('v2v4real_gt_object_id.shape: ', v2v4real_gt_object_id.shape)
        # (1,)

        if v2v4real_gt.shape[0] > max_num_gts:
          max_num_gts = v2v4real_gt.shape[0]
        
        # https://github.com/xinshuoweng/AB3DMOT/blob/master/docs/KITTI.md
        # https://github.com/ucla-mobility/V2V4Real/blob/a27925eba5bca69eff241cced4f1d84a224bf6b1/opencood/utils/box_utils.py#L14
        # (N, 8, 3) to (N , [xyz, h, w, l, theta])
        #print('v2v4real_gt: ', v2v4real_gt)
        boxes_3d = box_utils.corner_to_center(v2v4real_gt, order='hwl')
        #print('boxes_3d: ', boxes_3d)
        # (N, [xyz, h, w, l, theta]) to (N, (h, w, l, x, y, z, rot_y))
        boxes_3d = np.concatenate([boxes_3d[:, 3:6], boxes_3d[:, 0:3], boxes_3d[:, 6:7]], axis=1)
        #print('boxes_3d: ', boxes_3d)

        # transform to ab3dmot kitti coordinate system
        # swap y, z
        boxes_3d = np.concatenate([boxes_3d[:, 0:4], boxes_3d[:, 5:6], boxes_3d[:, 4:5], boxes_3d[:, 6:7]], axis=1)
        #print('boxes_3d: ', boxes_3d)
        #assert False

        # dmstrack, ab3dmot, kitti
        # https://github.com/xinshuoweng/AB3DMOT/blob/master/scripts/KITTI/label/0000.txt
        # https://github.com/xinshuoweng/AB3DMOT/blob/master/scripts/KITTI/evaluate.py#L268
        for gt_id in range(v2v4real_gt.shape[0]):
          frame = local_timestamp_index
          track_id = v2v4real_gt_object_id[gt_id]
          object_type = 'Car' # type string of Car in ab3dmot label KITTI
          truncation = 0
          occlusion = 0
          obs_angle = 0
          box_2d = '0 0 0 0' # ignore 2d box
          box_3d = ' '.join([str(value) for value in boxes_3d[gt_id]])
          #print('box_3d: ', box_3d)
          ab3dmot_tracking_label_string = '%d %d %s %d %d %d %s %s\n' % (
            local_timestamp_index, track_id, object_type, truncation, occlusion, obs_angle,
            box_2d, box_3d)
          #print('ab3dmot_tracking_label_string: ', ab3dmot_tracking_label_string)
          f.write(ab3dmot_tracking_label_string)

        # LLM
        co_llm_gt_label_save_file = os.path.join(co_llm_gt_label_save_path, '%04d_gt_llm.txt' % global_timestamp_index)
        with open(co_llm_gt_label_save_file, 'w') as f_co_llm:
          for gt_id in range(v2v4real_gt.shape[0]):
            # V2V4Real's detection inference and evaluation only has a single Car class
            object_type = 'Car' # type string of Car in ab3dmot label KITTI
            # reduce precision to reduce number of tokens
            # all precision 2
            # box_3d = ','.join([str(round(value, 2)) for value in boxes_3d[gt_id]])
            # precision 1 for first 6 values (h, w, l, x, y, z,), precision 2 for rotation angle
            box_3d = ','.join([
              str(round(boxes_3d[gt_id][i], 1)) if i < 6 else str(round(boxes_3d[gt_id][i], 2)) 
              for i in range(len(boxes_3d[gt_id]))
            ])
            #print('box_3d: ', box_3d)
            co_llm_gt_label_string = '%s,%s\n' % (
              object_type, box_3d)
            #print('co_llm_gt_label_string: ', co_llm_gt_label_string)
            #assert False
            # coordinate system:
            # https://github.com/eddyhkchiu/DMSTrack/issues/1
            # Right-hand coordinate system,
            # x: forward, y: up, z: right
            # theta: rotation around y-axis
            # (h, w, l, x, y, z, rot_y)
            # Car : [1.6980522, 2.0622315, 3.9799347, -20.500578, -0.978255, -0.10612263, 0.026179945]. 
            f_co_llm.write(co_llm_gt_label_string)
            #assert False
  print('max_num_gts: ', max_num_gts)
  return


def generate_v2v4real_dataset_for_llava(len_record, npy_save_path, cav_id_set):
  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, 'v2v4real_dataset_for_llava.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_dataset_for_llava.json

  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample)
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1):
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
      data_dict  = {
        'id': global_timestamp_index,
        # we do not have image for now, may extend to point cloud feature map
        #'image': 'image_path',
        'conversations': [{
          'from': 'human',
          #'value': 'Generate the cooperative detection result based on the following individual detection result.\n',
          }, {
          'from': 'gpt',
          #'value': 'The cooperative detection result is\n'
          }
        ],
        # extra fields not required for llava
        'scenario_index': scenario_index,
        'local_timestamp_index': local_timestamp_index,
        'global_timestamp_index': global_timestamp_index
      }
      #print('sources: ', sources)
      human_input = 'Generate the cooperative detection result based on the following individual detection result.\n'
      gpt_output = 'The cooperative detection result is\n'

      for cav_id in cav_id_set:
        individual_detection_file = os.path.join(llm_data_path, cav_id, '%04d_detection_llm.txt' % global_timestamp_index)
        #print('individual_detection_file: ', individual_detection_file)
        with open(individual_detection_file, 'r') as f:
          individual_detection = f.read()
        individual_detection = individual_detection.replace(',', ', ')
        #print('individual_detection: ', individual_detection)
        human_input += 'Agent ' + cav_id + "'s detection result:\n"
        human_input += individual_detection
      #print('human_input: ', human_input)
      data_dict['conversations'][0]['value'] = human_input

      gt_detection_file = os.path.join(llm_data_path, '%04d_gt_llm.txt' % global_timestamp_index)
      #print('gt_detection_file: ', gt_detection_file)
      with open(gt_detection_file, 'r') as f:
        gt_detection = f.read()
      gt_detection = gt_detection.replace(',', ', ')
      #print('gt_detection: ', gt_detection)
      gpt_output += gt_detection
      #print('gpt_output: ', gpt_output)
      data_dict['conversations'][1]['value'] = gpt_output
      #print('data_dict: ', data_dict)

      list_data_dict.append(data_dict)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  return


def load_gt_boxes_3d(npy_save_path, global_timestamp_index, cav_ids):
  v2v4real_gt_file = os.path.join(npy_save_path, '%04d_gt.npy' % global_timestamp_index)
  v2v4real_gt = np.load(v2v4real_gt_file)
  #print('v2v4real_gt.shape: ', v2v4real_gt.shape)
  # (1, 8, 3)
  # swap y,z to transform from v2v4real coordinate system to dmstrack's
  gt_box_corners = np.stack([
    v2v4real_gt[:, :, 0],
    v2v4real_gt[:, :, 2],
    v2v4real_gt[:, :, 1],
  ], axis=2)

  # https://github.com/xinshuoweng/AB3DMOT/blob/master/docs/KITTI.md
  # https://github.com/ucla-mobility/V2V4Real/blob/a27925eba5bca69eff241cced4f1d84a224bf6b1/opencood/utils/box_utils.py#L14
  # (N, 8, 3) to (N , [xyz, h, w, l, theta])
  #print('v2v4real_gt: ', v2v4real_gt)
  boxes_3d = box_utils.corner_to_center(v2v4real_gt, order='hwl')
  #print('boxes_3d: ', boxes_3d)
  # (N, [xyz, h, w, l, theta]) to (N, (h, w, l, x, y, z, rot_y))
  boxes_3d = np.concatenate([boxes_3d[:, 3:6], boxes_3d[:, 0:3], boxes_3d[:, 6:7]], axis=1)
  #print('boxes_3d: ', boxes_3d)

  # transform to ab3dmot kitti coordinate system
  # swap y, z
  boxes_3d = np.concatenate([boxes_3d[:, 0:4], boxes_3d[:, 5:6], boxes_3d[:, 4:5], boxes_3d[:, 6:7]], axis=1)
  #print('boxes_3d: ', boxes_3d)
  #assert False
  gt_boxes = boxes_3d

  #print('gt_boxes[0]: ', gt_boxes[0])
  # [  1.6982422    2.0622       3.97989    -20.500576    -0.97825503
  #   -0.10612261   0.02619088]
  #print('gt_box_corners[0]: ', gt_box_corners[0])
  # [[-18.484312    -1.8273761   -1.0847566 ]
  #  [-18.538317    -1.8273761    0.9767363 ]
  #  [-22.516842    -1.8273761    0.8725114 ]
  #  [-22.462837    -1.8273761   -1.1889815 ]
  #  [-18.484312    -0.12913392  -1.0847566 ]
  #  [-18.538317    -0.12913392   0.9767363 ]
  #  [-22.516842    -0.12913392   0.8725114 ]
  #  [-22.462837    -0.12913392  -1.1889815 ]]


  v2v4real_gt_object_id_file = os.path.join(npy_save_path, '%04d_gt_object_id.npy' % global_timestamp_index)
  v2v4real_gt_object_id = np.load(v2v4real_gt_object_id_file)
  #print('v2v4real_gt_object_id: ', v2v4real_gt_object_id)
  gt_object_ids = v2v4real_gt_object_id

  # visible and invisible gt object ids
  visible_gt_object_ids_dict = dict()
  invisible_gt_object_ids_dict = dict()
  for cav_id in cav_ids:
    visible_gt_object_id_file = os.path.join(npy_save_path, '%04d_gt_object_id_visible_to_%s.npy' % (global_timestamp_index, cav_id))  
    visible_gt_object_id = np.load(visible_gt_object_id_file)
    visible_gt_object_ids_dict[cav_id] = visible_gt_object_id
    invisible_gt_object_id_file = os.path.join(npy_save_path, '%04d_gt_object_id_invisible_to_%s.npy' % (global_timestamp_index, cav_id))  
    invisible_gt_object_id = np.load(invisible_gt_object_id_file)
    invisible_gt_object_ids_dict[cav_id] = invisible_gt_object_id

  #print('visible_gt_object_ids_dict: ', visible_gt_object_ids_dict)
  #print('invisible_gt_object_ids_dict: ', invisible_gt_object_ids_dict)
  #assert False
  return gt_boxes, gt_box_corners, gt_object_ids, visible_gt_object_ids_dict, invisible_gt_object_ids_dict


def generate_3d_grounding_qa_dataset(len_record, npy_save_path, cav_ids):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  For each gt box [h, w, l, x, y, z, a], generate the QA pair:
  Q: What is the object at the location [x, z]? What are its bounding box parameters?
  A: A car is at the location. Its bounding box parameters are [h, w, l, x, y, z, a].
  TODO: generate negative data samples:
  A: There is no object at the location.
  '''
  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, 'v2v4real_3d_grounding_qa_dataset.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset.json

  data_sample_id = 0

  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample)
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1):
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      boxes_3d, _, _, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index)
      #print('boxes_3d: ', boxes_3d)
      #print('boxes_3d.shape: ', boxes_3d.shape)
      # (2, 7)

      for gt_id in range(boxes_3d.shape[0]):
        box_3d_round = [
          round(boxes_3d[gt_id][i], 1) if i < 6 else round(boxes_3d[gt_id][i], 2)
          for i in range(len(boxes_3d[gt_id]))
        ]
        #print('box_3d_round: ', box_3d_round)
        # [1.7, 2.1, 4.0, -20.5, -1.0, -0.1, 0.03]
        # [h, w, l, x, y, z, a]
        x_z_str = '[' + str(box_3d_round[3]) + ', '  + str(box_3d_round[5]) + ']'
        #print(x_z_str)

        box_3d_str = ', '.join([str(value) for value in box_3d_round])
        box_3d_str = '[' + box_3d_str + ']'
        #print('box_3d_str: ', box_3d_str)

        # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
        data_dict  = {
          'id': data_sample_id,
          # we do not have image for now, may extend to point cloud feature map
          #'image': 'image_path',
          'conversations': [{
            'from': 'human',
            #'value': 'What is the object at the location [x, z]? What are its bounding box parameters? \n'
            }, {
            'from': 'gpt',
            #'value': 'A car is at the location. Its bounding box parameters are [h, w, l, x, y, z, a]. \n'
            }
          ],
          # extra fields not required for llava
          'scenario_index': scenario_index,
          'local_timestamp_index': local_timestamp_index,
          'global_timestamp_index': global_timestamp_index
        }
        #print('sources: ', sources)
        human_input = 'What is the object at the location %s? What are its bounding box parameters? \n' % x_z_str
        gpt_output = 'A car is at the location. Its bounding box parameters are %s. \n' % box_3d_str
        #print('human_input: ', human_input)
        data_dict['conversations'][0]['value'] = human_input
        #print('gpt_output: ', gpt_output)
        data_dict['conversations'][1]['value'] = gpt_output
        #print('data_dict: ', data_dict)
        #assert False
        list_data_dict.append(data_dict)
        data_sample_id += 1

  print('Total number of data samples: ', data_sample_id)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  return


def load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids):

  det_box_scores_dict = dict()
  det_box_corners_dict = dict()
  for cav_id in cav_ids:  
    v2v4real_detection_file = os.path.join(npy_save_path, cav_id, '%04d_pred.npy' % global_timestamp_index)
    v2v4real_detection = np.load(v2v4real_detection_file)
    #print('v2v4real_detection.shape: ', v2v4real_detection.shape)
    # (9, 8, 3)
    det_box_corners_dict[cav_id] = v2v4real_detection
    # swap y and z to transform from v2v4real coordinate system to dmstrack coordinate system
    det_box_corners_dict[cav_id] = np.stack([
      det_box_corners_dict[cav_id][:, :, 0],
      det_box_corners_dict[cav_id][:, :, 2],
      det_box_corners_dict[cav_id][:, :, 1]
    ], axis=2)

    v2v4real_detection_score_file = os.path.join(npy_save_path, cav_id, '%04d_pred_score.npy' % global_timestamp_index)
    v2v4real_detection_score = np.load(v2v4real_detection_score_file)
    #print('v2v4real_detection_score.shape: ', v2v4real_detection_score.shape)
    # (9)

    # https://github.com/xinshuoweng/AB3DMOT/blob/master/docs/KITTI.md
    # https://github.com/ucla-mobility/V2V4Real/blob/a27925eba5bca69eff241cced4f1d84a224bf6b1/opencood/utils/box_utils.py#L14
    # (N, 8, 3) to (N , [xyz, h, w, l, theta])
    #print('v2v4real_detection: ', v2v4real_detection)
    boxes_3d = box_utils.corner_to_center(v2v4real_detection, order='hwl')
    #print('boxes_3d: ', boxes_3d)
    # (N, [xyz, h, w, l, theta]) to (N, (h, w, l, x, y, z, rot_y))
    boxes_3d = np.concatenate([boxes_3d[:, 3:6], boxes_3d[:, 0:3], boxes_3d[:, 6:7]], axis=1)
    #print('boxes_3d: ', boxes_3d)

    # transform to ab3dmot kitti coordinate system
    # swap y, z
    boxes_3d = np.concatenate([boxes_3d[:, 0:4], boxes_3d[:, 5:6], boxes_3d[:, 4:5], boxes_3d[:, 6:7]], axis=1)
    #print('boxes_3d: ', boxes_3d)
    #print('boxes_3d.shape: ', boxes_3d.shape)
    # (9, 7)

    boxes_3d_scores = np.concatenate([
        boxes_3d, 
        np.expand_dims(v2v4real_detection_score, axis=1)
    ], axis=1)
    #print('boxes_3d_scores.shape: ', boxes_3d_scores.shape)
    # (9, 8)
    det_box_scores_dict[cav_id] = boxes_3d_scores


  # https://github.com/eddyhkchiu/DMSTrack/issues/1
  # dmstrack coordinate system
  #print('det_box_scores_dict["ego"][0]: ', det_box_scores_dict["ego"][0])
  # [  1.7270508    1.9570807    4.3018713  -20.695312    -1.1833496
  #  -0.09875488   0.04820127   0.55231124]
  # dmstrack coordinate system
  #print('det_box_corners_dict["ego"][0]: ', det_box_corners_dict["ego"][0])
  # [[-18.5         -2.046875    -0.97265625]
  #  [-18.59375     -2.046875     0.9824219 ]
  #  [-22.890625    -2.046875     0.77490234]
  #  [-22.796875    -2.046875    -1.1796875 ]
  #  [-18.5         -0.31982422  -0.97265625]
  #  [-18.59375     -0.31982422   0.9824219 ]
  #  [-22.890625    -0.31982422   0.77490234]
  #  [-22.796875    -0.31982422  -1.1796875 ]]
  return det_box_scores_dict, det_box_corners_dict



def get_sample_locations(gt_boxes, det_box_scores_dict, cav_1_location=None): 
  '''
  Get sample locations [x, z] for 3d grounding questions
    1. gt boxes
    2. each cav's detection boxes
    3. extrapolated locations: the location behind a detection box from a sensor's point of view
      we know cav-ego is at [0, 0], draw a line from [0, 0] to a detection box and extrapolate for 5 meters
    4. TODO: cav-1's extrapolated location. Need to know where the cav-1 is in the same coordinate system

  box: [h, w, l, x, y, z, a]  
  '''
  extra_distance = 5
  gt_locations = np.stack([gt_boxes[:, 3], gt_boxes[:, 5]], axis=1)
  det_ego_locations = np.stack([det_box_scores_dict['ego'][:, 3], det_box_scores_dict['ego'][:, 5]], axis=1)
  det_1_locations = np.stack([det_box_scores_dict['1'][:, 3], det_box_scores_dict['1'][:, 5]], axis=1)

  #print('det_ego_locations: ', det_ego_locations)
  extra_ego_locations = det_ego_locations.copy()
  theta = np.arctan2(det_ego_locations[:, 1], det_ego_locations[:, 0])
  #print('theta: ', theta)
  extra_ego_locations[:, 1] += extra_distance * np.sin(theta)
  extra_ego_locations[:, 0] += extra_distance * np.cos(theta)
  #print('extra_ego_locations: ', extra_ego_locations)

  sample_locations = np.concatenate([
    gt_locations, 
    det_ego_locations, 
    det_1_locations, 
    extra_ego_locations], axis=0)
  #print('sample_locations: ', sample_locations)

  return sample_locations


def get_sample_locations_v3(gt_boxes, det_box_scores_dict, cav_1_location=None): 
  '''
  Get sample locations [x, z] for 3d grounding questions
    1. each cav_ego's detection boxes
    4. TODO: cav_1's detection boxes . Need to know where the cav-1 is in the same coordinate system to generate the answer.

  box: [h, w, l, x, y, z, a]  
  '''
  det_ego_locations = np.stack([det_box_scores_dict['ego'][:, 3], det_box_scores_dict['ego'][:, 5]], axis=1)
  sample_locations = det_ego_locations
  return sample_locations




# https://stackoverflow.com/questions/17136084/checking-if-a-point-is-inside-a-rotated-rectangle
def get_triangle_area(P, A, B):
  '''
  Area = abs( (Bx * Ay - Ax * By) + (Cx * By - Bx * Cy) + (Ax * Cy - Cx * Ay) ) / 2
  Input:
    P: (M, 1, 2)
    A, B : (1, N, 2)
  Output:
    area: (M, N): triangle of P[m], A[n], B[n]
  '''
  C = P
  area = np.abs(
    (B[:, :, 0] * A[:, :, 1] - A[:, :, 0] * B[:, :, 1]) +
    (C[:, :, 0] * B[:, :, 1] - B[:, :, 0] * C[:, :, 1]) +
    (A[:, :, 0] * C[:, :, 1] - C[:, :, 0] * A[:, :, 1])
  ) / 2
  #print('area: ', area)
  #print('area.shape: ', area.shape)
  # (M, N)
  return area




# https://stackoverflow.com/questions/17136084/checking-if-a-point-is-inside-a-rotated-rectangle
def check_point_in_rotated_rectangles(sample_locations, gt_box_corners, gt_boxes):
  '''
  Input:
    both in dmstrack coordinate system
    sample_locations: (M, 2), 2: x, z
    gt_box_corners: (N, 8, 3), 8 corners, 3: x, y, z
    gt_boxes: (N, 7), 7: [h, w, l, x, y, z, a]
  Output:
    point_in_rotated_rectangles: (M) 
      index of gt box that the mth point is inside of. 
      -1 if the point is outside of every gt box
  '''
  # input box can be gt or det
  # it is possible that there is no det box at some frames
  if gt_box_corners.shape[0] == 0:
    return np.zeros(sample_locations.shape[0]) * -1


  #print('sample_locations: ', sample_locations)
  #print('gt_box_corners: ', gt_box_corners)
  #print('gt_boxes: ', gt_boxes)
  #print('sample_locations.shape: ', sample_locations.shape)
  #print('gt_box_corners.shape: ', gt_box_corners.shape)
  
  gt_box_corners_2d = np.stack([
    gt_box_corners[:, :4, 0],
    gt_box_corners[:, :4, 2]
  ], axis=2)
  #print('gt_box_corners_2d: ', gt_box_corners_2d)
  #print('gt_box_corners_2d.shape: ', gt_box_corners_2d.shape)
  # (N, 4, 2)

  P =  np.expand_dims(sample_locations, 1)
  #print('P.shape: ', P.shape)
  # (M, 1, 2)

  A = np.expand_dims(gt_box_corners_2d[:, 0, :], 0)
  #print('A.shape: ', A.shape)
  # (1, N, 2)
  B = np.expand_dims(gt_box_corners_2d[:, 1, :], 0)
  C = np.expand_dims(gt_box_corners_2d[:, 2, :], 0)
  D = np.expand_dims(gt_box_corners_2d[:, 3, :], 0)

  PAB = get_triangle_area(P, A, B)
  PBC = get_triangle_area(P, B, C)
  PCD = get_triangle_area(P, C, D)
  PDA = get_triangle_area(P, D, A)

  total_triangle_area = PAB + PBC + PCD + PDA
  #print('total_triangle_area: ', total_triangle_area)
  #print('total_triangle_area.shape: ', total_triangle_area.shape)
  # (M, N)
  rectangle_area = gt_boxes[:, 1] * gt_boxes[:, 2]
  #print('rectangle_area: ', rectangle_area)
  #print('rectangle_area.shape: ', rectangle_area.shape)
  # (N)

  # Add threshold due to limited numerical precision
  # also allow small tolerance if the point is just outside of the box a little bit
  threshold = 1e-2
  point_in_rotated_rectangles = total_triangle_area <= rectangle_area + threshold
  #print('point_in_rotated_rectangles: ', point_in_rotated_rectangles)
  #print('point_in_rotated_rectangles.shape: ', point_in_rotated_rectangles.shape)
  # (M, N)

  all_outside = np.sum(point_in_rotated_rectangles, axis=1) == 0
  #print('all_outside: ', all_outside)
  point_in_rotated_rectangles = np.argmax(point_in_rotated_rectangles, axis=1)
  #print('point_in_rotated_rectangles: ', point_in_rotated_rectangles)
  point_in_rotated_rectangles[all_outside] = -1
  #print('point_in_rotated_rectangles: ', point_in_rotated_rectangles)
  # (M)

  return point_in_rotated_rectangles


def round_to_str(values, center_only=False):
  '''
  Input:
    values: (2), (7), or (8)
      [x, z], [h, w, l, x, y, z, a] + optional [s]
  Output:
    values_str: string with rounded floating point numbers
  '''
  # for angle, round to 2 decimal digit
  # others, round to 1 decimal digit
  #print('values: ', values)

  if not center_only:
    values_str = [
      str(round(values[i], 1)) if i < 6 else str(round(values[i], 2)) for i in range(len(values))
    ]    
  else:  
    values_str = [
      str(round(values[i], 1)) if i < 6 else str(round(values[i], 2)) for i in [3, 5]
    ]    

  #print('values_str: ', values_str)
  #values_str = '[' + ', '.join(values_str) + ']'
  values_str = '(' + ','.join(values_str) + ')'
  #print('values_str: ', values_str)
  #if center_only:
  #  assert False
  return values_str


def generate_3d_grounding_qa_sample(i, sample_locations, point_in_gt_boxes, point_in_det_boxes, 
    gt_boxes, data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified):
  sample_location = sample_locations[i]
  sample_location = round_to_str(sample_location)
  #print('sample_location: ', sample_location)


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      ###'value': 'What is the object at the location [x, z]? What are its bounding box parameters? \n'
      #'value': 'Is there anything at the location [x, z]?'
      }, {
      'from': 'gpt',
      #'value': 'A car is at the location. Its bounding box parameters are [h, w, l, x, y, z, a]. \n'
      #'value': 'A car is at the location. Its center location is [x, z].'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)

  ###human_input = 'What is the object at the location %s? What are its bounding box parameters?' % sample_location
  human_input = 'Is there anything at the location %s?' % sample_location
  #print('human_input: ', human_input)

  gpt_output = ''
  # Answer depends on whether the sample location is in any gt box
  if point_in_gt_boxes[i] > -1:
    gt_box_str = round_to_str(gt_boxes[point_in_gt_boxes[i]])
    gt_center_str = round_to_str(gt_boxes[point_in_gt_boxes[i]], center_only=True)
    if not simplified:
      gpt_output += 'A car is at the location. Its bounding box parameters are %s.' % gt_box_str
    else:
      gpt_output += 'A car is at the location. Its center location is %s.' % gt_center_str  
    #print('gpt_output: ', gpt_output)

    # Reason depends on whether the sample location is in any det box
    if point_in_det_boxes['ego'][i] > -1 and point_in_det_boxes['1'][i] > -1:
      gpt_output += ' Both connected autonomous vehicles detect it.'
      qa_sub_type = 0
    elif point_in_det_boxes['ego'][i] > -1:
      gpt_output += ' Connected autonomous vehicle ego detects it.'  
      qa_sub_type = 1
    elif point_in_det_boxes['1'][i] > -1:
      gpt_output += ' Connected autonomous vehicle 1 detects it.'  
      qa_sub_type = 2
    else:
      gpt_output += ' None of the connected autonomous vehicles detects it but the merged feature maps indicate an object.'
      qa_sub_type = 3
  else:
    gpt_output += 'There is no object at the location.'
    # Reason depends on whether the sample location is in any det box
    if point_in_det_boxes['ego'][i] > -1 and point_in_det_boxes['1'][i] > -1:
      gpt_output += ' Based on the feature maps, both connected autonomous vehicles have false positive detections.'
      qa_sub_type = 4
    elif point_in_det_boxes['ego'][i] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle ego has false positive detections.'  
      qa_sub_type = 5
    elif point_in_det_boxes['1'][i] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle 1 has false positive detections.'  
      qa_sub_type = 6
    else:
      gpt_output += ' None of the connected autonomous vehicles detects any object at the location.'
      qa_sub_type = 7
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type
  #print('data_dict: ', data_dict)

  return data_dict, qa_sub_type



def generate_3d_grounding_qa_sample_v3(i, sample_locations, point_in_gt_boxes, point_in_det_boxes, 
    gt_boxes, data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index):
  sample_location = sample_locations[i]
  sample_location = round_to_str(sample_location)
  #print('sample_location: ', sample_location)


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'Is there anything behind the object at [x, z]?'
      }, {
      'from': 'gpt',
      #'value': 'Yes, there is a car behind the object. Its bounding box parameters are: [h, w, l, x, y, z, a].'
      #'value': 'No, there is nothing behind the object.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)

  human_input = 'Is there anything behind the object at %s?' % sample_location
  #print('human_input: ', human_input)

  gpt_output = ''
  # Answer depends on whether the sample location is in any gt box
  if point_in_gt_boxes[i] > -1:
    gt_box_str = round_to_str(gt_boxes[point_in_gt_boxes[i]])
    gpt_output += 'Yes, there is a car behind the object. Its bounding box parameters are %s.' % gt_box_str
    #print('gpt_output: ', gpt_output)
    # Reason depends on whether the sample location is in any det box
    if point_in_det_boxes['ego'][i] > -1 and point_in_det_boxes['1'][i] > -1:
      gpt_output += ' Both connected autonomous vehicles detect it.'
      qa_sub_type = 0
    elif point_in_det_boxes['ego'][i] > -1:
      gpt_output += ' Connected autonomous vehicle ego detects it.'  
      qa_sub_type = 1
    elif point_in_det_boxes['1'][i] > -1:
      gpt_output += ' Connected autonomous vehicle 1 detects it.'  
      qa_sub_type = 2
    else:
      gpt_output += ' None of the connected autonomous vehicles detects it but the merged feature maps indicate an object.'
      qa_sub_type = 3
  else:
    gpt_output += 'There is nothing behind the object.'
    # Reason depends on whether the sample location is in any det box
    if point_in_det_boxes['ego'][i] > -1 and point_in_det_boxes['1'][i] > -1:
      gpt_output += ' Based on the feature maps, both connected autonomous vehicles have false positive detections.'
      qa_sub_type = 4
    elif point_in_det_boxes['ego'][i] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle ego has false positive detections.'  
      qa_sub_type = 5
    elif point_in_det_boxes['1'][i] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle 1 has false positive detections.'  
      qa_sub_type = 6
    else:
      gpt_output += ' None of the connected autonomous vehicles detects anything behind the object.'
      qa_sub_type = 7
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type
  #print('data_dict: ', data_dict)

  return data_dict, qa_sub_type


def generate_3d_grounding_qa_sample_v4(sector_region_idx, sample_locations, gt_box_exists_in_regions, matched_gt_box_parameters, merged_reason_point_in_det_boxes_dict, 
    data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified):
  sample_location = sample_locations[sector_region_idx]
  sample_location = round_to_str(sample_location)
  #print('sample_location: ', sample_location)
  #print('matched_gt_box_parameters[sector_region_idx]: ', matched_gt_box_parameters[sector_region_idx])
  # we may have more than 1 matched gt box
  #print('merged_reason_point_in_det_boxes_dict["ego"]: ', merged_reason_point_in_det_boxes_dict["ego"])
  #print('merged_reason_point_in_det_boxes_dict["1"]: ', merged_reason_point_in_det_boxes_dict["1"])

  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'Is there anything behind the object at [x, z]?'
      }, {
      'from': 'gpt',
      #'value': 'Yes, there is a car behind the object. Its bounding box parameters are: [h, w, l, x, y, z, a].'
      #'value': 'Yes, there is a car behind the object. Its bounding center location is [x, z].'
      #'value': 'No, there is nothing behind the object.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)

  human_input = 'Is there anything behind the object at %s?' % sample_location
  #print('human_input: ', human_input)

  qa_sub_type = []

  gpt_output = ''
  # Answer depends on whether any gt box is in the "behind" region
  if gt_box_exists_in_regions[sector_region_idx]:
    #print('matched_gt_box_parameters: ', matched_gt_box_parameters)
    #print('matched_gt_box_parameters[sector_region_idx]: ', matched_gt_box_parameters[sector_region_idx])
    # (K_m=1, 7+1), 7+1: box parameters + dist to sector origin (cav_ego's location)

    gt_centers = np.stack([
      matched_gt_box_parameters[sector_region_idx][:, 3],
      matched_gt_box_parameters[sector_region_idx][:, 5]
    ], axis=1)
    #print('gt_centers.shape: ', gt_centers.shape)
    # (K_m, 2)
    behind_distance = np.linalg.norm(gt_centers - sample_locations[sector_region_idx], axis=1)
    #print('behind_distance: ', behind_distance)
    #print('behind_distance.shape: ', behind_distance.shape)
    # (K_m,)


    gt_box_str = [round_to_str(matched_gt_box_parameters[sector_region_idx][i]) for i in range(matched_gt_box_parameters[sector_region_idx].shape[0])]
    gt_center_str = [round_to_str(matched_gt_box_parameters[sector_region_idx][i], center_only=True) for i in range(matched_gt_box_parameters[sector_region_idx].shape[0])]
    #print('gt_box_str: ', gt_box_str)
    #print('gt_center_str: ', gt_center_str)
    gt_center_str = ', '.join(gt_center_str)
    #print('gt_center_str: ', gt_center_str)

    #if matched_gt_box_parameters[sector_region_idx].shape[0] > 1:
    #  assert False  

    if not simplified:
      print('Not implemented')
      assert False
      gpt_output += 'Yes, there is a car %d meters behind the object. Its bounding box parameters are %s.' % (behind_distance, gt_box_str)
    else:  
      if matched_gt_box_parameters[sector_region_idx].shape[0] == 1:  
        gpt_output += 'Yes, there is a car behind the object. Its center location is %s.' % gt_center_str
        #print('gpt_output: ', gpt_output)
      else:  
        gpt_output += 'Yes, there are cars behind the object. Their center locations are %s.' % gt_center_str  
        #print('gpt_output: ', gpt_output)


    #print('gpt_output: ', gpt_output)
    # Reason depends on whether the merged reason point is in any det box
    # if cav_k detects any one of gt boxes in region, we will say cav_k detects it (them) in the answer
    if matched_gt_box_parameters[sector_region_idx].shape[0] > 1:
      it_or_them = 'them'
    else:
      it_or_them = 'it'

    if np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1) and np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_output += ' Both connected autonomous vehicles detect %s.' % it_or_them
    elif np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1):
      gpt_output += ' Connected autonomous vehicle ego detects %s.' % it_or_them  
    elif np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_output += ' Connected autonomous vehicle 1 detects %s.' % it_or_them  
    else:
      gpt_output += ' None of the connected autonomous vehicles detects %s but the merged feature map detects %s' % (it_or_them, it_or_them)

    # however, we keep track the qa_sub_type info for each box in region
    for i in range(matched_gt_box_parameters[sector_region_idx].shape[0]):
      if merged_reason_point_in_det_boxes_dict['ego'][i] > -1 and merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(0)
      elif merged_reason_point_in_det_boxes_dict['ego'][i] > -1:
        qa_sub_type.append(1)
      elif merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(2)
      else:
        qa_sub_type.append(3)
    #print('qa_sub_type: ', qa_sub_type)    
  else:
    behind_distance = None  
    gpt_output += 'There is nothing behind the object.'
    # Reason depends on whether the merged reason point is in any det box
    # if there is no gt box in region, we only have one merged_reason_point, which is the extrapolated point at idx 0
    if merged_reason_point_in_det_boxes_dict['ego'][0] > -1 and merged_reason_point_in_det_boxes_dict['1'][0] > -1:
      gpt_output += ' Based on the feature maps, both connected autonomous vehicles have false positive detections.'
      qa_sub_type.append(4)
    elif merged_reason_point_in_det_boxes_dict['ego'][0] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle ego has false positive detections.'  
      qa_sub_type.append(5)
    elif merged_reason_point_in_det_boxes_dict['1'][0] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle 1 has false positive detections.'  
      qa_sub_type.append(6)
    else:
      gpt_output += ' None of the connected autonomous vehicles detects anything behind the object.'
      qa_sub_type.append(7)
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type
  data_dict['behind_distance'] = behind_distance.tolist() if behind_distance is not None else None
  #print('data_dict: ', data_dict)

  #if len(matched_gt_box_parameters[sector_region_idx]) > 1:
  #  assert False  
  return data_dict, qa_sub_type, behind_distance



def generate_3d_grounding_qa_sample_v5(sector_region_idx, sample_box_locations, sample_box_region_idx, view_region_names, 
    gt_box_exists_in_regions, matched_gt_box_parameters, merged_reason_point_in_det_boxes_dict, 
    data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified):
  print('sector_region_idx: ', sector_region_idx)  
  print('sample_box_locations: ', sample_box_locations)
  print('sample_box_region_idx: ', sample_box_region_idx)
  print('view_region_names: ', view_region_names)

  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'Is there anything behind the <front left> object?'
      }, {
      'from': 'gpt',
      #'value': 'Yes, there is a car behind the <front left> object. Its bounding box parameters are: [h, w, l, x, y, z, a].'
      #'value': 'Yes, there is a car behind the <front left> object. Its center location is [x, z].'
      #'value': 'No, there is nothing behind the <front left> object.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)

  region_name = view_region_names[sample_box_region_idx[sector_region_idx]]
  human_input = 'Is there anything behind the %s object?' % region_name
  print('human_input: ', human_input)
  sample_box_location = sample_box_locations[sector_region_idx]
  print('sample_box_location: ', sample_box_location)
  sample_box_center_str = round_to_str(np.array([0, 0, 0, sample_box_locations[sector_region_idx][0], 0, sample_box_locations[sector_region_idx][1], 0, 0]), center_only=True)
  print('sample_box_center_str: ', sample_box_center_str)

  qa_sub_type = []

  gpt_output = ''
  # Answer depends on whether any gt box is in the "behind" region
  if gt_box_exists_in_regions[sector_region_idx]:
    print('matched_gt_box_parameters: ', matched_gt_box_parameters)
    print('matched_gt_box_parameters[sector_region_idx]: ', matched_gt_box_parameters[sector_region_idx])
    # (K_m=1, 7+1), 7+1: box parameters + dist to sector origin (cav_ego's location)

    gt_centers = np.stack([
      matched_gt_box_parameters[sector_region_idx][:, 3],
      matched_gt_box_parameters[sector_region_idx][:, 5]
    ], axis=1)
    print('gt_centers.shape: ', gt_centers.shape)
    # (K_m, 2)
    behind_distance = np.linalg.norm(gt_centers - sample_box_locations[sector_region_idx], axis=1)
    print('behind_distance: ', behind_distance)
    print('behind_distance.shape: ', behind_distance.shape)
    # (K_m,)

    gt_box_str = [round_to_str(matched_gt_box_parameters[sector_region_idx][i]) for i in range(matched_gt_box_parameters[sector_region_idx].shape[0])]
    gt_center_str = [round_to_str(matched_gt_box_parameters[sector_region_idx][i], center_only=True) for i in range(matched_gt_box_parameters[sector_region_idx].shape[0])]
    print('gt_box_str: ', gt_box_str)
    print('gt_center_str: ', gt_center_str)
    gt_center_str = ', '.join(gt_center_str)
    print('gt_center_str: ', gt_center_str)

    #if matched_gt_box_parameters[sector_region_idx].shape[0] > 1:
    #  assert False  

    # old code, can be removed
    #behind_distance = np.linalg.norm([matched_gt_box_parameters[sector_region_idx][3]- sample_box_locations[sector_region_idx][0], matched_gt_box_parameters[sector_region_idx][5] - sample_box_locations[sector_region_idx][1]])
    #behind_distance = int(behind_distance)
    #gt_box_str = round_to_str(matched_gt_box_parameters[sector_region_idx])
    #gt_center_str = round_to_str(matched_gt_box_parameters[sector_region_idx], center_only=True)

    if not simplified:
      print('Not implemented')
      assert False    
      gpt_output += 'Yes, there is a car %d meters behind the %s object. Its bounding box parameters are %s.' % (behind_distance, region_name, gt_box_str)
    else:
      if matched_gt_box_parameters[sector_region_idx].shape[0] == 1:  
        gpt_output += 'Yes, there is a car behind the %s object. Its center location is %s.' % (region_name, gt_center_str)
      else:  
        gpt_output += 'Yes, there are cars behind the %s object. Their center locations are %s.' % (region_name, gt_center_str)
        #print('gpt_output: ', gpt_output)
        #assert False
    #print('gpt_output: ', gpt_output)
    #assert False
    
    #print('gpt_output: ', gpt_output)
    # Reason depends on whether the merged reason point is in any det box
    # if cav_k detects any one of gt boxes in region, we will say cav_k detects it (them) in the answer
    if matched_gt_box_parameters[sector_region_idx].shape[0] > 1:
      it_or_them = 'them'
    else:
      it_or_them = 'it'

    if np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1) and np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_output += ' Both connected autonomous vehicles detect %s.' % it_or_them
    elif np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1):
      gpt_output += ' Connected autonomous vehicle ego detects %s.' % it_or_them
    elif np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_output += ' Connected autonomous vehicle 1 detects %s.' % it_or_them
    else:
      gpt_output += ' None of the connected autonomous vehicles detects %s but the merged feature map detects %s' % (it_or_them, it_or_them)

    # however, we keep track the qa_sub_type info for each box in region
    for i in range(matched_gt_box_parameters[sector_region_idx].shape[0]):
      if merged_reason_point_in_det_boxes_dict['ego'][i] > -1 and merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(0)
      elif merged_reason_point_in_det_boxes_dict['ego'][i] > -1:
        qa_sub_type.append(1)
      elif merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(2)
      else:
        qa_sub_type.append(3)
    print('qa_sub_type: ', qa_sub_type)
    #assert False

  else:
    behind_distance = None  
    gpt_output += 'There is nothing behind the %s object.' % region_name
    # Reason depends on whether the merged reason point is in any det box
    # if there is no gt box in region, we only have one merged_reason_point, which is the extrapolated point at idx 0
    if merged_reason_point_in_det_boxes_dict['ego'][0] > -1 and merged_reason_point_in_det_boxes_dict['1'][0] > -1:
      gpt_output += ' Based on the feature maps, both connected autonomous vehicles have false positive detections.'
      qa_sub_type.append(4)
    elif merged_reason_point_in_det_boxes_dict['ego'][0] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle ego has false positive detections.'
      qa_sub_type.append(5)
    elif merged_reason_point_in_det_boxes_dict['1'][0] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle 1 has false positive detections.'
      qa_sub_type.append(6)
    else:
      gpt_output += ' None of the connected autonomous vehicles detects anything behind the object.'
      qa_sub_type.append(7)
  print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type
  data_dict['behind_distance'] = behind_distance.tolist() if behind_distance is not None else None

  region_idx = sample_box_region_idx[sector_region_idx]
  data_dict['region_idx'] = region_idx.item() # np scalar to python int
  #data_dict['sample_box_location'] = sample_box_locations[i]
  data_dict['sample_box_center_str'] = sample_box_center_str
  print('data_dict: ', data_dict)

  #if len(matched_gt_box_parameters[sector_region_idx]) > 1:
  #  assert False
  return data_dict, qa_sub_type, behind_distance, region_idx



def generate_3d_grounding_qa_dataset_v2(len_record, npy_save_path, cav_ids, simplified, debug):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  For each gt box [h, w, l, x, y, z, a], generate the QA pair:
  Q: What is the object at the location [x, z]? What are its bounding box parameters?
  A: A car is at the location. Its bounding box parameters are [h, w, l, x, y, z, a].
  
  simplified:
  A: A car is at the location. Its center position is [x, z].

  TODO: generate negative data samples:
  A: There is no object at the location.
  '''
  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  dataset_prefix = 'v2v4real_3d_grounding_qa_dataset_v2'
  if simplified:
    dataset_prefix += 's'
  if debug:  
    dataset_prefix += 'debug'  

  list_data_dict_save_file = os.path.join(llm_data_path, dataset_prefix + '.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v2.json

  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []
  list_data_dict_save_file_p1 = os.path.join(llm_data_path, dataset_prefix + '_p1.json')
  list_data_dict_save_file_p2 = os.path.join(llm_data_path, dataset_prefix + '_p2.json')
  list_data_dict_save_file_p3 = os.path.join(llm_data_path, dataset_prefix + '_p3.json')
  list_data_dict_save_file_p4 = os.path.join(llm_data_path, dataset_prefix + '_p4.json')


  data_sample_id = 0
  qa_sub_type_counter = np.zeros(8)

  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample)
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1):
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, _, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index)
      print('gt_boxes: ', gt_boxes)
      print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)
      assert False

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)

      sample_locations = get_sample_locations(gt_boxes, det_box_scores_dict)
      #print('sample_locations.shape')
      # (N, 2)

      point_in_gt_boxes = check_point_in_rotated_rectangles(sample_locations, gt_box_corners, gt_boxes)
      #print('point_in_gt_boxes: ', point_in_gt_boxes)
      # (M)

      point_in_det_boxes_dict = dict()
      for cav_id in cav_ids:
        point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
          sample_locations, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
      #print('point_in_det_boxes_dict: ', point_in_det_boxes_dict)

      for i in range(sample_locations.shape[0]):
        qa_sample_data_dict, qa_sub_type = generate_3d_grounding_qa_sample(
            i, sample_locations, point_in_gt_boxes, point_in_det_boxes_dict,
            gt_boxes, data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified)
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        list_data_dict.append(qa_sample_data_dict)
        data_sample_id += 1
        qa_sub_type_counter[qa_sub_type] += 1

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return


def get_extrapolated_locations(sensor_location, sample_locations, extrapolate_dist):
  #print('sensor_location: ', sensor_location)
  #print('sample_locations: ', sample_locations)
  #print('extrapolate_dist: ', extrapolate_dist)
  theta = np.arctan2(sample_locations[:, 1] - sensor_location[1], sample_locations[:, 0] - sensor_location[0])
  #print('theta: ', theta)
  extrapolated_locations = sample_locations.copy()
  extrapolated_locations[:, 1] += extrapolate_dist  * np.sin(theta)
  extrapolated_locations[:, 0] += extrapolate_dist  * np.cos(theta)
  #print('extrapolated_locations: ', extrapolated_locations)
  return extrapolated_locations


def generate_3d_grounding_qa_dataset_v3(len_record, npy_save_path, cav_ids):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  For each cav_ego's detection result box [h, w, l, x, y, z, a], generate the QA pair:
  Q: Is there anything behind the object at [x, z]?
  A: There is a car behind the object. Its bounding box parameters are [h, w, l, x, y, z, a].
  A: There is nothing behind the object.
  '''
  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, 'v2v4real_3d_grounding_qa_dataset_v3.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v3.json

  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []
  list_data_dict_save_file_p1 = os.path.join(llm_data_path, 'v2v4real_3d_grounding_qa_dataset_v3_p1.json')
  list_data_dict_save_file_p2 = os.path.join(llm_data_path, 'v2v4real_3d_grounding_qa_dataset_v3_p2.json')
  list_data_dict_save_file_p3 = os.path.join(llm_data_path, 'v2v4real_3d_grounding_qa_dataset_v3_p3.json')
  list_data_dict_save_file_p4 = os.path.join(llm_data_path, 'v2v4real_3d_grounding_qa_dataset_v3_p4.json')


  data_sample_id = 0
  qa_sub_type_counter = np.zeros(8)

  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample)
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1):
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, _, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)

      # get all cav_ego's detection result box location
      sample_locations = get_sample_locations_v3(gt_boxes, det_box_scores_dict)
      #print('sample_locations.shape')
      # (N, 2)

      # cav_ego's location in dmstrack coordinate system (cav_ego's coordinate system)
      sensor_location = np.array([0, 0])
      extrapolate_dist = 5 # meters
      extrapolated_locations = get_extrapolated_locations(sensor_location, sample_locations, extrapolate_dist)

      point_in_gt_boxes = check_point_in_rotated_rectangles(extrapolated_locations, gt_box_corners, gt_boxes)
      #print('point_in_gt_boxes: ', point_in_gt_boxes)
      # (M)

      point_in_det_boxes_dict = dict()
      for cav_id in cav_ids:
        point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
          sample_locations, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
      #print('point_in_det_boxes_dict: ', point_in_det_boxes_dict)

      for i in range(sample_locations.shape[0]):
        qa_sample_data_dict, qa_sub_type = generate_3d_grounding_qa_sample_v3(
            i, sample_locations, point_in_gt_boxes, point_in_det_boxes_dict,
            gt_boxes, data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index)
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        list_data_dict.append(qa_sample_data_dict)
        data_sample_id += 1
        qa_sub_type_counter[qa_sub_type] += 1

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return


class SectorRegions:
  '''
  Represent an occluded region
  '''
  def __init__(self, origin, min_dists, rotated_min_angles, rotated_max_angles, suggested_rotations):
    self.origin = origin
    # (2, ): [x, z]
    self.min_dists = min_dists
    # (N, )
    self.rotated_min_angles = rotated_min_angles
    # (N, )
    self.rotated_max_angles = rotated_max_angles
    # (N, )
    self.suggested_rotations = suggested_rotations
    # (N, )

    #self.sector_regions = np.concatenate([
    #  origins, min_dists, min_angles, max_angles, suggested_rotations
    #], axis=1)
    #print('self.sector_regions.shape: ', self.sector_regions)
    ## (N, 6) for N sector regions

  def __str__(self):
    print('self.origin: ', self.origin)
    print('self.min_dists: ', self.min_dists)
    print('self.rotated_min_angles: ', self.rotated_min_angles )
    print('self.rotated_max_angles: ', self.rotated_max_angles)
    print('self.suggested_rotations: ', self.suggested_rotations)
    return 'SectorRegions'



def radians_to_quadrants(angles):
  # angles: (4, )  -pi ~ +pi 
  #print('angles: ', angles)
  nagative_mask = angles < 0
  angles_new_range = angles.copy()
  angles_new_range[nagative_mask] += 2 * np.pi
  #print('angles_new_range: ', angles_new_range)
  quadrants = (angles_new_range / (np.pi / 2)).astype(np.int64) + 1
  #print('quadrants: ', quadrants)
  return quadrants


def normalize_angles(angles):
  # input: -2pi ~ +2pi  
  # output: -pi ~ +pi
  #print('angles: ', angles)
  if np.any(angles < -2*np.pi) or np.any(angles > 2*np.pi):
    assert False
  angles_new = angles.copy()
  angles_new[angles < -np.pi] += 2*np.pi  
  angles_new[angles >  np.pi] -= 2*np.pi  
  #print('angles_new: ', angles_new)
  return angles_new


def get_sector_regions_behind_det_box(sensor_location, det_box_scores, det_box_corners):
  #print('sensor_location: ', sensor_location)
  # [x, z]
  #print('det_box_scores: ', det_box_scores)
  # (5, 8)
  #print('det_box_corners: ', det_box_corners)
  #print('det_box_corners.shape: ', det_box_corners.shape)
  # (5, 8, 3) 3: [x, y, z]

  angles = np.arctan2(det_box_corners[:, :4, 2] - sensor_location[1], det_box_corners[:, :4, 0] - sensor_location[0])
  #print('angles: ', angles)
  #print('angles.shape: ', angles.shape)
  # (5, 4) # (num_det_boxes, 4 corners in lower face)

  # For each det box
  # check each corner's quadrant
  # if occupied both 2 and 3
  # calculate the suggested_rotation, one of [0, 90, 180] 
  # to avoid occpupying both 2 and 3
  #  
  # When checking whether a gt box is onside the sector
  # add suggested_rotation on the det 4 corner angles, normalize to -pi to pi
  # get the min and max angle
  # for each gt box, get its center angle
  # add the same suggested_rotation
  # and check whether updated gt_angle is between min and max angle
  # and check the distance
  # to determine if gt box is in the sector region

  # a sector region can be describe by 
  # [min_dist, min_angle, max_angle, suggested_rotation]
  # suggested_rotation: one of [0, 90, 180, 270]
  # to make rotated  min_angle < max_angle
  # rotated all angles not occupy quadrant 2 and 3 at the same time

  # algorithm is non-trivial
  # for loop code is easier
  quadrants = radians_to_quadrants(angles)
  # (num_det_boxes, 4 corners)
  # value: one of [1,2,3,4]
  #print('quadrants: ', quadrants)

  num_dets = det_box_scores.shape[0]
  suggested_rotations = np.zeros(num_dets)
  for i in range(num_dets):
    # if occupied quadrant 2 and 3
    if np.sum(quadrants[i] == 2) > 0 and np.sum(quadrants[i] == 3) > 0:
      if np.sum(quadrants[i] == 1) > 0 and np.sum(quadrants[i] == 4) > 0:
        # impossible occupy all [1,2,3,4]  
        # unless this det box is the sensor cav itself
        assert False
      elif np.sum(quadrants[i] == 1) > 0:
        # quadrant [1,2,3]
        suggested_rotations[i] = np.pi # 180
      elif np.sum(quadrants[i] == 4) > 0:
        suggested_rotation = np.pi / 2 # 90
      else:
        # actually same as above  
        suggested_rotations[i] = np.pi / 2 # 90  
      
      
  #print('suggested_rotations: ', suggested_rotations)    
  rotated_angles = angles + np.expand_dims(suggested_rotations, axis=1)
  #print('rotated_angles: ', rotated_angles)
  rotated_angles = normalize_angles(rotated_angles)
  #print('rotated_angles: ', rotated_angles)

  rotated_min_angles = np.min(rotated_angles, axis=1)
  #print('rotated_min_angles: ', rotated_min_angles)
  rotated_max_angles = np.max(rotated_angles, axis=1)
  #print('rotated_max_angles: ', rotated_max_angles)

  # get the min distance threshold of the sector region
  # using the max dist from det box corners to sensor
  corners_in_2d = np.stack([
    det_box_corners[:, :4, 0],
    det_box_corners[:, :4, 2]
  ], axis=2)
  #print('corners_in_2d: ', corners_in_2d)
  #print('corners_in_2d.shape: ', corners_in_2d.shape)
  # (5, 4, 2): num_det_boxes, 4 corners, [x, z]

  dist = np.linalg.norm(corners_in_2d - sensor_location, axis=2)
  #print('dist: ', dist)
  #print('dist.shape: ', dist.shape)
  region_min_dists = np.max(dist, axis=1)
  #print('region_min_dists: ', region_min_dists)

  sector_regions = SectorRegions(sensor_location, region_min_dists, rotated_min_angles, rotated_max_angles, suggested_rotations)
  #assert False
  return sector_regions  


def get_closest_box_in_sector_regions(sector_regions, gt_boxes):
  '''
  Input:
    sector_regions: SectorRegions M regions
      SectorRegions:
        self.origin = origin
        # (2, ): [x, z]
        self.min_dists = min_dists
        # (M, )
        self.rotated_min_angles = rotated_min_angles
        # (M, )
        self.rotated_max_angles = rotated_max_angles
        # (M, )
        self.suggested_rotations = suggested_rotations
        # (M, )
    gt_boxes: (N, 7)
  Output:
    box_exists_in_regions: (M, ) bool
    box_parameters: (M, 7) can contain dummy zeros if no box in the mth region
  '''
  num_regions = sector_regions.min_dists.shape[0]
  num_box_parameters = 7
  # [h, w, l, x, y, z, a]

  box_exists_in_regions = np.zeros(num_regions).astype(bool)
  box_parameters = np.zeros([num_regions, num_box_parameters])

  #print('sector_regions: ', sector_regions)
  #print('gt_boxes: ', gt_boxes)

  for m in range(num_regions):
    closest_dist = 1e7
    for g in range(gt_boxes.shape[0]):
      gt_box = gt_boxes[g]
      #print('gt_box: ', gt_box)
      gt_box_view_angle = np.arctan2(gt_box[5] - sector_regions.origin[1] , gt_box[3] - sector_regions.origin[0])
      gt_box_dist = np.linalg.norm([gt_box[5] - sector_regions.origin[1] , gt_box[3] - sector_regions.origin[0]])

      #print('gt_box_view_angle: ', gt_box_view_angle)
      rotated_gt_box_view_angle = gt_box_view_angle + sector_regions.suggested_rotations[m] 
      #print('rotated_gt_box_view_angle: ', rotated_gt_box_view_angle)
      # check if this gt is in region
      if (rotated_gt_box_view_angle >= sector_regions.rotated_min_angles[m] and 
          rotated_gt_box_view_angle <= sector_regions.rotated_max_angles[m] and
          gt_box_dist > sector_regions.min_dists[m]):
        # found one in region, check if closest and update
        box_exists_in_regions[m] = True
        # hope to have some data sample hit this, hit
        #assert False
        if gt_box_dist < closest_dist:
          closest_dist = gt_box_dist
          box_parameters[m] = gt_box.copy()
          # hope to have some data sample hit this, hit
          #print('gt_box: ', gt_box)
          #print('rotated_gt_box_view_angle: ', rotated_gt_box_view_angle)
          #print('gt_box_dist: ', gt_box_dist)
          #print('sector_regions: ', sector_regions)
          #print('m: ', m)
          #assert False

  #print('box_exists_in_regions: ', box_exists_in_regions)
  #print('box_parameters: ', box_parameters)
  #if np.any(box_exists_in_regions):
  #  assert False
  return box_exists_in_regions, box_parameters



def get_boxes_in_sector_regions(sector_regions, gt_boxes):
  '''
  Input:
    sector_regions: SectorRegions M regions
      SectorRegions:
        self.origin = origin
        # (2, ): [x, z]
        self.min_dists = min_dists
        # (M, )
        self.rotated_min_angles = rotated_min_angles
        # (M, )
        self.rotated_max_angles = rotated_max_angles
        # (M, )
        self.suggested_rotations = suggested_rotations
        # (M, )
    gt_boxes: (N, 7)
  Output:
    box_exists_in_regions: (M, ) bool
    box_parameters: list of M, each (K_m, 7+1) if K_m boxes in mth region
      7+1: box parameters + distance to sector origin, sort by dist
    gt_ids_in_gt_boxes_in_each_region: list of M, each is a list of gt_id in gt_boxes for gt in region  
  '''
  num_regions = sector_regions.min_dists.shape[0]
  num_box_parameters = 7
  # [h, w, l, x, y, z, a]

  box_exists_in_regions = np.zeros(num_regions).astype(bool)
  #box_parameters = np.zeros([num_regions, num_box_parameters])
  box_parameters = []

  #print('sector_regions: ', sector_regions)
  #print('gt_boxes: ', gt_boxes)

  gt_ids_in_gt_boxes_in_each_region = []

  for m in range(num_regions):
    closest_dist = 1e7
    box_parameters_in_a_region = []
    gt_ids_in_gt_boxes_in_this_region = []

    for g in range(gt_boxes.shape[0]):
      gt_box = gt_boxes[g]
      #print('gt_box: ', gt_box)
      gt_box_view_angle = np.arctan2(gt_box[5] - sector_regions.origin[1] , gt_box[3] - sector_regions.origin[0])
      gt_box_dist = np.linalg.norm([gt_box[5] - sector_regions.origin[1] , gt_box[3] - sector_regions.origin[0]])

      #print('gt_box_view_angle: ', gt_box_view_angle)
      rotated_gt_box_view_angle = gt_box_view_angle + sector_regions.suggested_rotations[m] 
      #print('rotated_gt_box_view_angle: ', rotated_gt_box_view_angle)
      # check if this gt is in region
      if (rotated_gt_box_view_angle >= sector_regions.rotated_min_angles[m] and 
          rotated_gt_box_view_angle <= sector_regions.rotated_max_angles[m] and
          gt_box_dist > sector_regions.min_dists[m]):

        # this gt is in this region
        gt_ids_in_gt_boxes_in_this_region.append(g)

        # found one in region
        box_exists_in_regions[m] = True
        #print('gt_box.shape: ', gt_box.shape)
        #print('gt_box_dist.shape: ', gt_box_dist.shape)
        gt_box_parameters_and_dist = np.concatenate([
          gt_box.copy(),
          np.expand_dims(gt_box_dist, axis=0)
        ])
        #print('gt_box_parameters_and_dist.shape: ', gt_box_parameters_and_dist.shape)
        #print('gt_box_parameters_and_dist: ', gt_box_parameters_and_dist)
        box_parameters_in_a_region.append(gt_box_parameters_and_dist)

    # finish scan all gt for this region
    gt_ids_in_gt_boxes_in_each_region.append(gt_ids_in_gt_boxes_in_this_region)
    # sort by distance
    if len(box_parameters_in_a_region) > 0:
      #print('box_parameters_in_a_region: ', box_parameters_in_a_region)  
      box_parameters_in_a_region = sorted(box_parameters_in_a_region, key=lambda x: x[-1])  
      #print('box_parameters_in_a_region: ', box_parameters_in_a_region)  
      box_parameters_in_a_region = np.stack(box_parameters_in_a_region, axis=0)
      #print('box_parameters_in_a_region.shape: ', box_parameters_in_a_region.shape)
      #if len(box_parameters_in_a_region) > 1:
      #  assert False  

    box_parameters.append(box_parameters_in_a_region)    

  #print('box_exists_in_regions: ', box_exists_in_regions)
  #print('box_parameters: ', box_parameters)
  #if np.any(box_exists_in_regions):
  #  assert False
  return box_exists_in_regions, box_parameters, gt_ids_in_gt_boxes_in_each_region



def generate_3d_grounding_qa_dataset_v4(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  For each cav_ego's detection result box [h, w, l, x, y, z, a], generate the QA pair:
  Q: Is there anything behind the object at [x, z]?
  A: There is a car behind the object. Its bounding box parameters are [h, w, l, x, y, z, a].
  A: There is nothing behind the object.
  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_v4'
  if downsample_negatives:
    # ending with 'v' when using val set stats to downsample
    exp_name = 'v2v4real_3d_grounding_qa_dataset_v4bv'
  if simplified:
    exp_name += 's'
  exp_name += 'm' + str(max_num_answer_objects)  

  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, exp_name + '.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v3.json

  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []
  list_data_dict_save_file_p1 = os.path.join(llm_data_path, exp_name + '_p1.json')
  list_data_dict_save_file_p2 = os.path.join(llm_data_path, exp_name + '_p2.json')
  list_data_dict_save_file_p3 = os.path.join(llm_data_path, exp_name + '_p3.json')
  list_data_dict_save_file_p4 = os.path.join(llm_data_path, exp_name + '_p4.json')


  data_sample_id = 0
  qa_sub_type_counter = np.zeros(8)
  behind_distance_stats = []
  max_num_gts_in_region_stats = 0

  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample)
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1):
      print('global_timestamp_index: ', global_timestamp_index)
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, _, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)

      # get cav_ego's detection result box center locations
      det_ego_locations = np.stack([det_box_scores_dict['ego'][:, 3], det_box_scores_dict['ego'][:, 5]], axis=1)
      sample_locations = det_ego_locations
      #print('det_ego_locations.shape')
      # (N, 2)

      # cav_ego's location in dmstrack coordinate system (cav_ego's coordinate system)
      sensor_location = np.array([0, 0])

      sector_regions = get_sector_regions_behind_det_box(sensor_location, det_box_scores_dict['ego'], det_box_corners_dict['ego'])
      #print('sector_regions: ', sector_regions)
       
      # New approach: get more than one box in sector regions 
      #gt_box_exists_in_regions, matched_gt_box_parameters = get_closest_box_in_sector_regions(sector_regions, gt_boxes)
      gt_box_exists_in_regions, matched_gt_box_parameters = get_boxes_in_sector_regions(sector_regions, gt_boxes, max_num_answer_objects)
      #print('gt_box_exists_in_regions: ', gt_box_exists_in_regions)
      #print('matched_gt_box_parameters: ', matched_gt_box_parameters)
      # negative samples:
      #[False False False False False]
      #[[], [], [], [], []]
      for sector_region_idx in range(gt_box_exists_in_regions.shape[0]):
        if len(matched_gt_box_parameters[sector_region_idx]) > max_num_gts_in_region_stats:  
          max_num_gts_in_region_stats = len(matched_gt_box_parameters[sector_region_idx])  



      # to generate the reason part of the answer
      # generate extrapolated_locations
      # (N=num_cav_det_result_boxes, 2)
      # cav_ego's location in dmstrack coordinate system (cav_ego's coordinate system)
      extrapolate_dist = 5 # meters
      extrapolated_locations = get_extrapolated_locations(sensor_location, det_ego_locations, extrapolate_dist)
      #print('extrapolated_locations: ', extrapolated_locations)
      # (5, 2)


      # For M det_boxes, we have M regions and will generate M QA pairs

      # for loop over each sector region
      num_sector_regions = gt_box_exists_in_regions.shape[0]
      for sector_region_idx in range(num_sector_regions):

        # first, to generate the reason,  
        # generate the "merged_reason_points" by
        # For positive samples, we use matched gt box center
        # For negative samples, we use extrapolated location
        # to check whether the point is inside any det result box

        #print('merged_reason_points: ', merged_reason_points)
        #print('merged_reason_points.shape: ', merged_reason_points.shape)
        # (K_m, 2), K_m: number of answer boxes in this sector region for positive sample
        # (1, 2): one extrapolated_location for negative sample
        if gt_box_exists_in_regions[sector_region_idx]:
          merged_reason_points = matched_gt_box_parameters[sector_region_idx].copy()
          #print('merged_reason_points: ', merged_reason_points)
          # (K_m=1, 8)
          merged_reason_points = np.stack([
            merged_reason_points[:, 3],
            merged_reason_points[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          #print('merged_reason_points.shape: ', merged_reason_points.shape)
        else:    
          merged_reason_points = extrapolated_locations[sector_region_idx].copy()
          merged_reason_points = np.expand_dims(merged_reason_points, axis=0)
          #print('merged_reason_points: ', merged_reason_points)
          #print('merged_reason_points.shape: ', merged_reason_points.shape)

        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        # {'ego': array([2]), '1': array([1])}  detected
        # {'ego': array([-1]), '1': array([-1])} not detected 

      # MY_DEBUG
      #if np.any(gt_box_exists_in_regions):
      #  #pass
      #  assert False
      #else:
      #  continue

        # generate one qa for this section region
      #for i in range(sample_locations.shape[0]):
        qa_sample_data_dict, qa_sub_type, behind_distance = generate_3d_grounding_qa_sample_v4(
            sector_region_idx, sample_locations, gt_box_exists_in_regions, matched_gt_box_parameters, merged_reason_point_in_det_boxes_dict,
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified)
        #assert False
        qa_sub_type = np.array(qa_sub_type)
        if np.any(qa_sub_type < 4):
          if np.any(qa_sub_type == 2):  
            pass  
            #print('qa_sample_data_dict: ', qa_sample_data_dict)
            #assert False

        # Downsample negative samples
        if downsample_negatives:
          if np.all(qa_sub_type >= 4):
            # #pos/#neg
            # single output
            #p = 1.0 * 17859 / 70253
            # multi output train set
            #p = 1.0 * 25700 / 70253
            # multi output val set
            p = 1.0 * 14425 / 21472
            if random.uniform(0, 1) > p:
              # discard this negative example  
              continue


        list_data_dict.append(qa_sample_data_dict)
        data_sample_id += 1

        for i in range(qa_sub_type.shape[0]):
          qa_sub_type_counter[qa_sub_type[i]] += 1
          if qa_sub_type[i] == 2: #  intersting case that needs cav_1
            behind_distance_stats.append(behind_distance[i])  

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

        # end of processing one sector region and generating one qa


  # finish processing all frames
  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  behind_distance_stats = sorted(behind_distance_stats)
  print('behind_distance_stats: ', behind_distance_stats)
  #print('behind_distance_stats[:10]: ', behind_distance_stats[:10])
  #print('behind_distance_stats[-10:]: ', behind_distance_stats[-10:])
  print('max_num_gts_in_region_stats: ', max_num_gts_in_region_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return


def view_angle_to_region_idx(view_angle):
  '''
  x: forward
  y: up
  z: right
  6 sector region ['front', 'front left', 'front right', 'back', 'back left', 'back right']
  view_angle: arctan2(z, x)
  0 front: -30 ~ +30
  1 front left: -90 ~ -30
  2 front right: +30 ~ +90
  3 back: -180 ~ -150 or +150 ~ +180
  4 back left: -150 ~ -90
  5 back right: +90 ~ +150
  '''
  if view_angle >= -np.pi / 6 and view_angle <= np.pi / 6:  
    return 0 # front
  elif view_angle >= -np.pi / 2 and view_angle <= -np.pi / 6:
    return 1 # front left
  elif view_angle >= np.pi / 6 and view_angle <= np.pi / 2:
    return 2 # front right
  elif ((view_angle >= -np.pi and view_angle <= -np.pi * (5/6)) or 
        (view_angle >= np.pi * (5/6) and view_angle <= np.pi)):
    return 3 # back
  elif view_angle >= -np.pi * (5/6) and view_angle <= -np.pi / 2:
    return 4 # back left
  elif view_angle >= np.pi / 2 and view_angle <= np.pi * (5/6):
    return 5 # back right
  else:
    print('view_angle: ', view_angle)
    # 3.1415927
    return 3 # back
    #assert False


def get_closest_det_in_view_regions(det_box_scores, det_box_corners):
  '''
   6 sector region ['front', 'front left', 'front right', 'back', 'back left', 'back right']

  Output:
    sample_box_scores: (N<=6, 8)
    sample_box_corners: (N<=6, 8, 3)
    sample_box_region_idx: (N<=6)

  '''
  #print('det_box_scores.shape: ', det_box_scores.shape)
  # (5, 8)
  #print('det_box_corners.shape: ', det_box_corners.shape)
  # (5, 8, 3)
  view_region_names =  ['front', 'front left', 'front right', 'back', 'back left', 'back right']
  num_view_regions = len(view_region_names)
  num_box_parameters = 7
  num_box_corners = 8
  num_dims = 3 # 3d space
  sample_box_scores = np.zeros([num_view_regions, num_box_parameters+1])
  sample_box_corners = np.zeros([num_view_regions, num_box_corners, num_dims])
  min_dist_in_regions = np.zeros(num_view_regions) + 1e7

  for i in range(det_box_scores.shape[0]):
    #print('det_box_scores[i]: ', det_box_scores[i])
    view_angle = np.arctan2(det_box_scores[i][5], det_box_scores[i][3])
    #print('view_angle: ', view_angle)
    view_region_idx = view_angle_to_region_idx(view_angle)
    #print('view_region_idx: ', view_region_idx)
    dist = np.linalg.norm([det_box_scores[i][5], det_box_scores[i][3]])
    #print('dist: ', dist)
    if dist < min_dist_in_regions[view_region_idx]:
      min_dist_in_regions[view_region_idx] = dist
      sample_box_scores[view_region_idx] = det_box_scores[i].copy()
      sample_box_corners[view_region_idx] = det_box_corners[i].copy()

  #print('sample_box_scores: ', sample_box_scores)
  #print('sample_box_corners: ', sample_box_corners)

  sample_box_region_idx = np.array(range(num_view_regions))
  #print('sample_box_region_idx: ', sample_box_region_idx)
  box_in_region_mask = sample_box_scores[:, 0] != 0
  #print('box_in_region_mask: ', box_in_region_mask)

  sample_box_scores = sample_box_scores[box_in_region_mask]
  sample_box_corners = sample_box_corners[box_in_region_mask]
  sample_box_region_idx = sample_box_region_idx[box_in_region_mask]
  #print('sample_box_scores: ', sample_box_scores)
  #print('sample_box_corners: ', sample_box_corners)
  #print('sample_box_region_idx: ', sample_box_region_idx)

  #assert False

  return sample_box_scores, sample_box_corners, sample_box_region_idx


def generate_3d_grounding_qa_dataset_v5(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  For each cav_ego's closest detection result box [h, w, l, x, y, z, a]
  in each of the 6 sector ['front', 'front left', 'front right', 'back', 'back left', 'back right'] generate the QA pair:
  Q: Is there anything behind the 'front left' object?
  A: There is a car behind the 'front left' object. Its bounding box parameters are [h, w, l, x, y, z, a].
  A: There is nothing behind the 'front left' object.
  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_v5'
  if downsample_negatives:
    exp_name = 'v2v4real_3d_grounding_qa_dataset_v5b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)  


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, exp_name + '.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v5.json

  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []
  list_data_dict_save_file_p1 = os.path.join(llm_data_path, exp_name + '_p1.json')
  list_data_dict_save_file_p2 = os.path.join(llm_data_path, exp_name + '_p2.json')
  list_data_dict_save_file_p3 = os.path.join(llm_data_path, exp_name + '_p3.json')
  list_data_dict_save_file_p4 = os.path.join(llm_data_path, exp_name + '_p4.json')


  data_sample_id = 0

  view_region_names = ['front', 'front left', 'front right', 'back', 'back left', 'back right']
  num_view_regions = len(view_region_names)
  region_idx_counter = np.zeros(num_view_regions)
  qa_sub_type_counter = np.zeros(8)
  behind_distance_stats = []
  max_num_gts_in_region_stats = 0


  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample)
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1):
      print('global_timestamp_index: ', global_timestamp_index)  
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners,_, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      # get cav_ego's closest detection result box in each of the 6 regions as sample locations
      sample_box_scores, sample_box_corners, sample_box_region_idx = get_closest_det_in_view_regions(
        det_box_scores_dict['ego'], det_box_corners_dict['ego'])
      #print('sample_box_scores.shape: ', sample_box_scores.shape)
      #print('sample_box_corners.shape: ', sample_box_corners.shape)
      # (<=6, 8), (<=6, 8, 3)
      sample_box_locations = np.stack([
        sample_box_scores[:, 3],
        sample_box_scores[:, 5],
      ], axis=1)

      # cav_ego's location in dmstrack coordinate system (cav_ego's coordinate system)
      sensor_location = np.array([0, 0])
      # get the behind sector region of each sample det box
      sector_regions = get_sector_regions_behind_det_box(sensor_location, sample_box_scores, sample_box_corners)
      #print('sector_regions: ', sector_regions)

      
      # New approach: get more than one box in sector regions
      #gt_box_exists_in_regions, matched_gt_box_parameters = get_closest_box_in_sector_regions(sector_regions, gt_boxes)
      gt_box_exists_in_regions, matched_gt_box_parameters = get_boxes_in_sector_regions(sector_regions, gt_boxes, max_num_answer_objects)
      #print('gt_box_exists_in_regions: ', gt_box_exists_in_regions)
      #print('matched_gt_box_parameters: ', matched_gt_box_parameters)
      for sector_region_idx in range(gt_box_exists_in_regions.shape[0]):
        if len(matched_gt_box_parameters[sector_region_idx]) > max_num_gts_in_region_stats:
          max_num_gts_in_region_stats = len(matched_gt_box_parameters[sector_region_idx])

      # to generate the reason part of the answer
      # generate extrapolated_locations
      # (N, 2)
      # cav_ego's location in dmstrack coordinate system (cav_ego's coordinate system)
      extrapolate_dist = 5 # meters
      extrapolated_locations = get_extrapolated_locations(sensor_location, sample_box_locations, extrapolate_dist)

      # For M det_boxes, we have M regions and will generate M QA pairs

      # for loop over each sector region
      num_sector_regions = gt_box_exists_in_regions.shape[0]
      for sector_region_idx in range(num_sector_regions):

        # first, to generate the reason part of the answer
        # generate the "merged_reason_points" by
        # For positive samples, we use matched gt box center
        # For negative samples, we use extrapolated location
        # to check whether the point is inside any det result box

        #print('merged_reason_points: ', merged_reason_points)
        #print('merged_reason_points.shape: ', merged_reason_points.shape)
        # (K_m, 2), K_m: number of answer boxes in this sector region for positive sample
        # (1, 2): one extrapolated_location for negative sample
        if gt_box_exists_in_regions[sector_region_idx]:
          merged_reason_points = matched_gt_box_parameters[sector_region_idx].copy()
          #print('merged_reason_points: ', merged_reason_points)
          # (K_m=1, 8)
          merged_reason_points = np.stack([
            merged_reason_points[:, 3],
            merged_reason_points[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          #print('merged_reason_points.shape: ', merged_reason_points.shape)
        else:
          merged_reason_points = extrapolated_locations[sector_region_idx].copy()
          merged_reason_points = np.expand_dims(merged_reason_points, axis=0)
          #print('merged_reason_points: ', merged_reason_points)
          #print('merged_reason_points.shape: ', merged_reason_points.shape)

        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        # {'ego': array([2]), '1': array([1])}  detected
        # {'ego': array([-1]), '1': array([-1])} not detected

      # MY_DEBUG
      #if np.any(gt_box_exists_in_regions):
      #  pass
        #assert False
      #else:
        #continue

        # generate one qa for this section region
      #for i in range(sample_box_region_idx.shape[0]):
        qa_sample_data_dict, qa_sub_type, behind_distance, region_idx = generate_3d_grounding_qa_sample_v5(
            sector_region_idx, sample_box_locations, sample_box_region_idx, view_region_names, 
            gt_box_exists_in_regions, matched_gt_box_parameters, merged_reason_point_in_det_boxes_dict,
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified)
        #assert False
        qa_sub_type = np.array(qa_sub_type)
        if np.any(qa_sub_type < 4):
          if np.any(qa_sub_type == 2):
            pass
            #print('qa_sample_data_dict: ', qa_sample_data_dict)
            #assert False

        # Downsample negative samples
        if downsample_negatives:
          if np.all(qa_sub_type >= 4):
            # original  single output stats 
            #p = 1.0 * 7197 / 19991
            # current multi output train stats
            p = 1.0 * 11402 / 19991
            if random.uniform(0, 1) > p:
              # discard this negative example  
              continue

        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)

        data_sample_id += 1
        for i in range(qa_sub_type.shape[0]):
          qa_sub_type_counter[qa_sub_type[i]] += 1
          if qa_sub_type[i] == 2: #  intersting case that needs cav_1
            behind_distance_stats.append(behind_distance[i])
        if np.any(qa_sub_type == 2):
          region_idx_counter[region_idx] += 1

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)
        # end of processing one sector region and generating one qa


  # finish processing all frames
  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  behind_distance_stats = sorted(behind_distance_stats)
  print('behind_distance_stats: ', behind_distance_stats)
  #print('behind_distance_stats[:10]: ', behind_distance_stats[:10])
  #print('behind_distance_stats[-10:]: ', behind_distance_stats[-10:])
  print('region_idx_counter: ', region_idx_counter)
  print('max_num_gts_in_region_stats: ', max_num_gts_in_region_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return


def get_cav_ego_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames):
  '''
  Get cav_ego's future num_future_frames  gt trajectory in the current frame's coordinate system
  '''

  cav_ego_future_trajectory = np.zeros([num_future_frames, 2])

  #print('npy_save_path: ', npy_save_path)
  #print('global_timestamp_index: ', global_timestamp_index)
  #print('start_global_timestamp_index: ', start_global_timestamp_index)
  #print('end_global_timestamp_index: ', end_global_timestamp_index)

  initial_lidar_pose = np.load(os.path.join(npy_save_path, 'ego', '%04d_lidar_pose.npy' % global_timestamp_index))

  for i in range(global_timestamp_index+1, global_timestamp_index + num_future_frames + 1):
    lidar_pose = np.load(os.path.join(npy_save_path, 'ego', '%04d_lidar_pose.npy' % i))
    # v2v4real coordinate, left hand: x forward, y right, z up
    #print('lidar_pose: ', lidar_pose)

    lidar_pose_in_initial_frame = x1_to_x2(lidar_pose, initial_lidar_pose)
    #print('lidar_pose_in_initial_frame: ', lidar_pose_in_initial_frame)

    location_2d = np.array([lidar_pose_in_initial_frame[0, 3], lidar_pose_in_initial_frame[1, 3]])
    # v2v4real [x, y]
    #print('location_2d: ', location_2d)
    # dmstrack coordinate, right hand: x forward, y up, z right
    # dmstrack [x, z]
    cav_ego_future_trajectory[i - global_timestamp_index - 1] = location_2d


  #print('cav_ego_future_trajectory: ', cav_ego_future_trajectory)
  #assert False
  return cav_ego_future_trajectory



def get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids):
  '''

  Output:
    double_cavs_future_trajectory_in_ego_current: 
      dict['ego']: cav_ego's future traj in cav_ego's current coordinate system
      dict['1']: cav_1's future traj in cav_ego's current coordinate system
    double_cavs_future_trajectory_in_self_current  
      dict['ego']: cav_ego's future traj in cav_ego's current coordinate system
      dict['1']: cav_1's future traj in cav_1's current coordinate system
    initial_lidar_pose
      each lidar_pose in current world coordinate system
  '''
  double_cavs_future_trajectory_in_ego_current = {
    'ego': np.zeros([num_future_frames, 2]),
    '1': np.zeros([num_future_frames, 2]),
  }
  double_cavs_future_trajectory_in_self_current = {
    'ego': np.zeros([num_future_frames, 2]),
    '1': np.zeros([num_future_frames, 2]),
  }

  initial_lidar_pose = dict()
  for cav_id in cav_ids:
    initial_lidar_pose[cav_id] = np.load(os.path.join(npy_save_path, cav_id, '%04d_lidar_pose.npy' % global_timestamp_index))

  for i in range(global_timestamp_index+1, global_timestamp_index + num_future_frames + 1):
    for cav_id in cav_ids:
      lidar_pose_in_world = np.load(os.path.join(npy_save_path, cav_id, '%04d_lidar_pose.npy' % i))
      # v2v4real coordinate, left hand: x forward, y right, z up
      #print('lidar_pose_in_world: ', lidar_pose_in_world)

      # double_cavs_future_trajectory_in_ego_current
      lidar_pose_in_ego_initial_frame = x1_to_x2(lidar_pose_in_world, initial_lidar_pose['ego'])
      #print('lidar_pose_in_ego_initial_frame: ', lidar_pose_in_ego_initial_frame)
      location_2d_in_ego_initial_frame = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
      # v2v4real [x, y]
      #print('location_2d_in_ego_initial_frame: ', location_2d_in_ego_initial_frame)
      # dmstrack coordinate, right hand: x forward, y up, z right
      # dmstrack [x, z]
      double_cavs_future_trajectory_in_ego_current[cav_id][i - global_timestamp_index - 1] = location_2d_in_ego_initial_frame
 

      # double_cavs_future_trajectory_in_self_current
      lidar_pose_in_self_initial_frame = x1_to_x2(lidar_pose_in_world, initial_lidar_pose[cav_id])
      #print('lidar_pose_in_self_initial_frame: ', lidar_pose_in_self_initial_frame)
      location_2d_in_self_initial_frame = np.array([lidar_pose_in_self_initial_frame[0, 3], lidar_pose_in_self_initial_frame[1, 3]])
      # v2v4real [x, y]
      #print('location_2d_in_self_initial_frame: ', location_2d_in_self_initial_frame)
      # dmstrack coordinate, right hand: x forward, y up, z right
      # dmstrack [x, z]
      double_cavs_future_trajectory_in_self_current[cav_id][i - global_timestamp_index - 1] = location_2d_in_self_initial_frame

  #print('double_cavs_future_trajectory_in_ego_current: ', double_cavs_future_trajectory_in_ego_current)
  #print('double_cavs_future_trajectory_in_self_current: ', double_cavs_future_trajectory_in_self_current)
  #assert False
  return double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose


def get_notable_gts_near_cav_future_trajectory(cav_ego_future_trajectory, gt_boxes, gt_box_corners, gt_object_ids, max_num_answer_objects, asker_initial_location):
  '''
  notable gts:
    close: gt distance to way points < 10 meters

    (optional)
    collide: gt contains way points

  get the first 1 gts for simplicity for initial dev
  get more than 1 notable objects
  '''
  # only generate 3 output notable_objects
  # to make the qa easier of llm
  #max_num_answer_objects = 3 or 50 if unlimited
  close_distance_threshold = 10

  #print('cav_ego_future_trajectory: ', cav_ego_future_trajectory)
  #print('gt_boxes: ', gt_boxes)
  #print('gt_box_corners: ', gt_box_corners)

  gt_locations = np.stack([
    gt_boxes[:, 3],
    gt_boxes[:, 5],
  ], axis=1)
  #print('gt_locations: ', gt_locations)
  # (N, 2)
 
  notable_gts_idx = dict()
  for t in range(cav_ego_future_trajectory.shape[0]):
    gt_waypoint_dist = np.linalg.norm(gt_locations - cav_ego_future_trajectory[t], axis=1, keepdims=True)
    #print('gt_waypoint_dist: ', gt_waypoint_dist)
    # 2025 0425
    # gt_original_index is used only for deduplicate
    # replace it with gt_object_ids
    #gt_original_index = np.expand_dims(np.array(range(gt_boxes.shape[0])), axis=1)
    gt_original_index = np.expand_dims(gt_object_ids, axis=1)
    #print('gt_original_index: ', gt_original_index)
    gt_waypoint_dist = np.concatenate([
      gt_boxes,
      gt_original_index,
      np.ones([gt_boxes.shape[0], 1]) * t,
      gt_waypoint_dist
    ], axis=1)
    #print('gt_waypoint_dist: ', gt_waypoint_dist)
    # (N, 7+1+1+1): box parameters, original index, future_time, distance
    #assert False

    # sort by distance
    gt_waypoint_dist = sorted(gt_waypoint_dist, key=lambda x: x[-1])
    #print('gt_waypoint_dist: ', gt_waypoint_dist)
    # list of np array (8,)

    for i in range(len(gt_waypoint_dist)):
      #print('t: ', t)
      #print('i: ', i)
      #print('gt_waypoint_dist[i]: ', gt_waypoint_dist[i])  
      if gt_waypoint_dist[i][-1] < close_distance_threshold:
        #print('< close_distance_threshold')  
        #print('cav_ego_future_trajectory[t]: ', cav_ego_future_trajectory[t])
        # ignore gt whose location is very close to [0, 0]
        # [0, 0] is location of cav_ego
        # new
        # avoid counting cav itself as notable object
        #print('gt_waypoint_dist[i]: ', gt_waypoint_dist[i])  
        #print('asker_initial_location: ', asker_initial_location)
        #dist = np.linalg.norm([gt_waypoint_dist[i][3]-asker_initial_location[0], gt_waypoint_dist[i][5]]-asker_initial_location[1])
        #print('dist: ', dist)
        if np.linalg.norm([gt_waypoint_dist[i][3]-asker_initial_location[0], gt_waypoint_dist[i][5]]-asker_initial_location[1]) < 2:
          #print('very close to origin')  
          #print('gt_waypoint_dist[i]: ', gt_waypoint_dist[i])  
          #print('asker_initial_location: ', asker_initial_location)
          #if asker_initial_location[0] > 1 or asker_initial_location[0] < -1:
          #  assert False
          continue
        # new
        # ignore object behind cav itself 
        if gt_waypoint_dist[i][3] < asker_initial_location[0]:
          #print('behind asker cav')
          #print('gt_waypoint_dist[i]: ', gt_waypoint_dist[i])  
          #print('asker_initial_location: ', asker_initial_location)
          #assert False
          continue  

        # deduplicate using original index
        if gt_waypoint_dist[i][-3] in notable_gts_idx:
          #print('gt has been included earlier')  
          # if this gt has been included in notable_gts earlier
          # if this gt's distance_to_waypoint is larger than before, skip this
          if gt_waypoint_dist[i][-1] >= notable_gts_idx[gt_waypoint_dist[i][-3]][-1]:
            #print('skip duplicate due to larger distance')  
            continue  
          else: # replace the existing one
            notable_gts_idx[gt_waypoint_dist[i][-3]] = gt_waypoint_dist[i]  

        else: # this gt has not been included before     
          #print('include this gt')
          #print('gt_waypoint_dist[i]: ', gt_waypoint_dist[i])
          #print('notable_gts_idx: ', notable_gts_idx)
          #notable_gts.append(gt_waypoint_dist[i])
          # deduplicate using original index
          notable_gts_idx[gt_waypoint_dist[i][-3]] = gt_waypoint_dist[i]  
          #print('notable_gts_idx: ', notable_gts_idx)
          #print('gt_waypoint_dist[i]: ', gt_waypoint_dist[i])  
          #print('asker_initial_location: ', asker_initial_location)
          #dist = np.linalg.norm([gt_waypoint_dist[i][3]-asker_initial_location[0], gt_waypoint_dist[i][5]]-asker_initial_location[1])
          #print('dist: ', dist)
          #assert False

        # do not stop here
        #if len(notable_gts_idx) >= max_num_answer_objects:
        #  break
      else:
        break  

    # do not stop here
    #if len(notable_gts_idx) >= max_num_answer_objects:
    #  break


  if len(notable_gts_idx) == 0:
    return None 


  # get the notable_gts by notable_gts_idx's values
  #print('notable_gts_idx: ', notable_gts_idx)
  notable_gts = list(notable_gts_idx.values())
  #print('notable_gts: ', notable_gts)


  #another approach is sort by distance to closes waypoint
  # to get the subset, then sort by distance to cav_ego
  notable_gts = sorted(notable_gts, key=lambda x: x[-1])

  # only return at most max_num_answer_objects 
  if len(notable_gts) > max_num_answer_objects:
    notable_gts = notable_gts[:max_num_answer_objects]    

  # sort by distance to cav_ego current position [0, 0]
  # TODO: better to sort by dist to asker cav's current position
  notable_gts = sorted(notable_gts, key=lambda x: x[3]**2 + x[5]**2)
  #print('notable_gts: ', notable_gts)


  notable_gts = np.stack(notable_gts)
  # need deduplicate
  #print('notable_gts: ', notable_gts)
  #  (M, 7 + 1 + 1 + 1) # box_parameters, gt_object_id, future_time, distance_to_waypoint
  #if notable_gts.shape[0] > 1:
  #  print('notable_gts: ', notable_gts)
    #if asker_initial_location[0] < -7:
    #  assert False
  #assert False  
  return notable_gts



def get_future_trajectory_str(cav_ego_future_trajectory, num_future_waypoints):
  '''
  From the input 30 frames waypoint for future 3 seconds

  If num_future_waypoints == 3:
    get the 3 waypoint after each future one second
    at 9, 19, 29
    in string format round to 1 digit after decimal point
    '[(x0, z0), (x1, z1), (x2, z2)]'

  If num_future_waypoints == 6:  
    get the 3 waypoint after each future one second
    at 4, 9, 14, 19, 24, 29
    in string format round to 1 digit after decimal point
    '[(x0, z0), (x1, z1), (x2, z2), (x3, z3), (x4, z4), (x5, z5)]'
  '''
  if num_future_waypoints == 0:
    future_trajectory_str = ''
  elif num_future_waypoints == 1:
    future_trajectory = [
      cav_ego_future_trajectory[29, 0],
      cav_ego_future_trajectory[29, 1]
    ]
    #print('future_trajectory: ', future_trajectory)
    future_trajectory = [
      str(round(v, 1)) for v in future_trajectory
    ]
    #print('future_trajectory: ', future_trajectory)
    future_trajectory_str = '[(' + future_trajectory[0] + ',' + future_trajectory[1] + ')]'


  elif num_future_waypoints == 3:

    future_trajectory = [
      cav_ego_future_trajectory[9, 0],
      cav_ego_future_trajectory[9, 1],
      cav_ego_future_trajectory[19, 0],
      cav_ego_future_trajectory[19, 1],
      cav_ego_future_trajectory[29, 0],
      cav_ego_future_trajectory[29, 1]
    ]
    #print('future_trajectory: ', future_trajectory)

    future_trajectory = [
      str(round(v, 1)) for v in future_trajectory
    ]
    #print('future_trajectory: ', future_trajectory)

    future_trajectory_str = '[(' + future_trajectory[0] + ',' + future_trajectory[1] + '),' + \
      '(' + future_trajectory[2] + ',' + future_trajectory[3] + '),' + \
      '(' + future_trajectory[4] + ',' + future_trajectory[5] + ')]'

  elif num_future_waypoints == 6:    
    future_trajectory = [
      cav_ego_future_trajectory[4, 0],
      cav_ego_future_trajectory[4, 1],
      cav_ego_future_trajectory[9, 0],
      cav_ego_future_trajectory[9, 1],
      cav_ego_future_trajectory[14, 0],
      cav_ego_future_trajectory[14, 1],
      cav_ego_future_trajectory[19, 0],
      cav_ego_future_trajectory[19, 1],
      cav_ego_future_trajectory[24, 0],
      cav_ego_future_trajectory[24, 1],
      cav_ego_future_trajectory[29, 0],
      cav_ego_future_trajectory[29, 1]
    ]
    #print('future_trajectory: ', future_trajectory)

    future_trajectory = [
      str(round(v, 1)) for v in future_trajectory
    ]
    #print('future_trajectory: ', future_trajectory)

    future_trajectory_str = '[(' + future_trajectory[0] + ',' + future_trajectory[1] + '),' + \
      '(' + future_trajectory[2] + ',' + future_trajectory[3] + '),' + \
      '(' + future_trajectory[4] + ',' + future_trajectory[5] + '),' + \
      '(' + future_trajectory[6] + ',' + future_trajectory[7] + '),' + \
      '(' + future_trajectory[8] + ',' + future_trajectory[9] + '),' + \
      '(' + future_trajectory[10] + ',' + future_trajectory[11] + ')]'

  else:
    assert False  

  #print('future_trajectory_str: ', future_trajectory_str)
  return future_trajectory_str


def get_suggested_speed_steering(cav_current_location_in_self, cav_self_future_trajectory, num_future_waypoints):
  '''
  From the input 30 frames waypoint for future 3 seconds

  If num_future_waypoints == 6:  
    get the 6 waypoint after each future 0.5 second
    at 4, 9, 14, 19, 24, 29
    in string format round to 1 digit after decimal point
    '[(x0, z0), (x1, z1), (x2, z2), (x3, z3), (x4, z4), (x5, z5)]'

  Then get the diff between consecutive waypoints from the above 6 waypoints
  (x0 - 0, z0 - 0), (x1 - x0, z1 - z0), ...
  Then get the mean of above 6 diffs, 
  Calculate the dist and angle of the mean diff 2d vector
  Then classify to one of 5 speed classes and one of 5 steering classes
  Speed: (dist in meters for 0.5 seconds)
    8 ~ inf: fast1: 'fast'
    4 ~ 8: moderate: 'moderate'
    2 ~ 4: slow1: 'slow'
    0.1 ~ 2: slow2: 'very slow'
    ~ 0.1: stop: 'stop'
  Steering: (angle diff in degrees for 0.5 seconds) 
    -infinity ~ -15: turn left 2: 'left'
    -5 ~ -15: turn left 1: 'slightly left'
    -5 ~ 5: straight: 'straight'
    +5 ~ +15: turn right 1: 'slightly right'
    +15 ~ infinity: turn right 2: 'right'

  Output:
    suggested_speed_idx
    suggested_steering_idx
  '''
  assert(num_future_waypoints == 6)
  #print('cav_self_future_trajectory: ', cav_self_future_trajectory)
  # (30, 2)

  waypoints = np.array([
    cav_self_future_trajectory[4],
    cav_self_future_trajectory[9],
    cav_self_future_trajectory[14],
    cav_self_future_trajectory[19],
    cav_self_future_trajectory[24],
    cav_self_future_trajectory[29]
  ])    
  #print('waypoints: ', waypoints)

  waypoints_diff = np.array([
    waypoints[0] - cav_current_location_in_self,
    waypoints[1] - waypoints[0],
    waypoints[2] - waypoints[1],
    waypoints[3] - waypoints[2],
    waypoints[4] - waypoints[3],
    waypoints[5] - waypoints[4]
  ])
  #print('waypoints_diff: ', waypoints_diff)
  # (6, 2)

  waypoints_diff_mean = np.mean(waypoints_diff, axis=0)
  #print('waypoints_diff_mean: ', waypoints_diff_mean)
  # (2, )

  dist = np.linalg.norm(waypoints_diff_mean)
  angle = np.arctan2(waypoints_diff_mean[1], waypoints_diff_mean[0])
  angle *= 180 / np.pi
  #print('dist: ', dist) # 8.18
  #print('angle: ', angle) # 11.1

  # Speed
  if dist > 8: 
    sugested_speed_idx = 0
  elif dist > 4: 
    sugested_speed_idx = 1
  elif dist > 2: 
    sugested_speed_idx = 2
  #elif dist > 0.01:  
  elif dist > 0.1:  
  #elif dist > 0.5:  
    sugested_speed_idx = 3
  else: # stop 
    sugested_speed_idx = 4

  
  # Steering # 0.5 meters per 0.5 second is human walking speed
  if dist < 0.5 or (angle <= 5 and angle >= -5):
    # straight  
    suggested_steering_idx = 2  
  elif angle <= -15:
    # left  
    suggested_steering_idx = 0  
  elif angle > -15 and angle < -5:
    # slightly left  
    suggested_steering_idx = 1  
  elif angle > 5 and angle < 15:
    # slightly right
    suggested_steering_idx = 3
  elif angle >= 15:
    # right
    suggested_steering_idx = 4
  else:  
    # straight  
    suggested_steering_idx = 2  

  #print('sugested_speed_idx: ', sugested_speed_idx)
  #print('suggested_steering_idx: ', suggested_steering_idx)
  #assert False
  return sugested_speed_idx, suggested_steering_idx, dist, angle



def generate_3d_grounding_qa_sample_v6(
  notable_gts, cav_ego_future_trajectory,
  merged_reason_point_in_det_boxes_dict,
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified):


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'Is there anything I need to be aware of if my planned future trajectory is [(x0, z0), (x1, z1), (x2, z2)]?'
      }, {
      'from': 'gpt',
      #'value': 'Yes. There is a car at [x, z], which is close to your planned future trajectory.'
      #'value': 'There is nothing you need to be aware of.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  num_future_waypoints = 3
  future_trajectory_str = get_future_trajectory_str(cav_ego_future_trajectory, num_future_waypoints)
  human_input = 'Is there anything I need to be aware of if my planned future trajectory is %s?' % future_trajectory_str
  #print('human_input: ', human_input)

  #print('notable_gts: ', notable_gts)
  # None or (1, 7+1): box parameter + distance to waypoint

  qa_sub_type = []

  gpt_output = ''
  # Answer depends on whether there is notable_gts
  if notable_gts is not None:
    distance_to_waypoint = notable_gts[:, -1]
    future_time = notable_gts[:, -2]
    gt_box_str = [round_to_str(notable_gts[i]) for i in range(notable_gts.shape[0])]
    gt_center_str = [round_to_str(notable_gts[i], center_only=True) for i in range(notable_gts.shape[0])]
    #print('notable_gts: ', notable_gts)
    #print('gt_box_str: ', gt_box_str)
    #print('gt_center_str: ', gt_center_str)
    gt_center_str = ', '.join(gt_center_str)  
    #print('gt_center_str: ', gt_center_str)
    #if notable_gts.shape[0] > 1:
    #  assert False

    if not simplified:
      print('Not implemented')  
      assert False  
      gpt_output += 'Yes, there is a car close to your planned future trajectory. Its bounding box parameters are %s.' % (gt_box_str)
    else:
      if notable_gts.shape[0] == 1:  
        gpt_output += 'Yes, there is a car at %s, which is close to your planned future trajectory.' % (gt_center_str)
      else:
        gpt_output += 'Yes, there are cars at %s, which are close to your planned future trajectory.' % (gt_center_str)
        #print('gpt_output: ', gpt_output)
        #assert False  


    #print('gpt_output: ', gpt_output)
    #assert False

    #print('gpt_output: ', gpt_output)
    # Reason depends on whether the merged reason point is in any det box
    # TODO: set i = 0, use first notable_gts to generate reason, fix this part later
    # if cav_k detects any one of notable_gts, we will say cav_k detects it (them) in the answer
    #print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
    if notable_gts.shape[0] > 1:
      it_or_them = 'them'
    else:  
      it_or_them = 'it'  

    if np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1) and np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_output += ' Both connected autonomous vehicles detect %s.' % it_or_them
    elif np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1):
      gpt_output += ' Connected autonomous vehicle ego detects %s.' % it_or_them 
    elif np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_output += ' Connected autonomous vehicle 1 detects %s.' % it_or_them 
    else:
      gpt_output += ' None of the connected autonomous vehicles detects %s but the merged feature map detects %s.' % (it_or_them, it_or_them)

    # however, we keep track the qa_sub_type info for each notable_gt
    for i in range(notable_gts.shape[0]):
      if merged_reason_point_in_det_boxes_dict['ego'][i] > -1 and merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(0)
      elif merged_reason_point_in_det_boxes_dict['ego'][i] > -1:
        qa_sub_type.append(1)
      elif merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(2)
      else:
        qa_sub_type.append(3)

  else: # notable_gts is None
    distance_to_waypoint = None
    future_time = None
    gpt_output += 'There is nothing you need to be aware of.'
    # Reason depends on whether the merged reason point is in any det box
    # TODO: set i = 0, use first notable_gts to generate reason, fix this part later
    i = 0
    if merged_reason_point_in_det_boxes_dict['ego'][i] > -1 and merged_reason_point_in_det_boxes_dict['1'][i] > -1:
      gpt_output += ' Based on the feature maps, both connected autonomous vehicles have false positive detections.'
      qa_sub_type.append(4)
    elif merged_reason_point_in_det_boxes_dict['ego'][i] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle ego has false positive detections.'  
      qa_sub_type.append(5)
    elif merged_reason_point_in_det_boxes_dict['1'][i] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle 1 has false positive detections.'  
      qa_sub_type.append(6)
    else:
      gpt_output += ' None of the connected autonomous vehicles detects anything near your planned future trajectory.'
      qa_sub_type.append(7)
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['distance_to_waypoint'] = distance_to_waypoint.tolist() if distance_to_waypoint is not None else None
  data_dict['future_time'] = future_time.tolist() if future_time is not None else None


  #print('data_dict: ', data_dict)
  #assert False
  return data_dict, qa_sub_type, future_time, distance_to_waypoint



def generate_3d_grounding_qa_sample_v6_double(
  asker_cav_id, initial_lidar_pose,
  notable_gts, cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,      
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified):


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'Is there anything I need to be aware of if my planned future trajectory is [(x0, z0), (x1, z1), (x2, z2)]?'
      }, {
      'from': 'gpt',
      #'value': 'Yes. There is a car at [x, z], which is close to your planned future trajectory.'
      #'value': 'There is nothing you need to be aware of.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  #print('sources: ', sources)
  num_future_waypoints = 6
  # we use future traj in ego's current coordinate system in question
  # for evaluation and visualization
  # we need to transform the answer to cav_ego's current coordinate system
  # by using the current lidar pose of cav_ego and cav_1 in world coordinate system
  # so we also need to store above two matrix in QA data file
  # probably not needed, we also load noy files during evaluation and visualization
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)
  # use traj in ego coordinate system in answer
  future_trajectory_str = future_trajectory_str_in_ego


  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO. '
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1. '
  else:
    assert False
  human_input += 'Is there anything I need to be aware of if my planned future trajectory is %s?' % future_trajectory_str
  #print('human_input: ', human_input)

  #print('notable_gts: ', notable_gts)
  # None or (1, 7+1): box parameter + distance to waypoint
  # notable_gts is in cav_ego's coordinate system

  qa_sub_type = []

  gpt_output = ''
  # Answer depends on whether there is notable_gts
  if notable_gts is not None:
    distance_to_waypoint = notable_gts[:, -1]
    future_time = notable_gts[:, -2]
    gt_box_str = [round_to_str(notable_gts[i]) for i in range(notable_gts.shape[0])]
    gt_center_str = [round_to_str(notable_gts[i], center_only=True) for i in range(notable_gts.shape[0])]
    #print('notable_gts: ', notable_gts)
    #print('gt_box_str: ', gt_box_str)
    #print('gt_center_str: ', gt_center_str)
    gt_center_str = ', '.join(gt_center_str)  
    #print('gt_center_str: ', gt_center_str)
    #if notable_gts.shape[0] > 1:
    #  assert False

    if not simplified:
      print('Not implemented')  
      assert False  
      gpt_output += 'Yes, there is a car close to your planned future trajectory. Its bounding box parameters are %s.' % (gt_box_str)
    else:
      if notable_gts.shape[0] == 1:  
        gpt_output += 'Yes, there is a car at %s, which is close to your planned future trajectory.' % (gt_center_str)
      else:
        gpt_output += 'Yes, there are cars at %s, which are close to your planned future trajectory.' % (gt_center_str)
        #print('gpt_output: ', gpt_output)
        #assert False  


    #print('gpt_output: ', gpt_output)
    #assert False

    #print('gpt_output: ', gpt_output)
    # Reason depends on whether the merged reason point is in any det box
    # set i = 0, use first notable_gts to generate reason, fix this part later
    # if cav_k detects any one of notable_gts, we will say cav_k detects it (them) in the answer
    #print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
    if notable_gts.shape[0] > 1:
      it_or_them = 'them'
    else:  
      it_or_them = 'it'  

    if np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1) and np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_output += ' Both connected autonomous vehicles detect %s.' % it_or_them
    elif np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1):
      gpt_output += ' Connected autonomous vehicle ego detects %s.' % it_or_them 
    elif np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_output += ' Connected autonomous vehicle 1 detects %s.' % it_or_them 
    else:
      gpt_output += ' None of the connected autonomous vehicles detects %s but the merged feature map detects %s.' % (it_or_them, it_or_them)

    # however, we keep track the qa_sub_type info for each notable_gt
    for i in range(notable_gts.shape[0]):
      if merged_reason_point_in_det_boxes_dict['ego'][i] > -1 and merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(0)
      elif merged_reason_point_in_det_boxes_dict['ego'][i] > -1:
        qa_sub_type.append(1)
      elif merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(2)
      else:
        qa_sub_type.append(3)

  else: # notable_gts is None
    distance_to_waypoint = None
    future_time = None
    gpt_output += 'There is nothing you need to be aware of.'
    # Reason depends on whether the merged reason point is in any det box
    # TODO: set i = 0, use first notable_gts to generate reason, fix this part later
    i = 0
    if merged_reason_point_in_det_boxes_dict['ego'][i] > -1 and merged_reason_point_in_det_boxes_dict['1'][i] > -1:
      gpt_output += ' Based on the feature maps, both connected autonomous vehicles have false positive detections.'
      qa_sub_type.append(4)
    elif merged_reason_point_in_det_boxes_dict['ego'][i] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle ego has false positive detections.'  
      qa_sub_type.append(5)
    elif merged_reason_point_in_det_boxes_dict['1'][i] > -1:
      gpt_output += ' Based on the feature maps, connected autonomous vehicle 1 has false positive detections.'  
      qa_sub_type.append(6)
    else:
      gpt_output += ' None of the connected autonomous vehicles detects anything near your planned future trajectory.'
      qa_sub_type.append(7)
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['distance_to_waypoint'] = distance_to_waypoint.tolist() if distance_to_waypoint is not None else None
  data_dict['future_time'] = future_time.tolist() if future_time is not None else None

  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()

  #print('data_dict: ', data_dict)
  #assert False
  return data_dict, qa_sub_type, future_time, distance_to_waypoint



def generate_3d_grounding_qa_sample_v7(
  notable_gts, cav_ego_future_trajectory,
  merged_reason_point_in_det_boxes_dict,
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified):


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'What is the suggested future trajectory to avoid collision with nearby objects?'
      }, {
      'from': 'gpt',
      #'value': 'The suggested future trajectory is [(x0, z0), (x1, z1), (x2, z2), (x3, z3), (x4, z4), (x5, z5)].'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  num_future_waypoints = 6
  future_trajectory_str = get_future_trajectory_str(cav_ego_future_trajectory, num_future_waypoints)


  human_input = 'What is the suggested future trajectory to avoid collision with nearby objects?'
  #print('human_input: ', human_input)

  #print('notable_gts: ', notable_gts)
  # None or (1, 7+1): box parameter + distance to waypoint

  qa_sub_type = []

  gpt_output = ''
  include_gpt_reasoning_output = False
  gpt_reasoning_output = ''
  # Answer depends on whether there is notable_gts
  if notable_gts is not None:
    distance_to_waypoint = notable_gts[:, -1]
    future_time = notable_gts[:, -2]
    gt_box_str = [round_to_str(notable_gts[i]) for i in range(notable_gts.shape[0])]
    gt_center_str = [round_to_str(notable_gts[i], center_only=True) for i in range(notable_gts.shape[0])]
    #print('notable_gts: ', notable_gts)
    #print('gt_box_str: ', gt_box_str)
    #print('gt_center_str: ', gt_center_str)
    gt_center_str = ', '.join(gt_center_str)  
    #print('gt_center_str: ', gt_center_str)
    #if notable_gts.shape[0] > 1:
    #  assert False

    if not simplified:
      print('Not implemented')  
      assert False  
    else:
      gpt_output += 'The suggested future trajectory is %s.' % future_trajectory_str  
      if notable_gts.shape[0] == 1:  
        gpt_reasoning_output += 'Yes, there is a car at %s, which is close to your planned future trajectory.' % (gt_center_str)
      else:
        gpt_reasoning_output += 'Yes, there are cars at %s, which are close to your planned future trajectory.' % (gt_center_str)
        #print('gpt_output: ', gpt_output)
        #assert False  


    #print('gpt_output: ', gpt_output)
    #assert False

    #print('gpt_output: ', gpt_output)
    # Reason depends on whether the merged reason point is in any det box
    # set i = 0, use first notable_gts to generate reason, fix this part later
    # if cav_k detects any one of notable_gts, we will say cav_k detects it (them) in the answer
    #print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
    if notable_gts.shape[0] > 1:
      it_or_them = 'them'
    else:  
      it_or_them = 'it'  

    if np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1) and np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_reasoning_output += ' Both connected autonomous vehicles detect %s.' % it_or_them
    elif np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1):
      gpt_reasoning_output += ' Connected autonomous vehicle ego detects %s.' % it_or_them 
    elif np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_reasoning_output += ' Connected autonomous vehicle 1 detects %s.' % it_or_them 
    else:
      gpt_reasoning_output += ' None of the connected autonomous vehicles detects %s but the merged feature map detects %s.' % (it_or_them, it_or_them)

    # however, we keep track the qa_sub_type info for each notable_gt
    for i in range(notable_gts.shape[0]):
      if merged_reason_point_in_det_boxes_dict['ego'][i] > -1 and merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(0)
      elif merged_reason_point_in_det_boxes_dict['ego'][i] > -1:
        qa_sub_type.append(1)
      elif merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(2)
      else:
        qa_sub_type.append(3)

  else: # notable_gts is None
    distance_to_waypoint = None
    future_time = None
    gpt_output += 'The suggested future trajectory is %s.' % future_trajectory_str  
    # Reason depends on whether the merged reason point is in any det box
    # set i = 0, use first notable_gts to generate reason, fix this part later
    i = 0
    if merged_reason_point_in_det_boxes_dict['ego'][i] > -1 and merged_reason_point_in_det_boxes_dict['1'][i] > -1:
      gpt_reasoning_output += ' Based on the feature maps, both connected autonomous vehicles have false positive detections.'
      qa_sub_type.append(4)
    elif merged_reason_point_in_det_boxes_dict['ego'][i] > -1:
      gpt_reasoning_output += ' Based on the feature maps, connected autonomous vehicle ego has false positive detections.'  
      qa_sub_type.append(5)
    elif merged_reason_point_in_det_boxes_dict['1'][i] > -1:
      gpt_reasoning_output += ' Based on the feature maps, connected autonomous vehicle 1 has false positive detections.'  
      qa_sub_type.append(6)
    else:
      gpt_reasoning_output += ' None of the connected autonomous vehicles detects anything near your planned future trajectory.'
      qa_sub_type.append(7)

  if include_gpt_reasoning_output:
    assert False  
    gpt_output += gpt_reasoning_output  

  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['distance_to_waypoint'] = distance_to_waypoint.tolist() if distance_to_waypoint is not None else None
  data_dict['future_time'] = future_time.tolist() if future_time is not None else None

  data_dict['gpt_reasoning_output'] = gpt_reasoning_output

  #print('data_dict: ', data_dict)
  #assert False
  return data_dict, qa_sub_type, future_time, distance_to_waypoint



def generate_3d_grounding_qa_sample_v7_double(
  asker_cav_id, initial_lidar_pose,
  notable_gts, cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified):


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'I am CAV_EGO. What is the suggested future trajectory to avoid collision with nearby objects?'
      #'value': 'I am CAV_1. What is the suggested future trajectory to avoid collision with nearby objects?'
      }, {
      'from': 'gpt',
      #'value': 'The suggested future trajectory is [(x0, z0), (x1, z1), (x2, z2), (x3, z3), (x4, z4), (x5, z5)].'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  num_future_waypoints = 6
  # we use future traj in asker's current coordinate system in answer
  # for evaluation and visualization
  # we need to transform the answer to cav_ego's current coordinate system
  # by using the current lidar pose of cav_ego and cav_1 in world coordinate system
  # so we also need to store above two matrix in QA data file
  # probably not needed, we also load noy files during evaluation and visualization
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)
  # use traj in self coordinate system in answer
  future_trajectory_str = future_trajectory_str_in_self
  

  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO. '  
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1. '  
  else:
    assert False  
  human_input += 'What is the suggested future trajectory to avoid collision with nearby objects?'
  #print('human_input: ', human_input)

  #print('notable_gts: ', notable_gts)
  # None or (1, 7+1): box parameter + distance to waypoint

  qa_sub_type = []

  gpt_output = ''
  include_gpt_reasoning_output = False
  gpt_reasoning_output = ''
  # Answer depends on whether there is notable_gts
  if notable_gts is not None:
    distance_to_waypoint = notable_gts[:, -1]
    future_time = notable_gts[:, -2]
    gt_box_str = [round_to_str(notable_gts[i]) for i in range(notable_gts.shape[0])]
    gt_center_str = [round_to_str(notable_gts[i], center_only=True) for i in range(notable_gts.shape[0])]
    #print('notable_gts: ', notable_gts)
    #print('gt_box_str: ', gt_box_str)
    #print('gt_center_str: ', gt_center_str)
    gt_center_str = ', '.join(gt_center_str)  
    #print('gt_center_str: ', gt_center_str)
    #if notable_gts.shape[0] > 1:
    #  assert False

    if not simplified:
      print('Not implemented')  
      assert False  
    else:
      gpt_output += 'The suggested future trajectory is %s.' % future_trajectory_str  
      if notable_gts.shape[0] == 1:  
        gpt_reasoning_output += 'Yes, there is a car at %s, which is close to your planned future trajectory.' % (gt_center_str)
      else:
        gpt_reasoning_output += 'Yes, there are cars at %s, which are close to your planned future trajectory.' % (gt_center_str)
        #print('gpt_output: ', gpt_output)
        #assert False  


    #print('gpt_output: ', gpt_output)
    #assert False

    #print('gpt_output: ', gpt_output)
    # Reason depends on whether the merged reason point is in any det box
    # set i = 0, use first notable_gts to generate reason, fix this part later
    # if cav_k detects any one of notable_gts, we will say cav_k detects it (them) in the answer
    #print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
    if notable_gts.shape[0] > 1:
      it_or_them = 'them'
    else:  
      it_or_them = 'it'  

    if np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1) and np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_reasoning_output += ' Both connected autonomous vehicles detect %s.' % it_or_them
    elif np.any(merged_reason_point_in_det_boxes_dict['ego'] > -1):
      gpt_reasoning_output += ' Connected autonomous vehicle ego detects %s.' % it_or_them 
    elif np.any(merged_reason_point_in_det_boxes_dict['1'] > -1):
      gpt_reasoning_output += ' Connected autonomous vehicle 1 detects %s.' % it_or_them 
    else:
      gpt_reasoning_output += ' None of the connected autonomous vehicles detects %s but the merged feature map detects %s.' % (it_or_them, it_or_them)

    # however, we keep track the qa_sub_type info for each notable_gt
    for i in range(notable_gts.shape[0]):
      if merged_reason_point_in_det_boxes_dict['ego'][i] > -1 and merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(0)
      elif merged_reason_point_in_det_boxes_dict['ego'][i] > -1:
        qa_sub_type.append(1)
      elif merged_reason_point_in_det_boxes_dict['1'][i] > -1:
        qa_sub_type.append(2)
      else:
        qa_sub_type.append(3)

  else: # notable_gts is None
    distance_to_waypoint = None
    future_time = None
    gpt_output += 'The suggested future trajectory is %s.' % future_trajectory_str  
    # Reason depends on whether the merged reason point is in any det box
    # set i = 0, use first notable_gts to generate reason, fix this part later
    i = 0
    if merged_reason_point_in_det_boxes_dict['ego'][i] > -1 and merged_reason_point_in_det_boxes_dict['1'][i] > -1:
      gpt_reasoning_output += ' Based on the feature maps, both connected autonomous vehicles have false positive detections.'
      qa_sub_type.append(4)
    elif merged_reason_point_in_det_boxes_dict['ego'][i] > -1:
      gpt_reasoning_output += ' Based on the feature maps, connected autonomous vehicle ego has false positive detections.'  
      qa_sub_type.append(5)
    elif merged_reason_point_in_det_boxes_dict['1'][i] > -1:
      gpt_reasoning_output += ' Based on the feature maps, connected autonomous vehicle 1 has false positive detections.'  
      qa_sub_type.append(6)
    else:
      gpt_reasoning_output += ' None of the connected autonomous vehicles detects anything near your planned future trajectory.'
      qa_sub_type.append(7)

  if include_gpt_reasoning_output:
    assert False  
    gpt_output += gpt_reasoning_output  

  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['distance_to_waypoint'] = distance_to_waypoint.tolist() if distance_to_waypoint is not None else None
  data_dict['future_time'] = future_time.tolist() if future_time is not None else None

  data_dict['gpt_reasoning_output'] = gpt_reasoning_output


  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()
  # (16)

  #print('data_dict: ', data_dict)
  #assert False
  return data_dict, qa_sub_type, future_time, distance_to_waypoint


def generate_3d_grounding_qa_sample_nq8(
  asker_cav_id, initial_lidar_pose,
  notable_gts, notable_gts_future_trajectory,
  cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
  with_context,
  num_future_waypoints, asker_initial_location,
  context_sample_list, context_list_from_gt,
  asker_initial_location_dict, double_cavs_future_trajectory_in_ego_current):
  '''
  NQ8 Suggested Action Classification

  if with_context, include NQ7's gt answer as context in question.
  '''  

  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'I am CAV_EGO. What are the suggested speed and steering settings?'
      #'value': 'I am CAV_1. What are the suggested speed and steering settings?'
      }, {
      'from': 'gpt',
      #'value': 
      #  'The suggested speed setting is: very fast.' # or fast, moderate, slow, very slow.
      #  'The suggested steering setting is: left.' # or slightly left, straight, slightly right, right.
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  default_num_future_waypoints = 6
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, default_num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, default_num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)

  # for NQ8, use cav_future_trajectory_in_self to get the gt action classification

  asker_initial_location_str = round_to_str(asker_initial_location, center_only=False)
  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO at %s. ' % asker_initial_location_str
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1 at %s. ' % asker_initial_location_str
  else:
    assert False  

  human_input += 'What are the suggested speed and steering settings to avoid collision with nearby objects? '
  #print('human_input: ', human_input)

  # nq7 and nq5 can only handle at most 1 future waypoint
  context_num_future_waypoints = 1
  # the following 3 are only used in context
  another_cav_is_notable_object = is_another_cav_notable_object(asker_cav_id, asker_initial_location_dict, notable_gts)
  another_cav_location_str = get_another_cav_location_str(asker_cav_id, asker_initial_location_dict)
  another_cav_future_trajectory_str = get_another_cav_future_trajectory_str(asker_cav_id, double_cavs_future_trajectory_in_ego_current, context_num_future_waypoints)

  if with_context:
    # context question
    human_input += 'Context: '
    # nq7 question (same as nq5 question)
    human_input += 'Where might those notable objects move in the future if my planned future trajectory is %s? ' % future_trajectory_str_in_ego
    #human_input += 'Where might those notable objects move in the future? '

    if context_sample_list is None:
      # context answer
      # use gt nq7 answer (same as gt nq5 answer)
      # Answer depends on whether there is notable_gts
      # Notable object action classification, on steering
      notable_gts_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
      notable_gts_action_indices = get_notable_gts_action(notable_gts, notable_gts_future_trajectory, notable_gts_action_classes)
      #assert False


      if notable_gts is not None:
        for i in range(notable_gts.shape[0]):
          # in ego coordinate  
          notable_gt_future_trajectory_str = get_future_trajectory_str(notable_gts_future_trajectory[i], context_num_future_waypoints)
          #print('notable_gt_future_trajectory_str: ', notable_gt_future_trajectory_str)
          gt_current_location_str = round_to_str(notable_gts[i], center_only=True)
          #print('gt_current_location_str: ', gt_current_location_str)

          if another_cav_is_notable_object == i:
            if context_num_future_waypoints == 0:
              human_input += 'There is a car at %s %s. ' % (another_cav_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])])
            else:
              human_input += 'There is a car at %s %s. The predicted future trajectory is %s. ' % (another_cav_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])], another_cav_future_trajectory_str)
            #print('human_input: ', human_input)
            #assert False

          else:
            if context_num_future_waypoints == 0:
              human_input += 'There is a car at %s %s. ' % (gt_current_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])])
            else:
              human_input += 'There is a car at %s %s. The predicted future trajectory is %s. ' % (gt_current_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])], notable_gt_future_trajectory_str)



      else: # notable_gts is None
        human_input += 'There is no notable object.'

    else: # use context_sample_list
      if context_list_from_gt:
        for context_sample in context_sample_list:
          human_input += context_sample['conversations'][1]['value']
          human_input += ' '
      else:
        # use inference result as context
        #print('graph of thoughts')
        for context_sample in context_sample_list:
          human_input += context_sample['outputs']
          human_input += ' '

  #print('human_input: ', human_input)
  #assert False

  
  speed_classes = ['fast', 'moderate', 'slow', 'very slow', 'stop']
  steering_classes = ['left', 'slightly left', 'straight', 'slightly right', 'right']
  cav_current_location_in_self = np.zeros(2)
  suggested_speed_idx, suggested_steering_idx, dist, angle = get_suggested_speed_steering(cav_current_location_in_self, cav_future_trajectory_in_self, num_future_waypoints)
  #print('cav_future_trajectory_in_self: ', cav_future_trajectory_in_self)
  #print('dist: ', dist)
  #print('angle: ', angle)
  #print('suggested_speed_idx: ', suggested_speed_idx, speed_classes[suggested_speed_idx])
  #print('suggested_steering_idx: ', suggested_steering_idx, steering_classes[suggested_steering_idx])
  gpt_output = 'The suggested speed setting is: %s. The suggested steering setting is: %s.' % (speed_classes[suggested_speed_idx], steering_classes[suggested_steering_idx])
  # TODO: include reason in the answer
  #print('gpt_output: ', gpt_output)

  qa_sub_type = suggested_speed_idx * 5 + suggested_steering_idx
  # 0 ~ 24

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type
  data_dict['suggested_speed_idx'] = suggested_speed_idx
  data_dict['suggested_steering_idx'] = suggested_steering_idx
  data_dict['dist'] = dist
  data_dict['angle'] = angle

  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()
  # (16)

  #print('data_dict: ', data_dict)
  #if notable_gts is not None:
  #  assert False
  return data_dict, qa_sub_type, suggested_speed_idx, suggested_steering_idx


def generate_3d_grounding_qa_sample_nq9(
  asker_cav_id, initial_lidar_pose,
  notable_gts, cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
  with_context,
  asker_initial_location,
  context_sample_list, context_list_from_gt):
  '''
  NQ9 Suggested future trajectory

  If with_context, use gt nq8 answer
  '''


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'I am CAV_EGO. What is the suggested trajectory to avoid collision with nearby objects?'
      #'value': 'I am CAV_1. What is the suggested trajectory to avoid collision with nearby objects?'
      }, {
      'from': 'gpt',
      #'value': 
      #  'The suggested trajectory is [(x0, y0), (x1, y1), ..., (x5, y5)] .
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  num_future_waypoints = 6
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)

  # for NQ8, use cav_future_trajectory_in_self to get the gt action classification
  # for NQ9, use cav_future_trajectory_in_ego to get the gt future trajectory

  asker_initial_location_str = round_to_str(asker_initial_location, center_only=False)
  #print('asker_initial_location_str: ', asker_initial_location_str)
  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO at %s. ' % asker_initial_location_str
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1 at %s. ' % asker_initial_location_str
  else:
    assert False  


  # Question
  human_input += 'What is the suggested future trajectory to avoid collision with nearby objects? '

  # Question Context
  speed_classes = ['fast', 'moderate', 'slow', 'very slow', 'stop']
  steering_classes = ['left', 'slightly left', 'straight', 'slightly right', 'right']
  cav_current_location_in_self = np.zeros(2)
  suggested_speed_idx, suggested_steering_idx, dist, angle = get_suggested_speed_steering(cav_current_location_in_self, cav_future_trajectory_in_self, num_future_waypoints)
  #print('cav_future_trajectory_in_self: ', cav_future_trajectory_in_self)
  #print('dist: ', dist)
  #print('angle: ', angle)
  #print('suggested_speed_idx: ', suggested_speed_idx, speed_classes[suggested_speed_idx])
  #print('suggested_steering_idx: ', suggested_steering_idx, steering_classes[suggested_steering_idx])
  if with_context:
    human_input += 'Context: '

    # determine whether to include context question depending on 
    # whether the context is from nq8 or not

    if context_sample_list is None:
      # context answer
      # use gt nq8
      human_input += 'What are the suggested speed and steering settings to avoid collision with nearby objects? '
      human_input += 'The suggested speed setting is: %s. The suggested steering setting is: %s.' % (speed_classes[suggested_speed_idx], steering_classes[suggested_steering_idx])
    else:
      #print('context_sample_list: ', context_sample_list)
      if context_list_from_gt:
        #print('context_list_from_gt')
        for context_sample in context_sample_list:
          # check if context is nq8, if so, include nq8 question
          if 'speed' in context_sample['conversations'][1]['value']:
            human_input += 'What are the suggested speed and steering settings to avoid collision with nearby objects? '

          human_input += context_sample['conversations'][1]['value']
          human_input += ' '
      else:
        # use inference result as context
        #print('graph of thoughts')
        for context_sample in context_sample_list:
          # check if context is nq8, if so, include nq8 question
          if 'speed' in context_sample['outputs']:
            human_input += 'What are the suggested speed and steering settings to avoid collision with nearby objects? '

          human_input += context_sample['outputs']
          human_input += ' '
 
  #print('human_input: ', human_input)
  #assert False


  # Answer
  gpt_output = 'The suggested future trajectory is %s.' % future_trajectory_str_in_ego
  # TODO: include reason in the answer
  #print('gpt_output: ', gpt_output)


  # Data
  qa_sub_type = suggested_speed_idx * 5 + suggested_steering_idx
  # 0 ~ 24

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type
  data_dict['suggested_speed_idx'] = suggested_speed_idx
  data_dict['suggested_steering_idx'] = suggested_steering_idx
  data_dict['dist'] = dist
  data_dict['angle'] = angle

  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()
  # (16)

  #print('data_dict: ', data_dict)
  #if asker_cav_id == '1':
  #  assert False
  return data_dict, qa_sub_type, suggested_speed_idx, suggested_steering_idx



def get_notable_gts_action(notable_gts, notable_gts_future_trajectory, notable_gts_action_classes):
  '''
  notable_gts_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  speed_classes = ['fast', 'moderate', 'slow', 'very slow', 'stop']
  steering_classes = ['left', 'slightly left', 'straight', 'slightly right', 'right']

  notable_gts: (N, 7), where [:,3], [:,5] represent current 2d location
  notable_gts_future_trajectory: (N, num_future_frames, 2)
  '''
  if notable_gts is None:
    return None  

  num_future_waypoints = 6
  num_notable_gts = notable_gts_future_trajectory.shape[0]
  notable_gts_action_indices = np.zeros(num_notable_gts)

  #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)

  for i in range(num_notable_gts):
    #print('i: ', i)
    # check whether static
    notable_gts_current_location = np.array([notable_gts[i][3], notable_gts[i][5]])
    suggested_speed_idx, suggested_steering_idx, dist, angle = get_suggested_speed_steering(notable_gts_current_location, notable_gts_future_trajectory[i], num_future_waypoints)
    #print('suggested_speed_idx: ', suggested_speed_idx)
    #assert False
    if suggested_speed_idx == 4: # stop
      notable_gts_action_indices[i] = 3 # staying at the same location 
    else:  
      # check direction
      # note that notable_gts_future_trajectory is in cav_ego coordinate system
      # so we need to pass the trajectory_diff (velocity) to get_suggested_speed_steering()
      # may need different threshold
      notable_gts_current_location = np.array([
        notable_gts[i,3],
        notable_gts[i,5]
      ])
      #print('notable_gts_current_location: ', notable_gts_current_location)
      #print('notable_gts_current_location.shape: ', notable_gts_current_location.shape)
      # (2, )

      notable_gts_current_location = np.expand_dims(notable_gts_current_location, axis=0)
      #print('notable_gts_current_location.shape: ', notable_gts_current_location.shape)
      # (1, 2)

      notable_gts_future_trajectory_concat = np.concatenate([
        notable_gts_current_location,
        notable_gts_future_trajectory[i]
      ], axis=0)
      #print('notable_gts_future_trajectory_concat: ', notable_gts_future_trajectory_concat)
      #print('notable_gts_future_trajectory_concat.shape: ', notable_gts_future_trajectory_concat.shape)
      # (num_future_frames+1, 2)

      notable_gts_future_trajectory_diff = notable_gts_future_trajectory_concat[1:] - notable_gts_future_trajectory_concat[:-1]
      #print('notable_gts_future_trajectory_diff: ', notable_gts_future_trajectory_diff)
      #print('notable_gts_future_trajectory_diff.shape: ', notable_gts_future_trajectory_diff.shape)
      # (num_future_frames, 2)
      notable_gts_future_trajectory_diff_initial = notable_gts_future_trajectory_diff[0]
      suggested_speed_idx, suggested_steering_idx, dist, angle = get_suggested_speed_steering(notable_gts_future_trajectory_diff_initial, notable_gts_future_trajectory_diff, num_future_waypoints)
      if suggested_steering_idx in [0, 1]:
        notable_gts_action_indices[i] = 1 # turning left
      elif suggested_steering_idx in [3, 4]: 
        notable_gts_action_indices[i] = 2 # turning right
      else:  
        notable_gts_action_indices[i] = 0 # moving forward  
      #print('notable_gts_action_indices[i]: ', notable_gts_action_indices[i])  

  #print('notable_gts_action_indices: ', notable_gts_action_indices)
  return notable_gts_action_indices


def get_another_cav_location_str(asker_cav_id, asker_initial_location_dict):
  for cav_id in asker_initial_location_dict:
    if cav_id != asker_cav_id:
      another_cav_location_str = round_to_str(asker_initial_location_dict[cav_id], center_only=False)
  return another_cav_location_str


def get_another_cav_future_trajectory_str(asker_cav_id, double_cavs_future_trajectory_in_ego_current, num_future_waypoints):
  for cav_id in double_cavs_future_trajectory_in_ego_current:
    if cav_id != asker_cav_id:
      another_cav_future_trajectory_str = get_future_trajectory_str(double_cavs_future_trajectory_in_ego_current[cav_id], num_future_waypoints)
  return another_cav_future_trajectory_str


def generate_3d_grounding_qa_sample_nq5(
  asker_cav_id, initial_lidar_pose,
  notable_gts, notable_gts_future_trajectory,
  cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,      
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
  num_future_waypoints, 
  with_context, asker_initial_location,
  context_sample_list, context_list_from_gt, 
  asker_initial_location_dict, double_cavs_future_trajectory_in_ego_current):
  '''
  NQ5 Prediction by Observation

  if with_context, include NQ4's gt answer as context in question.
  '''


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'I am CAV_EGO. Where might those notable objects move in the future if my planned future trajectory is [(x0, y0), (x1, y1), ...]?'
      }, {
      'from': 'gpt',
      #'value': 'There is a car moving forward. The predicted future trajectory is [(x0, y0), ...]. There is a car ...' 
      #'value': 'There is no notable object.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  #print('sources: ', sources)
  default_num_future_waypoints = 6
  # we use future traj in cav_ego's current coordinate system in question
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, default_num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, default_num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)


  asker_initial_location_str = round_to_str(asker_initial_location, center_only=False)
  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO at %s. ' % asker_initial_location_str  
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1 at %s. ' % asker_initial_location_str  
  else:
    assert False
  human_input += 'Where might those notable objects move in the future if my planned future trajectory is %s? ' % future_trajectory_str_in_ego


  another_cav_is_notable_object = is_another_cav_notable_object(asker_cav_id, asker_initial_location_dict, notable_gts)
  another_cav_location_str = get_another_cav_location_str(asker_cav_id, asker_initial_location_dict)
  another_cav_future_trajectory_str = get_another_cav_future_trajectory_str(asker_cav_id, double_cavs_future_trajectory_in_ego_current, num_future_waypoints)
  #print('another_cav_is_notable_object: ', another_cav_is_notable_object)
  #print('another_cav_location_str: ', another_cav_location_str)
  #print('another_cav_future_trajectory_str: ', another_cav_future_trajectory_str)
  #assert False


  if with_context:
    # Question Context
    # from nq4 notable object identification
    human_input += 'Context: '
    # skip duplicate planned future trajectory
    #human_input += 'Is there anything I need to be aware of if my planned future trajectory is %s? ' % future_trajectory_str_in_ego
    human_input += 'Is there anything I need to be aware of given my planned future trajectory? '

    if context_sample_list is None:
      # use gt context
      if notable_gts is not None:
        for i in range(notable_gts.shape[0]):  
          if another_cav_is_notable_object == i:
            human_input += 'There is a car at %s. ' % another_cav_location_str
            #print('human_input: ', human_input)
            #assert False
          else:
            gt_current_location_str = round_to_str(notable_gts[i], center_only=True)
            #print('gt_current_location_str: ', gt_current_location_str)  
            human_input += 'There is a car at %s. ' % gt_current_location_str
      else:    
          human_input += 'There is no notable object. '
    else:
      if context_list_from_gt:
        for context_sample in context_sample_list:
          human_input += context_sample['conversations'][1]['value']
          human_input += ' '
      else:
        # use nq4 inference result as context
        #print('graph of thoughts')
        for context_sample in context_sample_list:
          human_input += context_sample['outputs']
          human_input += ' '

  #print('human_input: ', human_input)
  #assert False


  #print('notable_gts: ', notable_gts)
  # None or (1, 7+1+1+1): box parameter, gt_object_ids, future timestep,  distance to waypoint
  # notable_gts is in cav_ego's coordinate system
  #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
  #assert False

  # Notable object action classification, on steering
  notable_gts_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  notable_gts_action_indices = get_notable_gts_action(notable_gts, notable_gts_future_trajectory, notable_gts_action_classes)
  #assert False
   

  qa_sub_type = notable_gts_action_indices.copy().tolist() if notable_gts_action_indices is not None else [-1]

  gpt_output = ''
  # Answer depends on whether there is notable_gts
  if notable_gts is not None:
    distance_to_waypoint = notable_gts[:, -1]
    future_time = notable_gts[:, -2]
    for i in range(notable_gts.shape[0]):
      notable_gt_future_trajectory_str = get_future_trajectory_str(notable_gts_future_trajectory[i], num_future_waypoints)  
      #print('notable_gt_future_trajectory_str: ', notable_gt_future_trajectory_str)
      gt_current_location_str = round_to_str(notable_gts[i], center_only=True)
      #print('gt_current_location_str: ', gt_current_location_str)
      if not simplified:
        print('Not implemented')  
        assert False  
      else:

        if another_cav_is_notable_object == i:
          if num_future_waypoints == 0:  
            gpt_output += 'There is a car at %s %s. ' % (another_cav_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])])  
          else:  
            gpt_output += 'There is a car at %s %s. The predicted future trajectory is %s. ' % (another_cav_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])], another_cav_future_trajectory_str)  
          #print('gpt_output: ', gpt_output)
          #assert False

        else:
          if num_future_waypoints == 0:  
            gpt_output += 'There is a car at %s %s. ' % (gt_current_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])])  
          else:  
            gpt_output += 'There is a car at %s %s. The predicted future trajectory is %s. ' % (gt_current_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])], notable_gt_future_trajectory_str)  
          #print('gpt_output: ', gpt_output)

    #print('gpt_output: ', gpt_output)
  else: # notable_gts is None
    distance_to_waypoint = None
    future_time = None
    gpt_output += 'There is no notable object.'
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['distance_to_waypoint'] = distance_to_waypoint.tolist() if distance_to_waypoint is not None else None
  data_dict['future_time'] = future_time.tolist() if future_time is not None else None

  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()

  #print('data_dict: ', data_dict)
  #if notable_gts is not None:
  #  assert False
  #if asker_cav_id == '1':
  #  assert False
  return data_dict, qa_sub_type, future_time, distance_to_waypoint



def generate_3d_grounding_qa_sample_nq6(
  asker_cav_id, initial_lidar_pose,
  notable_gts, notable_gts_future_trajectory,
  cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,      
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
  num_future_waypoints, 
  with_context, asker_initial_location, 
  cav_ids, double_cavs_future_trajectory_in_ego_current, asker_initial_location_dict,
  context_sample_list, context_list_from_gt):
  '''
  NQ6 Prediction by Planning

  if with_context, include nq4 answer and all CAVs' planned future trajectory as context in question.
  '''

  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'I am CAV_EGO at (x0, y0). Where might other CAVs move in the future given their planned future trajectories? 
      }, {
      'from': 'gpt',
      #'value': 'CAV_1 is at (x0, y0) moving forward. Its planned future trajectory is [(x0, y0), (x1, y1) ...].' 
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  #print('sources: ', sources)
  default_num_future_waypoints = 6
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, default_num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, default_num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)

  asker_initial_location_str = round_to_str(asker_initial_location, center_only=False)
  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO at %s. ' % asker_initial_location_str  
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1 at %s. ' % asker_initial_location_str  
  else:
    assert False
  human_input += 'Where might other CAVs move in the future given their planned future trajectories? '


  another_cav_is_notable_object = is_another_cav_notable_object(asker_cav_id, asker_initial_location_dict, notable_gts)
  another_cav_location_str = get_another_cav_location_str(asker_cav_id, asker_initial_location_dict)
  another_cav_future_trajectory_str = get_another_cav_future_trajectory_str(asker_cav_id, double_cavs_future_trajectory_in_ego_current, num_future_waypoints)

  cav_name_dict = {'ego': 'CAV_EGO', '1': 'CAV_1'}
  if with_context:
    # Question Context
    human_input += 'Context: '
    # use GT CAV future trajectory
    for cav_id in cav_ids:
      if cav_id != asker_cav_id:
        context_initial_location_str = round_to_str(asker_initial_location_dict[cav_id], center_only=False) 
        human_input += '%s is at %s. ' % (cav_name_dict[cav_id], context_initial_location_str)
        context_future_trajectory_str_in_ego = get_future_trajectory_str(double_cavs_future_trajectory_in_ego_current[cav_id], default_num_future_waypoints) 
        human_input += 'Its planned future trajectory is %s. ' % context_future_trajectory_str_in_ego

    # include notable object context (nq4 output)
    human_input += 'Notable object context: '
    if context_sample_list is None:
      # use gt context
      if notable_gts is not None:
        for i in range(notable_gts.shape[0]):
          if another_cav_is_notable_object == i:
            human_input += 'There is a car at %s. ' % another_cav_location_str
            #print('human_input: ', human_input)
            #assert False
          else:
            gt_current_location_str = round_to_str(notable_gts[i], center_only=True)
            #print('gt_current_location_str: ', gt_current_location_str)  
            human_input += 'There is a car at %s. ' % gt_current_location_str
      else:
          human_input += 'There is no notable object. '
    else:
      if context_list_from_gt:
        for context_sample in context_sample_list:
          human_input += context_sample['conversations'][1]['value']
          human_input += ' '
      else:
        # use nq4 inference result as context
        #print('graph of thoughts')
        for context_sample in context_sample_list:
          human_input += context_sample['outputs']
          human_input += ' '
        #print('human_input: ', human_input)
        #assert False

  #print('human_input: ', human_input)
  #assert False

  # Answer
  qa_sub_type = []
  gpt_output = ''
  for cav_id in cav_ids:
    if cav_id != asker_cav_id:  
      answer_initial_location_str = round_to_str(asker_initial_location_dict[cav_id], center_only=False)
      answer_future_trajectory_str_in_ego = get_future_trajectory_str(double_cavs_future_trajectory_in_ego_current[cav_id], num_future_waypoints)

      answer_cav_box = np.array([[0, 0, 0, asker_initial_location_dict[cav_id][0], 0, asker_initial_location_dict[cav_id][1], 0, 0]])
      #print('answer_cav_box: ', answer_cav_box)
      answer_cav_future_trajectory = np.expand_dims(double_cavs_future_trajectory_in_ego_current[cav_id], axis=0)
      #print('answer_cav_future_trajectory: ', answer_cav_future_trajectory)
      answer_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
      answer_action_indices = get_notable_gts_action(answer_cav_box, answer_cav_future_trajectory, answer_action_classes)
      #print('answer_action_indices: ', answer_action_indices)
      answer_action = answer_action_classes[int(answer_action_indices[0])]
      qa_sub_type.append(int(answer_action_indices[0]))

      gpt_output += '%s is at %s %s. Its planned future trajectory is %s. ' % (cav_name_dict[cav_id], answer_initial_location_str, answer_action, answer_future_trajectory_str_in_ego)

      if another_cav_is_notable_object >= 0:
        gpt_output += '%s is a notable object. Its planned future trajectory is its predicted future trajectory. ' % cav_name_dict[cav_id]
        #print('gpt_output: ', gpt_output)
        #assert False
      else:
        gpt_output += '%s is a not notable object. ' % cav_name_dict[cav_id]


  #print('gpt_output: ', gpt_output)
  #assert False


  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()

  #print('data_dict: ', data_dict)
  #if notable_gts is not None:
  #  assert False
  #if asker_cav_id == '1':
  #  assert False
  return data_dict, qa_sub_type 


def has_cav_in_k_meters(asker_cav_id, asker_initial_location_dict, k):
  #print('asker_cav_id: ', asker_cav_id)
  #print('asker_initial_location_dict: ', asker_initial_location_dict)
  for cav_id in asker_initial_location_dict:
    #print('cav_id: ', cav_id)
    if cav_id != asker_cav_id:
      dist = np.linalg.norm(asker_initial_location_dict[cav_id] - asker_initial_location_dict[asker_cav_id])
      #print('dist: ', dist)
      if dist < k:
        return True

  return False


def is_another_cav_notable_object(asker_cav_id, asker_initial_location_dict, notable_gts):
  '''
  if another cav is not a notable object, return -1
  otherwise return the idx in notable_gts that is related to the another cav
  '''
  #print('asker_cav_id: ', asker_cav_id)
  #print('asker_initial_location_dict: ', asker_initial_location_dict)
  #print('notable_gts: ', notable_gts)
  if notable_gts is None:
    return -1
  for cav_id in asker_initial_location_dict:
    if cav_id != asker_cav_id:
      another_cav_location = asker_initial_location_dict[cav_id]
      for i in range(len(notable_gts)):
        dist = np.sqrt((another_cav_location[0] - notable_gts[i][3])**2 + (another_cav_location[1] - notable_gts[i][5])**2)
        #print('dist: ', dist)
        if dist < 1:
          #assert False
          return i
     
    #assert False

  #assert False
  return -1


def generate_3d_grounding_qa_sample_nq7(
  asker_cav_id, initial_lidar_pose,
  notable_gts, notable_gts_future_trajectory,
  cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,      
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
  num_future_waypoints, 
  with_context, asker_initial_location,
  cav_ids, double_cavs_future_trajectory_in_ego_current, asker_initial_location_dict,
  context_sample_list, context_list_from_gt):
  '''
  NQ7 Prediction 

  if with_context, include NQ5 and NQ6's gt answer as context in question.
  '''


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'I am CAV_EGO. Where might those notable objects move in the future if my planned future trajectory is [(x0, y0), (x1, y1), ...]?'
      }, {
      'from': 'gpt',
      #'value': 'There is a car moving forward. The predicted future trajectory is [(x0, y0), ...]. There is a car ...' 
      #'value': 'There is no notable object.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  #print('sources: ', sources)
  default_num_future_waypoints = 6
  # we use future traj in cav_ego's current coordinate system in question
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, default_num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, default_num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)


  asker_initial_location_str = round_to_str(asker_initial_location, center_only=False)
  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO at %s. ' % asker_initial_location_str  
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1 at %s. ' % asker_initial_location_str  
  else:
    assert False
  human_input += 'Where might those notable objects move in the future if my planned future trajectory is %s? ' % future_trajectory_str_in_ego


  another_cav_is_notable_object = is_another_cav_notable_object(asker_cav_id, asker_initial_location_dict, notable_gts)
  another_cav_location_str = get_another_cav_location_str(asker_cav_id, asker_initial_location_dict)
  another_cav_future_trajectory_str = get_another_cav_future_trajectory_str(asker_cav_id, double_cavs_future_trajectory_in_ego_current, num_future_waypoints)


  # Notable object action classification, on steering
  notable_gts_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  if notable_gts is not None:
    notable_gts_action_indices = get_notable_gts_action(notable_gts, notable_gts_future_trajectory, notable_gts_action_classes)
  else:  
    notable_gts_action_indices = None  
  #assert False

  cav_name_dict = {'ego': 'CAV_EGO', '1': 'CAV_1'}
  if with_context:
    # Question Context
    # from nq5 
    human_input += 'Context: '
    # skip duplicate question

    if context_sample_list is None:
      # use gt context

      if notable_gts is not None:
        for i in range(notable_gts.shape[0]):  
          notable_gt_future_trajectory_str = get_future_trajectory_str(notable_gts_future_trajectory[i], num_future_waypoints)  
          gt_current_location_str = round_to_str(notable_gts[i], center_only=True)
          #print('gt_current_location_str: ', gt_current_location_str)  

          # always use original notable_gts location to generate training dataset simulation of detection error
          if False and another_cav_is_notable_object == i:
            human_input += 'There is a car at %s %s. The predicted future trajectory is %s. ' % (another_cav_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])], another_cav_future_trajectory_str)
          else:
            human_input += 'There is a car at %s %s. The predicted future trajectory is %s. ' % (gt_current_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])], notable_gt_future_trajectory_str)
      else:    
          human_input += 'There is no notable object. '

      # from nq6
      for cav_id in cav_ids:
        if cav_id != asker_cav_id:
          context_initial_location_str = round_to_str(asker_initial_location_dict[cav_id], center_only=False)
          context_future_trajectory_str_in_ego = get_future_trajectory_str(double_cavs_future_trajectory_in_ego_current[cav_id], num_future_waypoints)

          context_cav_box = np.array([[0, 0, 0, asker_initial_location_dict[cav_id][0], 0, asker_initial_location_dict[cav_id][1], 0, 0]])
          context_cav_future_trajectory = np.expand_dims(double_cavs_future_trajectory_in_ego_current[cav_id], axis=0)
          context_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
          context_action_indices = get_notable_gts_action(context_cav_box, context_cav_future_trajectory, context_action_classes)
          context_action = context_action_classes[int(context_action_indices[0])]

          human_input += '%s is at %s %s. Its planned future trajectory is %s. ' % (cav_name_dict[cav_id], context_initial_location_str, context_action, context_future_trajectory_str_in_ego)

          if another_cav_is_notable_object >= 0:
            human_input += '%s is a notable object. Its planned future trajectory is its predicted future trajectory. ' % cav_name_dict[cav_id]
            #print('human_input: ', human_input)
            #assert False
          else:
            human_input += '%s is a not notable object. ' % cav_name_dict[cav_id]

    else:
      if context_list_from_gt:
        for context_sample in context_sample_list:
          human_input += context_sample['conversations'][1]['value']
          human_input += ' '
      else:
        # use nq5 nq6 inference results as context
        #print('graph of thoughts')
        for context_sample in context_sample_list:
          human_input += context_sample['outputs']
          human_input += ' '

  #print('human_input: ', human_input)
  #assert False


  #print('notable_gts: ', notable_gts)
  # None or (1, 7+1+1+1): box parameter, gt_object_ids, future timestep,  distance to waypoint
  # notable_gts is in cav_ego's coordinate system
  #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
  #assert False

   

  qa_sub_type = notable_gts_action_indices.copy().tolist() if notable_gts_action_indices is not None else [-1]

  gpt_output = ''
  # Answer depends on whether there is notable_gts
  if notable_gts is not None:
    distance_to_waypoint = notable_gts[:, -1]
    future_time = notable_gts[:, -2]
    for i in range(notable_gts.shape[0]):
      notable_gt_future_trajectory_str = get_future_trajectory_str(notable_gts_future_trajectory[i], num_future_waypoints)  
      #print('notable_gt_future_trajectory_str: ', notable_gt_future_trajectory_str)
      gt_current_location_str = round_to_str(notable_gts[i], center_only=True)
      #print('gt_current_location_str: ', gt_current_location_str)

      if not simplified:
        print('Not implemented')  
        assert False  
      else:
        if another_cav_is_notable_object == i:
          if num_future_waypoints == 0:
            gpt_output += 'There is a car at %s %s. ' % (another_cav_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])])
          else:
            gpt_output += 'There is a car at %s %s. The predicted future trajectory is %s. ' % (another_cav_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])], another_cav_future_trajectory_str)
          #print('gpt_output: ', gpt_output)
          #assert False

        else:
          if num_future_waypoints == 0:
            gpt_output += 'There is a car at %s %s. ' % (gt_current_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])])
          else:
            gpt_output += 'There is a car at %s %s. The predicted future trajectory is %s. ' % (gt_current_location_str, notable_gts_action_classes[int(notable_gts_action_indices[i])], notable_gt_future_trajectory_str)
          #print('gpt_output: ', gpt_output)



    #print('gpt_output: ', gpt_output)
  else: # notable_gts is None
    distance_to_waypoint = None
    future_time = None
    gpt_output += 'There is no notable object.'
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['distance_to_waypoint'] = distance_to_waypoint.tolist() if distance_to_waypoint is not None else None
  data_dict['future_time'] = future_time.tolist() if future_time is not None else None

  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()


  # conditions for dynamic graph
  #print('data_dict: ', data_dict)
  # simplified prediction
  # whether two cav are nearby
  data_dict['has_nearby_cav'] = dict()
  for k in [10, 20, 30, 40, 50]:
    has_nearby_cav = has_cav_in_k_meters(asker_cav_id, asker_initial_location_dict, k)
    data_dict['has_nearby_cav'][k] = has_nearby_cav
  

  # check whether other cav is a notable object
  another_cav_is_notable_object = is_another_cav_notable_object(asker_cav_id, asker_initial_location_dict, notable_gts)
  data_dict['another_cav_is_notable_object'] = another_cav_is_notable_object
  #assert False


  #print('data_dict: ', data_dict)
  #assert False
  #if notable_gts is not None:
  #  assert False
  #if asker_cav_id == '1':
  #  assert False
  return data_dict, qa_sub_type, future_time, distance_to_waypoint, another_cav_is_notable_object



def has_det_in_k_meters(det_box_scores, asker_initial_location, k):
  #print('det_box_scores: ', det_box_scores)

  if det_box_scores is None or len(det_box_scores) == 0:
    return False

  det_dists = np.sqrt((det_box_scores[:, 3] - asker_initial_location[0])**2 + (det_box_scores[:, 5] -  asker_initial_location[1])**2)
  #print('det_dists: ', det_dists)

  return bool(np.any(det_dists < k))


def generate_3d_grounding_qa_sample_nq4(
  asker_cav_id, initial_lidar_pose,
  notable_gts, notable_gts_future_trajectory,
  cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,      
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
  num_future_waypoints, with_context,
  visible_gt_object_ids_dict, invisible_gt_object_ids_dict,
  occluding_gt_ids_in_gt_boxes, occluded_gt_ids_in_gt_boxes, gt_boxes, asker_initial_location,
  context_sample_list, context_list_from_gt,
  det_box_scores,
  asker_initial_location_dict):
  '''
  NQ4 Notable object identification

  if with_context, 
    if context_sample_list is None, include NQ1 and NQ3's gt answer as context in question.
    else use context_sample_list to generate context
  '''


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'I am CAV_EGO. Is there anything I need to be aware of if my planned future trajectory is [(x0, y0), (x1, y1), ...]?'
      }, {
      'from': 'gpt',
      #'value': 'There is a car at (x0, y0). There is a pedestrian at ...' 
      #'value': 'There is no notable object.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  #print('sources: ', sources)
  default_num_future_waypoints = 6
  # we use future traj in cav_ego's current coordinate system in question
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, default_num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, default_num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)

  asker_initial_location_str = round_to_str(asker_initial_location, center_only=False)
  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO at %s. ' % asker_initial_location_str
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1 at %s. ' % asker_initial_location_str
  else:
    assert False
  human_input += 'Is there anything I need to be aware of if my planned future trajectory is %s? ' % future_trajectory_str_in_ego

  assert(simplified)

  another_cav_is_notable_object = is_another_cav_notable_object(asker_cav_id, asker_initial_location_dict, notable_gts)

  if with_context:
    # from nq1 and nq3 
    # context question
    human_input += 'Context: '
    # simplify nq1 and nq3 questions to avoid running out of tokens
    #human_input += 'Is there anything I need to be aware of if my planned future trajectory is %s? ' % future_trajectory_str_in_ego
    human_input += 'Is there anything I need to be aware of given my planned future trajectory? '

    # context answer
    if context_sample_list is None:
      # use gt context
      if notable_gts is not None:
        for i in range(notable_gts.shape[0]):
          gt_current_location_str = round_to_str(notable_gts[i], center_only=True)
          #print('gt_current_location_str: ', gt_current_location_str)
          notable_gt_id = notable_gts[i, 7]
          if notable_gt_id in visible_gt_object_ids_dict[asker_cav_id]:

            if another_cav_is_notable_object == i:
              for cav_id in asker_initial_location_dict:
                if cav_id != asker_cav_id:
                  another_cav_location_str = round_to_str(asker_initial_location_dict[cav_id], center_only=False)
                  human_input += 'There is a car at %s visible to you. ' % another_cav_location_str
                  #print('another_cav_is_notable_object: ', another_cav_is_notable_object)
                  #print('notable_gts: ', notable_gts)
                  #print('asker_initial_location_dict: ', asker_initial_location_dict)
                  #print('asker_cav_id: ', asker_cav_id)
                  #print('human_input: ', human_input)
                  #assert False
            else:
              human_input += 'There is a car at %s visible to you. ' % gt_current_location_str
              #print('gpt_output: ', gpt_output)
          elif notable_gt_id in invisible_gt_object_ids_dict[asker_cav_id]: 
            
            if another_cav_is_notable_object == i:
              for cav_id in asker_initial_location_dict:
                if cav_id != asker_cav_id:
                  another_cav_location_str = round_to_str(asker_initial_location_dict[cav_id], center_only=False)
                  human_input += 'There is a car at %s invisible to you. ' % another_cav_location_str
                  #print('another_cav_is_notable_object: ', another_cav_is_notable_object)
                  #print('notable_gts: ', notable_gts)
                  #print('asker_initial_location_dict: ', asker_initial_location_dict)
                  #print('asker_cav_id: ', asker_cav_id)
                  #print('human_input: ', human_input)
                  #assert False
            
            else:
              human_input += 'There is a car at %s invisible to you. ' % gt_current_location_str
          else:  
            print('a notable object neither visible or invisible')  
            assert False
      else: # notable_gts is None
        human_input += 'There is no notable object.'
    else:
      if context_list_from_gt:
        for context_sample in context_sample_list:
          human_input += context_sample['conversations'][1]['value']
          human_input += ' '
      else:
        # use nq1 nq3 inference result as context
        #print('graph of thoughts')
        for context_sample in context_sample_list:
          human_input += context_sample['outputs']
          human_input += ' '
  
  # end of generating context
  #print('human_input: ', human_input)
  #assert False


  #print('notable_gts: ', notable_gts)
  # None or (1, 7+1+1+1): box parameter, gt_object_ids, future timestep,  distance to waypoint
  # notable_gts is in cav_ego's coordinate system
  #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
  #assert False

  # Notable object action classification, on steering
  notable_gts_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  notable_gts_action_indices = get_notable_gts_action(notable_gts, notable_gts_future_trajectory, notable_gts_action_classes)
  #assert False
   

  qa_sub_type = notable_gts_action_indices.copy().tolist() if notable_gts_action_indices is not None else [-1]

  gpt_output = ''
  # Answer depends on whether there is notable_gts
  if notable_gts is not None:
    distance_to_waypoint = notable_gts[:, -1]
    future_time = notable_gts[:, -2]
    for i in range(notable_gts.shape[0]):
      notable_gt_future_trajectory_str = get_future_trajectory_str(notable_gts_future_trajectory[i], num_future_waypoints)  
      #print('notable_gt_future_trajectory_str: ', notable_gt_future_trajectory_str)
      gt_current_location_str = round_to_str(notable_gts[i], center_only=True)
      #print('gt_current_location_str: ', gt_current_location_str)

      if not simplified:
        print('Not implemented')  
        assert False  
      else:

        if another_cav_is_notable_object == i:
          for cav_id in asker_initial_location_dict:
            if cav_id != asker_cav_id:
              another_cav_location_str = round_to_str(asker_initial_location_dict[cav_id], center_only=False)
              gpt_output += 'There is a car at %s. ' % another_cav_location_str
              #print('gpt_output: ', gpt_output)
              #assert False
        else:
          gpt_output += 'There is a car at %s. ' % gt_current_location_str 
          #print('gpt_output: ', gpt_output)

    #print('gpt_output: ', gpt_output)
  else: # notable_gts is None
    distance_to_waypoint = None
    future_time = None
    gpt_output += 'There is no notable object.'
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['distance_to_waypoint'] = distance_to_waypoint.tolist() if distance_to_waypoint is not None else None
  data_dict['future_time'] = future_time.tolist() if future_time is not None else None

  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()


  # conditions for dynamic graph
  #print('data_dict: ', data_dict)
  # simplified perception
  # has nearby detection
  #print('det_box_scores: ', det_box_scores)
  data_dict['has_nearby_detection'] = dict()
  for k in [10, 20, 30, 40, 50]:
    has_nearby_detection = has_det_in_k_meters(det_box_scores, asker_initial_location, k)
    data_dict['has_nearby_detection'][k] = has_nearby_detection

  #print('data_dict: ', data_dict)
  #if notable_gts is not None:
  #  assert False
  #if asker_cav_id == '1':
  #  assert False  
  #assert False
  return data_dict, qa_sub_type, future_time, distance_to_waypoint



def generate_3d_grounding_qa_sample_nq3(
  asker_cav_id, initial_lidar_pose,
  notable_gts, notable_gts_future_trajectory,
  cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,      
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
  num_future_waypoints, with_context,
  visible_gt_object_ids_dict, invisible_gt_object_ids_dict,
  occluding_gt_ids_in_gt_boxes, occluded_gt_ids_in_gt_boxes, gt_boxes, asker_initial_location,
  context_sample_list, context_list_from_gt,
  asker_initial_location_dict):
  '''
  NQ3 Invisible Notable object identification

  if with_context, 
    if context_sample is None, include NQ2's gt answer as context in question.
    else use context_sample to generate context
  '''


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'I am CAV_EGO. What are the notable objects invisible to me near my planned future trajectory [(x0, y0), (x1, y1), ...]?'
      }, {
      'from': 'gpt',
      #'value': 'There is a car at (x0, y0). There is a pedestrian at ...' 
      #'value': 'There is no notable object.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  #print('sources: ', sources)
  default_num_future_waypoints = 6
  # we use future traj in cav_ego's current coordinate system in question
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, default_num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, default_num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)

  asker_initial_location_str = round_to_str(asker_initial_location, center_only=False)
  #print('asker_initial_location_str: ', asker_initial_location_str)
  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO at %s. ' % asker_initial_location_str
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1 at %s. ' % asker_initial_location_str
  else:
    assert False
  human_input += 'What are the notable objects invisible to me near my planned future trajectory %s? ' % future_trajectory_str_in_ego

  if with_context:
    # from nq2
    # context question
    # simply to skip duplicate info
    human_input += 'Context: '
    human_input += 'What objects might obstruct my view? '

    # context answer
    if context_sample_list is None:
      # use gt context
      if len(occluding_gt_ids_in_gt_boxes) > 0:
        for gt_id_in_gt_boxes in occluding_gt_ids_in_gt_boxes:
          gt_current_location_str = round_to_str(gt_boxes[gt_id_in_gt_boxes], center_only=True)
          #print('gt_current_location_str: ', gt_current_location_str)
          assert(simplified)
          human_input += 'There is a car at %s obstructing your view. ' % gt_current_location_str   
      else:    
        human_input += 'There is no object obstructing your view.'  
    else:
      if context_list_from_gt:
        for context_sample in context_sample_list:
          human_input += context_sample['conversations'][1]['value']
          human_input += ' '
      else:
        # use nq2 inference result as context
        #print('context_sample: ', context_sample)
        for context_sample in context_sample_list:
          human_input += context_sample['outputs']
          human_input += ' '

  #print('human_input: ', human_input)
  #assert False

  #print('visible_gt_object_ids_dict: ', visible_gt_object_ids_dict)
  #print('invisible_gt_object_ids_dict: ', invisible_gt_object_ids_dict)
  #print('notable_gts: ', notable_gts)
  # None or (1, 7+1+1+1): box parameter, gt_object_ids, future timestep,  distance to waypoint
  # notable_gts is in cav_ego's coordinate system
  #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
  #assert False

  # Notable object action classification, on steering
  notable_gts_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  notable_gts_action_indices = get_notable_gts_action(notable_gts, notable_gts_future_trajectory, notable_gts_action_classes)
  #assert False
   

  qa_sub_type = notable_gts_action_indices.copy().tolist() if notable_gts_action_indices is not None else [-1]

  gpt_output = ''
  # Answer depends on whether there is notable_gts
  another_cav_is_notable_object = is_another_cav_notable_object(asker_cav_id, asker_initial_location_dict, notable_gts)
  num_invisible_notable_gts = 0
  if notable_gts is not None:
    distance_to_waypoint = notable_gts[:, -1]
    future_time = notable_gts[:, -2]

    for i in range(notable_gts.shape[0]):
      notable_gt_future_trajectory_str = get_future_trajectory_str(notable_gts_future_trajectory[i], num_future_waypoints)  
      #print('notable_gt_future_trajectory_str: ', notable_gt_future_trajectory_str)
      gt_current_location_str = round_to_str(notable_gts[i], center_only=True)
      #print('gt_current_location_str: ', gt_current_location_str)

      notable_gt_id = notable_gts[i, 7]
      if notable_gt_id in invisible_gt_object_ids_dict[asker_cav_id]:
        num_invisible_notable_gts += 1  
        assert(simplified)  
        # check if this notable object is another cav,
        # if so use another cav's info
        if another_cav_is_notable_object == i:
          for cav_id in asker_initial_location_dict:
            if cav_id != asker_cav_id:
              another_cav_location_str = round_to_str(asker_initial_location_dict[cav_id], center_only=False)
          gpt_output += 'There is a car at %s invisible to you. ' % another_cav_location_str
          #print('another_cav_is_notable_object: ', another_cav_is_notable_object)
          #print('notable_gts: ', notable_gts)
          #print('asker_initial_location_dict: ', asker_initial_location_dict)
          #print('asker_cav_id: ', asker_cav_id)
          #print('gpt_output: ', gpt_output)
          #assert False
          # val set not hit, no cav is an invisible object to another cav
        else:
          gpt_output += 'There is a car at %s invisible to you. ' % gt_current_location_str 
          #print('gpt_output: ', gpt_output)

    # end of for loop over notable_gts
    if num_invisible_notable_gts == 0:
      gpt_output += 'There is no notable object invisible to you.'
    #print('gpt_output: ', gpt_output)
    #assert False

  else: # notable_gts is None
    distance_to_waypoint = None
    future_time = None
    gpt_output += 'There is no notable object invisible to you.'
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['distance_to_waypoint'] = distance_to_waypoint.tolist() if distance_to_waypoint is not None else None
  data_dict['future_time'] = future_time.tolist() if future_time is not None else None

  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()

  #print('data_dict: ', data_dict)
  #if notable_gts is not None:
  #  assert False
  #if asker_cav_id == '1':
  #  assert False
  return data_dict, qa_sub_type, future_time, distance_to_waypoint, num_invisible_notable_gts



def generate_3d_grounding_qa_sample_nq1(
  asker_cav_id, initial_lidar_pose,
  notable_gts, notable_gts_future_trajectory,
  cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,      
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
  num_future_waypoints, with_context,
  visible_gt_object_ids_dict, invisible_gt_object_ids_dict,
  asker_initial_location,
  asker_initial_location_dict):
  '''
  NQ1 Visible Notable object identification
  '''


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'I am CAV_EGO. What are the notable objects visible to me near my planned future trajectory [(x0, y0), (x1, y1), ...]?'
      }, {
      'from': 'gpt',
      #'value': 'There is a car at (x0, y0). There is a pedestrian at ...' 
      #'value': 'There is no notable object.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  #print('sources: ', sources)
  default_num_future_waypoints = 6
  # we use future traj in cav_ego's current coordinate system in question
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, default_num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, default_num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)

  asker_initial_location_str = round_to_str(asker_initial_location, center_only=False)
  #print('asker_initial_location_str: ', asker_initial_location_str)
  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO at %s. ' % asker_initial_location_str
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1 at %s. ' % asker_initial_location_str
  else:
    assert False
  human_input += 'What are the notable objects visible to me near my planned future trajectory %s? ' % future_trajectory_str_in_ego

  if with_context:
    assert False
  #print('human_input: ', human_input)

  #print('visible_gt_object_ids_dict: ', visible_gt_object_ids_dict)
  #print('invisible_gt_object_ids_dict: ', invisible_gt_object_ids_dict)
  #print('notable_gts: ', notable_gts)
  # None or (1, 7+1+1+1): box parameter, gt_object_ids, future timestep,  distance to waypoint
  # notable_gts is in cav_ego's coordinate system
  #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
  #assert False

  # Notable object action classification, on steering
  notable_gts_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  notable_gts_action_indices = get_notable_gts_action(notable_gts, notable_gts_future_trajectory, notable_gts_action_classes)
  #assert False
   

  qa_sub_type = notable_gts_action_indices.copy().tolist() if notable_gts_action_indices is not None else [-1]

  gpt_output = ''

  # Answer depends on whether there is notable_gts
  another_cav_is_notable_object = is_another_cav_notable_object(asker_cav_id, asker_initial_location_dict, notable_gts)
  num_visible_notable_gts = 0
  if notable_gts is not None:
    distance_to_waypoint = notable_gts[:, -1]
    future_time = notable_gts[:, -2]

    for i in range(notable_gts.shape[0]):
      notable_gt_future_trajectory_str = get_future_trajectory_str(notable_gts_future_trajectory[i], num_future_waypoints)  
      #print('notable_gt_future_trajectory_str: ', notable_gt_future_trajectory_str)
      gt_current_location_str = round_to_str(notable_gts[i], center_only=True)
      #print('gt_current_location_str: ', gt_current_location_str)

      notable_gt_id = notable_gts[i, 7]
      if notable_gt_id in visible_gt_object_ids_dict[asker_cav_id]:
        num_visible_notable_gts += 1  
        assert(simplified)  
        # check if this notable object is another cav,
        # if so use another cav's info
        if another_cav_is_notable_object == i:
          for cav_id in asker_initial_location_dict:
            if cav_id != asker_cav_id:
              another_cav_location_str = round_to_str(asker_initial_location_dict[cav_id], center_only=False)
          gpt_output += 'There is a car at %s visible to you. ' % another_cav_location_str
          #print('another_cav_is_notable_object: ', another_cav_is_notable_object)
          #print('notable_gts: ', notable_gts)
          #print('asker_initial_location_dict: ', asker_initial_location_dict)
          #print('asker_cav_id: ', asker_cav_id)
          #print('gpt_output: ', gpt_output)
          #assert False
        else:
          gpt_output += 'There is a car at %s visible to you. ' % gt_current_location_str 
        #print('gpt_output: ', gpt_output)

    # end of for loop over notable_gts
    if num_visible_notable_gts == 0:
      gpt_output += 'There is no notable object visible to you.'
    #print('gpt_output: ', gpt_output)
    #assert False

  else: # notable_gts is None
    distance_to_waypoint = None
    future_time = None
    gpt_output += 'There is no notable object visible to you.'
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['distance_to_waypoint'] = distance_to_waypoint.tolist() if distance_to_waypoint is not None else None
  data_dict['future_time'] = future_time.tolist() if future_time is not None else None

  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()

  #print('data_dict: ', data_dict)
  #if notable_gts is not None:
  #  assert False
  #if asker_cav_id == '1':
  #  assert False  
  return data_dict, qa_sub_type, future_time, distance_to_waypoint, num_visible_notable_gts



def generate_3d_grounding_qa_sample_nq2(
  asker_cav_id, initial_lidar_pose,
  notable_gts, notable_gts_future_trajectory,
  cav_future_trajectory_in_ego, cav_future_trajectory_in_self,
  merged_reason_point_in_det_boxes_dict,      
  data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
  num_future_waypoints, with_context,
  visible_gt_object_ids_dict, invisible_gt_object_ids_dict,
  occluding_gt_ids_in_gt_boxes, occluded_gt_ids_in_gt_boxes, gt_boxes, asker_initial_location):
  '''
  NQ2 Occluding object identification
  '''


  # https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md
  data_dict  = {
    'id': data_sample_id,
    # we do not have image for now, may extend to point cloud feature map
    #'image': 'image_path',
    'conversations': [{
      'from': 'human',
      #'value': 'I am CAV_EGO at (X0, Y0). What objects might obstruct my view?'
      }, {
      'from': 'gpt',
      #'value': 'There is a car at (x0, y0) obstructing your view. There is a pedestrian ...' 
      #'value': 'There is no object obstructing your view.'
      }
    ],
    # extra fields not required for llava
    'scenario_index': scenario_index,
    'local_timestamp_index': local_timestamp_index,
    'global_timestamp_index': global_timestamp_index
  }
  #print('sources: ', sources)
  #print('sources: ', sources)
  default_num_future_waypoints = 6
  # we use future traj in cav_ego's current coordinate system in question
  future_trajectory_str_in_ego = get_future_trajectory_str(cav_future_trajectory_in_ego, default_num_future_waypoints)
  #print('future_trajectory_str_in_ego: ', future_trajectory_str_in_ego)
  future_trajectory_str_in_self = get_future_trajectory_str(cav_future_trajectory_in_self, default_num_future_waypoints)
  #print('future_trajectory_str_in_self: ', future_trajectory_str_in_self)

  asker_initial_location_str = round_to_str(asker_initial_location, center_only=False)
  #print('asker_initial_location_str: ', asker_initial_location_str)
  if asker_cav_id == 'ego':
    human_input = 'I am CAV_EGO at %s. ' % asker_initial_location_str
  elif asker_cav_id == '1':
    human_input = 'I am CAV_1 at %s. ' % asker_initial_location_str
  else:
    assert False
  human_input += 'What objects might obstruct my view?'

  if with_context:
    assert False
  #print('human_input: ', human_input)

  #print('visible_gt_object_ids_dict: ', visible_gt_object_ids_dict)
  #print('invisible_gt_object_ids_dict: ', invisible_gt_object_ids_dict)
  #print('notable_gts: ', notable_gts)
  # None or (1, 7+1+1+1): box parameter, gt_object_ids, future timestep,  distance to waypoint
  # notable_gts is in cav_ego's coordinate system
  #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
  #assert False

  # Notable object action classification, on steering
  notable_gts_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  notable_gts_action_indices = get_notable_gts_action(notable_gts, notable_gts_future_trajectory, notable_gts_action_classes)
  #assert False
   

  qa_sub_type = notable_gts_action_indices.copy().tolist() if notable_gts_action_indices is not None else [-1]

  gpt_output = ''
  # Answer depends on whether there is occluding_gt_ids_in_gt_boxes
  if len(occluding_gt_ids_in_gt_boxes) > 0:

    for gt_id_in_gt_boxes in occluding_gt_ids_in_gt_boxes:
      gt_current_location_str = round_to_str(gt_boxes[gt_id_in_gt_boxes], center_only=True)
      #print('gt_current_location_str: ', gt_current_location_str)
      assert(simplified)
      gpt_output += 'There is a car at %s obstructing your view. ' % gt_current_location_str 
      #print('gpt_output: ', gpt_output)

  else: 
    gpt_output += 'There is no object obstructing your view.'
  #print('gpt_output: ', gpt_output)

  data_dict['conversations'][0]['value'] = human_input
  data_dict['conversations'][1]['value'] = gpt_output
  data_dict['qa_sub_type'] = qa_sub_type

  data_dict['future_trajectory_str_in_ego'] = future_trajectory_str_in_ego
  data_dict['future_trajectory_str_in_self'] = future_trajectory_str_in_self
  data_dict['asker_cav_id'] = asker_cav_id

  data_dict['cav_ego_lidar_pose'] = initial_lidar_pose['ego'].flatten().tolist()
  data_dict['cav_1_lidar_pose'] = initial_lidar_pose['1'].flatten().tolist()

  #print('data_dict: ', data_dict)
  num_occluding_gts = len(occluding_gt_ids_in_gt_boxes)
  future_time = None
  distance_to_waypoint = None
  #assert False
  return data_dict, qa_sub_type, future_time, distance_to_waypoint, num_occluding_gts


def generate_3d_grounding_qa_dataset_v6(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  For cav_ego's planned future trajectory in the future 3 seconds,
  check whether there are gt boxes close or overlap with the path

  planned future trajectory can be:
  1. gt future trajectory in the current frame's coordinate
  2. motion premitives: (assume constant speed for now)
    1. forward 
    2. left turn
    3. right turn

  Starting from some simpler cases.  

  Q: Is there anything I need to be aware of if I am driving forward?
  Q: Is there anything I need to be aware of if I am turning left?
  Q: Is there anything I need to be aware of if my planned future trajectory is [(x0, z0), (x1, z1), (x2, z2)]?

  A: Yes. There is a car at the location [x, z], which is close to your future path. CAV_1 detects it.
  A: There is nothing you need to be aware of.
  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_v6'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, exp_name + '.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json

  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []
  list_data_dict_save_file_p1 = os.path.join(llm_data_path, exp_name + '_p1.json')
  list_data_dict_save_file_p2 = os.path.join(llm_data_path, exp_name + '_p2.json')
  list_data_dict_save_file_p3 = os.path.join(llm_data_path, exp_name + '_p3.json')
  list_data_dict_save_file_p4 = os.path.join(llm_data_path, exp_name + '_p4.json')


  data_sample_id = 0

  motion_names = ['moving forward', 'turning left', 'turning right']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(8)
  future_time_stats = []
  distance_to_waypoint_stats = []

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, _, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index)
      print('gt_boxes: ', gt_boxes)
      print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)
      assert False

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      cav_ego_future_trajectory = get_cav_ego_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames)
      #print('cav_ego_future_trajectory: ', cav_ego_future_trajectory)
      #print('cav_ego_future_trajectory.shape: ', cav_ego_future_trajectory.shape)
      # (num_future_frames, 2)

      notable_gts = get_notable_gts_near_cav_future_trajectory(cav_ego_future_trajectory, gt_boxes, gt_box_corners, max_num_answer_objects)
      if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
        max_num_notable_gts_stats = notable_gts.shape[0]
        print('notable_gts: ', notable_gts)

      # check whether we have positive samples
      if notable_gts is None:
        # if no notable_gts, check any cav detects something at a future waypoint 
        # may be just show 3 way points in question
        # the first one is 9th waypoint for 1 second in the future
        merged_reason_points = cav_ego_future_trajectory[9:10, :]
        #print('merged_reason_points: ', merged_reason_points)
        # (1, 2)
        #assert False
      else:
        # check any cav detects the notable_gts
        # for generating the reasons
        merged_reason_points = np.stack([
          notable_gts[:, 3],
          notable_gts[:, 5]
        ], axis=1)
        #print('merged_reason_points: ', merged_reason_points)
        # (1, 2)
        #assert False


      merged_reason_point_in_det_boxes_dict = dict()
      for cav_id in cav_ids:
        merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
          merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
      #if notable_gts is not None and notable_gts.shape[0] > 1:  
      #  print('notable_gts: ', notable_gts)  
      #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
      #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
      #  assert False


      # at one frame we only generate zero or one notable object and 1 qa for now
      #assert(notable_gts is None or notable_gts.shape[0] == 1)
      #for i in range(sample_box_region_idx.shape[0]):
      # here we only generate one qa sample using the first notable_gts
      #for i in range(1):
      # generate one qa using all notable_gts
      if True:
        qa_sample_data_dict, qa_sub_type, future_time, distance_to_waypoint = generate_3d_grounding_qa_sample_v6(
            notable_gts, cav_ego_future_trajectory,
            merged_reason_point_in_det_boxes_dict,
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified)
        qa_sub_type = np.array(qa_sub_type)
        if np.any(qa_sub_type < 4):
          if np.any(qa_sub_type == 2): # interesting case only cav_1 detects it
            print('qa_sample_data_dict: ', qa_sample_data_dict)
            #assert False

        # TODO enable downsample after we get the stats
        # Downsample negative samples
        #if downsample_negatives:
        #  if qa_sub_type >= 4:
        #    p = 1.0 * 7197 / 19991
        #    if random.uniform(0, 1) > p:
        #      # discard this negative example  
        #      continue

        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        for i in range(qa_sub_type.shape[0]):
          qa_sub_type_counter[qa_sub_type[i]] += 1
          if qa_sub_type[i] == 2: #  intersting case that needs cav_1
            distance_to_waypoint_stats.append(distance_to_waypoint[i])
            future_time_stats.append(future_time[i])

        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        # folowing merged to above
        #if np.any(qa_sub_type) == 2: # intersting case that needs cav_1
        #  if distance_to_waypoint is not None:
        #    distance_to_waypoint_stats.append(distance_to_waypoint)
        #  if future_time is not None:
        #    future_time_stats.append(future_time)  

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  distance_to_waypoint_stats = sorted(distance_to_waypoint_stats)
  print('distance_to_waypoint_stats: ', distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats[:10]: ', distance_to_waypoint_stats[:10])
  #print('distance_to_waypoint_stats[-10:]: ', distance_to_waypoint_stats[-10:])
  future_time_stats = sorted(future_time_stats)
  print('future_time_stats: ', future_time_stats)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return



def generate_3d_grounding_qa_dataset_v6_double(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, double_cavs):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  For cav_ego's planned future trajectory in the future 3 seconds,
  check whether there are gt boxes close or overlap with the path

  planned future trajectory can be:
  1. gt future trajectory in the current frame's coordinate
  2. motion premitives: (assume constant speed for now)
    1. forward 
    2. left turn
    3. right turn

  Starting from some simpler cases.  

  Q: Is there anything I need to be aware of if I am driving forward?
  Q: Is there anything I need to be aware of if I am turning left?
  Q: Is there anything I need to be aware of if my planned future trajectory is [(x0, z0), (x1, z1), (x2, z2)]?

  A: Yes. There is a car at the location [x, z], which is close to your future path. CAV_1 detects it.
  A: There is nothing you need to be aware of.
  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_v6'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)
  if double_cavs:
    exp_name += 'double'

  # cav_1's answer exclude cav_1 and include cav_ego
  exp_name += 'new'  


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, exp_name + '.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json

  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []
  list_data_dict_save_file_p1 = os.path.join(llm_data_path, exp_name + '_p1.json')
  list_data_dict_save_file_p2 = os.path.join(llm_data_path, exp_name + '_p2.json')
  list_data_dict_save_file_p3 = os.path.join(llm_data_path, exp_name + '_p3.json')
  list_data_dict_save_file_p4 = os.path.join(llm_data_path, exp_name + '_p4.json')


  data_sample_id = 0

  motion_names = ['moving forward', 'turning left', 'turning right']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(8)
  future_time_stats = []
  distance_to_waypoint_stats = []

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):

      # MY_DEBUG
      #if global_timestamp_index != 1962:
      #  continue  

      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, _, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)

      # notable gts in ego coordinate system
      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      for asker_cav_id in cav_ids:
        # get both cav's current location in cav_ego's current coordinate
        lidar_pose_in_ego_initial_frame = x1_to_x2(initial_lidar_pose[asker_cav_id], initial_lidar_pose['ego'])
        asker_initial_location = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
        #print('asker_cav_id: ', asker_cav_id)
        #print('asker_initial_location: ', asker_initial_location)

        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, max_num_answer_objects, asker_initial_location)
        #if asker_cav_id == '1':
        #  assert False
        asker_notable_gts_dict[asker_cav_id] = notable_gts
        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False

        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict



      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id
      for asker_cav_id in cav_ids:
        qa_sample_data_dict, qa_sub_type, future_time, distance_to_waypoint = generate_3d_grounding_qa_sample_v6_double(
            asker_cav_id, initial_lidar_pose,
            asker_notable_gts_dict[asker_cav_id],
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],    
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified)
        qa_sub_type = np.array(qa_sub_type)
        #assert False
        if np.any(qa_sub_type < 4):
          if np.any(qa_sub_type == 2): # interesting case only cav_1 detects it
            print('qa_sample_data_dict: ', qa_sample_data_dict)
            #assert False

        # TODO enable downsample after we get the stats
        # Downsample negative samples
        #if downsample_negatives:
        #  if qa_sub_type >= 4:
        #    p = 1.0 * 7197 / 19991
        #    if random.uniform(0, 1) > p:
        #      # discard this negative example  
        #      continue

        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        for i in range(qa_sub_type.shape[0]):
          qa_sub_type_counter[qa_sub_type[i]] += 1
          if qa_sub_type[i] == 2: #  intersting case that needs cav_1
            distance_to_waypoint_stats.append(distance_to_waypoint[i])
            future_time_stats.append(future_time[i])

        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        # folowing merged to above
        #if np.any(qa_sub_type) == 2: # intersting case that needs cav_1
        #  if distance_to_waypoint is not None:
        #    distance_to_waypoint_stats.append(distance_to_waypoint)
        #  if future_time is not None:
        #    future_time_stats.append(future_time)  

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  distance_to_waypoint_stats = sorted(distance_to_waypoint_stats)
  print('distance_to_waypoint_stats: ', distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats[:10]: ', distance_to_waypoint_stats[:10])
  #print('distance_to_waypoint_stats[-10:]: ', distance_to_waypoint_stats[-10:])
  future_time_stats = sorted(future_time_stats)
  print('future_time_stats: ', future_time_stats)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return



def generate_3d_grounding_qa_dataset_v7(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  Output: cav_ego's gt future trajectory in the future 3 seconds,
  num_future_waypoints = 6
  one waypoint for every future 0.5 seconds

  Starting from an easy version without hierarchical reasoning (notable objects)

  Q: What is the suggested future trajectory to avoid collision with nearby objects?
  A: The suggested future trajectory is [(x0, z0), (x1, z1), (x2, z2), (x3, z3), (x4, z4), (x5, z5)].

  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_v7'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)  
  exp_name += 'w' + str(num_future_waypoints)


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, exp_name + '.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v7.json

  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []
  list_data_dict_save_file_p1 = os.path.join(llm_data_path, exp_name + '_p1.json')
  list_data_dict_save_file_p2 = os.path.join(llm_data_path, exp_name + '_p2.json')
  list_data_dict_save_file_p3 = os.path.join(llm_data_path, exp_name + '_p3.json')
  list_data_dict_save_file_p4 = os.path.join(llm_data_path, exp_name + '_p4.json')


  data_sample_id = 0

  # not used so far
  motion_names = ['moving forward', 'turning left', 'turning right']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(8)
  future_time_stats = []
  distance_to_waypoint_stats = []

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, _, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      cav_ego_future_trajectory = get_cav_ego_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames)
      #print('cav_ego_future_trajectory: ', cav_ego_future_trajectory)
      #print('cav_ego_future_trajectory.shape: ', cav_ego_future_trajectory.shape)
      # (num_future_frames, 2)


      notable_gts = get_notable_gts_near_cav_future_trajectory(cav_ego_future_trajectory, gt_boxes, gt_box_corners, max_num_answer_objects)
      if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
        max_num_notable_gts_stats = notable_gts.shape[0]
        print('notable_gts: ', notable_gts)

      # check whether we have positive samples
      if notable_gts is None:
        # if no notable_gts, check any cav detects something at a future waypoint 
        # may be just show 3 way points in question
        # the first one is 9th waypoint for 1 second in the future
        merged_reason_points = cav_ego_future_trajectory[9:10, :]
        #print('merged_reason_points: ', merged_reason_points)
        # (1, 2)
        #assert False
      else:
        # check any cav detects the notable_gts
        # for generating the reasons
        merged_reason_points = np.stack([
          notable_gts[:, 3],
          notable_gts[:, 5]
        ], axis=1)
        #print('merged_reason_points: ', merged_reason_points)
        # (1, 2)
        #assert False


      merged_reason_point_in_det_boxes_dict = dict()
      for cav_id in cav_ids:
        merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
          merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
      #if notable_gts is not None and notable_gts.shape[0] > 1:  
      #  print('notable_gts: ', notable_gts)  
      #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
      #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
      #  assert False


      # at one frame we only generate zero or one notable object and 1 qa for now
      #assert(notable_gts is None or notable_gts.shape[0] == 1)
      #for i in range(sample_box_region_idx.shape[0]):
      # here we only generate one qa sample using the first notable_gts
      #for i in range(1):
      # generate one qa using all notable_gts
      if True:
        qa_sample_data_dict, qa_sub_type, future_time, distance_to_waypoint = generate_3d_grounding_qa_sample_v7(
            notable_gts, cav_ego_future_trajectory,
            merged_reason_point_in_det_boxes_dict,
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified)
        qa_sub_type = np.array(qa_sub_type)
        #assert False
        if np.any(qa_sub_type < 4):
          if np.any(qa_sub_type == 2): # interesting case only cav_1 detects it
            print('qa_sample_data_dict: ', qa_sample_data_dict)
            #assert False

        # TODO enable downsample after we get the stats
        # Downsample negative samples
        #if downsample_negatives:
        #  if qa_sub_type >= 4:
        #    p = 1.0 * 7197 / 19991
        #    if random.uniform(0, 1) > p:
        #      # discard this negative example  
        #      continue

        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        for i in range(qa_sub_type.shape[0]):
          qa_sub_type_counter[qa_sub_type[i]] += 1
          if qa_sub_type[i] == 2: #  intersting case that needs cav_1
            distance_to_waypoint_stats.append(distance_to_waypoint[i])
            future_time_stats.append(future_time[i])

        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        # folowing merged to above
        #if np.any(qa_sub_type) == 2: # intersting case that needs cav_1
        #  if distance_to_waypoint is not None:
        #    distance_to_waypoint_stats.append(distance_to_waypoint)
        #  if future_time is not None:
        #    future_time_stats.append(future_time)  

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  distance_to_waypoint_stats = sorted(distance_to_waypoint_stats)
  print('distance_to_waypoint_stats: ', distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats[:10]: ', distance_to_waypoint_stats[:10])
  #print('distance_to_waypoint_stats[-10:]: ', distance_to_waypoint_stats[-10:])
  future_time_stats = sorted(future_time_stats)
  print('future_time_stats: ', future_time_stats)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return




def generate_3d_grounding_qa_dataset_v7_double(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints, double_cavs):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  Output: cav_ego's gt future trajectory in the future 3 seconds,
  num_future_waypoints = 6
  one waypoint for every future 0.5 seconds

  Starting from an easy version without hierarchical reasoning (notable objects)

  Q: What is the suggested future trajectory to avoid collision with nearby objects?
  A: The suggested future trajectory is [(x0, z0), (x1, z1), (x2, z2), (x3, z3), (x4, z4), (x5, z5)].

  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_v7'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)  
  exp_name += 'w' + str(num_future_waypoints)
  if double_cavs:
    exp_name += 'double'  


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, exp_name + '.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v7.json

  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []
  list_data_dict_save_file_p1 = os.path.join(llm_data_path, exp_name + '_p1.json')
  list_data_dict_save_file_p2 = os.path.join(llm_data_path, exp_name + '_p2.json')
  list_data_dict_save_file_p3 = os.path.join(llm_data_path, exp_name + '_p3.json')
  list_data_dict_save_file_p4 = os.path.join(llm_data_path, exp_name + '_p4.json')


  data_sample_id = 0

  # not used so far
  motion_names = ['moving forward', 'turning left', 'turning right']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(8)
  future_time_stats = []
  distance_to_waypoint_stats = []

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, _, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)


      # So far our planning QA does not include reason in answer
      # Here still get reason based on notable objects
      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      for asker_cav_id in cav_ids:
        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, max_num_answer_objects)
        asker_notable_gts_dict[asker_cav_id] = notable_gts
        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False

        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict



      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id 
      for asker_cav_id in cav_ids:
        qa_sample_data_dict, qa_sub_type, future_time, distance_to_waypoint = generate_3d_grounding_qa_sample_v7_double(
            asker_cav_id, initial_lidar_pose,   
            asker_notable_gts_dict[asker_cav_id], 
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified)
        qa_sub_type = np.array(qa_sub_type)
        #assert False

        if np.any(qa_sub_type < 4):
          if np.any(qa_sub_type == 2): # interesting case only cav_1 detects it
            print('qa_sample_data_dict: ', qa_sample_data_dict)
            #assert False

        # TODO enable downsample after we get the stats
        # Downsample negative samples
        #if downsample_negatives:
        #  if qa_sub_type >= 4:
        #    p = 1.0 * 7197 / 19991
        #    if random.uniform(0, 1) > p:
        #      # discard this negative example  
        #      continue

        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        for i in range(qa_sub_type.shape[0]):
          qa_sub_type_counter[qa_sub_type[i]] += 1
          if qa_sub_type[i] == 2: #  intersting case that needs cav_1
            distance_to_waypoint_stats.append(distance_to_waypoint[i])
            future_time_stats.append(future_time[i])

        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        # folowing merged to above
        #if np.any(qa_sub_type) == 2: # intersting case that needs cav_1
        #  if distance_to_waypoint is not None:
        #    distance_to_waypoint_stats.append(distance_to_waypoint)
        #  if future_time is not None:
        #    future_time_stats.append(future_time)  

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  distance_to_waypoint_stats = sorted(distance_to_waypoint_stats)
  print('distance_to_waypoint_stats: ', distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats[:10]: ', distance_to_waypoint_stats[:10])
  #print('distance_to_waypoint_stats[-10:]: ', distance_to_waypoint_stats[-10:])
  future_time_stats = sorted(future_time_stats)
  print('future_time_stats: ', future_time_stats)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return



def generate_3d_grounding_qa_dataset_nq8(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints, double_cavs, with_context, context_list, output_file, context_list_from_gt):
  '''
  # NQ8: Suggested Action Classification
  # Suggested Speed and Steering based on CAV's GT future trajectory

  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  num_future_waypoints = 6
  one waypoint for every future 0.5 seconds

  Q: What are the suggested speed and steering to avoid collision with nearby objects? CONTEXT: NQ7 + ANSWER
  A: The suggested speed is [very fast, fast, moderate, slow, very slow]. 
     The suggested steering is [left, slightly left, straight, slightly right, right].


  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_nq8'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)  
  exp_name += 'w' + str(num_future_waypoints)
  if double_cavs:
    exp_name += 'd'  
  if with_context:
    exp_name += 'c'


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []

  if with_context and context_list is not None:
    # generating temp qa for graph inference
    list_data_dict_save_file = output_file
    list_data_dict_save_file_p1 = list_data_dict_save_file[:-5] + '_p1.json'
    list_data_dict_save_file_p2 = list_data_dict_save_file[:-5] + '_p2.json'
    list_data_dict_save_file_p3 = list_data_dict_save_file[:-5] + '_p3.json'
    list_data_dict_save_file_p4 = list_data_dict_save_file[:-5] + '_p4.json'
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)
  else:
    # generating training and validation qa
    output_save_path = llm_data_path
    list_data_dict_save_file = os.path.join(output_save_path, exp_name + '.json')
    list_data_dict_save_file_p1 = os.path.join(output_save_path, exp_name + '_p1.json')
    list_data_dict_save_file_p2 = os.path.join(output_save_path, exp_name + '_p2.json')
    list_data_dict_save_file_p3 = os.path.join(output_save_path, exp_name + '_p3.json')
    list_data_dict_save_file_p4 = os.path.join(output_save_path, exp_name + '_p4.json')
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)



  data_sample_id = 0

  speed_classes = ['very fast', 'fast', 'moderate', 'slow', 'very slow']
  steering_classes = ['left', 'slightly left', 'straight', 'slightly right', 'right']
  num_actions = len(speed_classes) * len(steering_classes)
  qa_sub_type_counter = np.zeros(num_actions)
  qa_sub_type_speed_counter = np.zeros(len(speed_classes))
  qa_sub_type_steering_counter = np.zeros(len(steering_classes))
  future_time_stats = []
  distance_to_waypoint_stats = []

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, gt_object_ids, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)


      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      asker_notable_gts_future_trajectory_dict = dict()
      asker_initial_location_dict = dict()
      for asker_cav_id in cav_ids:
        # get both cav's current location in cav_ego's current coordinate
        lidar_pose_in_ego_initial_frame = x1_to_x2(initial_lidar_pose[asker_cav_id], initial_lidar_pose['ego'])
        asker_initial_location = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
        asker_initial_location_dict[asker_cav_id] = asker_initial_location.copy()
        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, gt_object_ids, max_num_answer_objects, asker_initial_location)
        asker_notable_gts_dict[asker_cav_id] = notable_gts
        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        notable_gts_future_trajectory = get_notable_gts_future_trajectory(notable_gts, npy_save_path, global_timestamp_index, num_future_frames, cav_ids)
        #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
        #assert False
        asker_notable_gts_future_trajectory_dict[asker_cav_id] = notable_gts_future_trajectory

        # Old reason code
        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False

        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict
        # END of old reason code  



      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id 
      for asker_cav_id in cav_ids:

        # if we are generating temp qa for graph inference
        if with_context and context_list is not None:
          context_sample_list = []
          for context in context_list:
            context_sample = context[data_sample_id]
            assert(context_sample['id'] == data_sample_id)
            assert(context_sample['global_timestamp_index'] == global_timestamp_index)
            assert(context_sample['asker_cav_id'] == asker_cav_id)
            context_sample_list.append(context_sample)
        else:
          context_sample_list = None

        qa_sample_data_dict, qa_sub_type, suggested_speed_idx, suggested_steering_idx = generate_3d_grounding_qa_sample_nq8(
            asker_cav_id, initial_lidar_pose,   
            asker_notable_gts_dict[asker_cav_id], asker_notable_gts_future_trajectory_dict[asker_cav_id],
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
            with_context,
            num_future_waypoints, asker_initial_location_dict[asker_cav_id],
            context_sample_list, context_list_from_gt,
            asker_initial_location_dict, double_cavs_future_trajectory_in_ego_current)
        qa_sub_type = np.array(qa_sub_type)

        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        qa_sub_type_counter[qa_sub_type] += 1
        qa_sub_type_speed_counter[suggested_speed_idx] += 1
        qa_sub_type_steering_counter[suggested_steering_idx] += 1
        #assert False

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  print('qa_sub_type_speed_counter: ', qa_sub_type_speed_counter)
  print('qa_sub_type_steering_counter: ', qa_sub_type_steering_counter)

  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return


def generate_3d_grounding_qa_dataset_nq9(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints, double_cavs, with_context, context_list, output_file, context_list_from_gt):
  '''
  # NQ9: Suggested Trajectory
  # Question include context from NQ8 (Suggested Action Classification) as context

  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  num_future_waypoints = 6
  one waypoint for every future 0.5 seconds

  Q: What is the suggested trajectory to avoid collision with nearby objects? CONTEXT: NQ8 + ANSWER
  A: The suggested trajectory is [(x0, y0), (x1, y1), ..., (x5, y5))] + REASON


  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_nq9'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)  
  exp_name += 'w' + str(num_future_waypoints)
  if double_cavs:
    exp_name += 'd'  
  if with_context:
    exp_name += 'c'  



  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []

  if with_context and context_list is not None:
    # generating temp qa for graph inference
    list_data_dict_save_file = output_file
    list_data_dict_save_file_p1 = list_data_dict_save_file[:-5] + '_p1.json'
    list_data_dict_save_file_p2 = list_data_dict_save_file[:-5] + '_p2.json'
    list_data_dict_save_file_p3 = list_data_dict_save_file[:-5] + '_p3.json'
    list_data_dict_save_file_p4 = list_data_dict_save_file[:-5] + '_p4.json'
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)
  else:
    # generating training and validation qa
    output_save_path = llm_data_path
    list_data_dict_save_file = os.path.join(output_save_path, exp_name + '.json')
    list_data_dict_save_file_p1 = os.path.join(output_save_path, exp_name + '_p1.json')
    list_data_dict_save_file_p2 = os.path.join(output_save_path, exp_name + '_p2.json')
    list_data_dict_save_file_p3 = os.path.join(output_save_path, exp_name + '_p3.json')
    list_data_dict_save_file_p4 = os.path.join(output_save_path, exp_name + '_p4.json')
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)


  data_sample_id = 0

  speed_classes = ['very fast', 'fast', 'moderate', 'slow', 'very slow']
  steering_classes = ['left', 'slightly left', 'straight', 'slightly right', 'right']
  num_actions = len(speed_classes) * len(steering_classes)
  qa_sub_type_counter = np.zeros(num_actions)
  qa_sub_type_speed_counter = np.zeros(len(speed_classes))
  qa_sub_type_steering_counter = np.zeros(len(steering_classes))
  future_time_stats = []
  distance_to_waypoint_stats = []

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  
      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, gt_object_ids, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)


      # So far our planning QA does not include reason in answer
      # Here still get reason based on notable objects
      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      asker_initial_location_dict = dict()
      for asker_cav_id in cav_ids:
        # get both cav's current location in cav_ego's current coordinate
        lidar_pose_in_ego_initial_frame = x1_to_x2(initial_lidar_pose[asker_cav_id], initial_lidar_pose['ego'])
        asker_initial_location = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
        asker_initial_location_dict[asker_cav_id] = asker_initial_location.copy()
        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, gt_object_ids, max_num_answer_objects, asker_initial_location)
        asker_notable_gts_dict[asker_cav_id] = notable_gts
        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False

        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict
      # END of old reason code  



      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id 
      for asker_cav_id in cav_ids:

        # if we are generating temp qa for graph inference
        if with_context and context_list is not None:
          context_sample_list = []
          for context in context_list:
            context_sample = context[data_sample_id]
            assert(context_sample['id'] == data_sample_id)
            assert(context_sample['global_timestamp_index'] == global_timestamp_index)
            assert(context_sample['asker_cav_id'] == asker_cav_id)
            context_sample_list.append(context_sample)
        else:
          context_sample_list = None

        qa_sample_data_dict, qa_sub_type, suggested_speed_idx, suggested_steering_idx = generate_3d_grounding_qa_sample_nq9(
            asker_cav_id, initial_lidar_pose,   
            asker_notable_gts_dict[asker_cav_id], 
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
            with_context,
            asker_initial_location_dict[asker_cav_id],
            context_sample_list, context_list_from_gt)
        qa_sub_type = np.array(qa_sub_type)

        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        qa_sub_type_counter[qa_sub_type] += 1
        qa_sub_type_speed_counter[suggested_speed_idx] += 1
        qa_sub_type_steering_counter[suggested_steering_idx] += 1
        #assert False

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  print('qa_sub_type_speed_counter: ', qa_sub_type_speed_counter)
  print('qa_sub_type_steering_counter: ', qa_sub_type_steering_counter)

  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return


def load_gt_object_ids(npy_save_path, global_timestamp_index):
  gt_object_id_file = os.path.join(npy_save_path, '%04d_gt_object_id.npy' % global_timestamp_index)
  gt_object_ids = np.load(gt_object_id_file)
  return gt_object_ids


def get_notable_gts_future_trajectory(notable_gts, npy_save_path, global_timestamp_index, num_future_frames, cav_ids):
    '''
    Input:
      notable_gts: (N, 7+1+1+1), box parameters, gt_object_id, future time step, dist to cav
      box parameters: [h, w, l, x, y, z, a], (x, z) is the 2d location
    Output:
      notable_gts_future_trajectory: (N, num_future_frames, 2) including current 2d location
    '''
    #print('notable_gts: ', notable_gts)
    if notable_gts is None:
      return None  


    num_notable_gts = notable_gts.shape[0]
    notable_gts_future_trajectory = np.zeros([num_notable_gts, num_future_frames, 2])

    # current location
    notable_gts_future_trajectory[:, 0, 0] = notable_gts[:, 3].copy()
    notable_gts_future_trajectory[:, 0, 1] = notable_gts[:, 5].copy()

    # future location
    # need to transform to cav_ego coordinate system at global_timestamp_index
    current_lidar_pose_ego_in_world = np.load(os.path.join(npy_save_path, 'ego', '%04d_lidar_pose.npy' % global_timestamp_index))
    #print('current_lidar_pose_ego_in_world: ', current_lidar_pose_ego_in_world)
    for future_timestamp_index in range(global_timestamp_index+1, global_timestamp_index+num_future_frames+1):
      future_lidar_pose_ego_in_world = np.load(os.path.join(npy_save_path, 'ego', '%04d_lidar_pose.npy' % future_timestamp_index))
      #print('future_lidar_pose_ego_in_world: ', future_lidar_pose_ego_in_world)
      future_lidar_pose_ego_in_current_ego = x1_to_x2(future_lidar_pose_ego_in_world, current_lidar_pose_ego_in_world)
      #print('future_lidar_pose_ego_in_current_ego: ', future_lidar_pose_ego_in_current_ego)

      # find matching future gt location
      future_gt_boxes_3d, _, future_gt_object_ids, _, _ = load_gt_boxes_3d(npy_save_path, future_timestamp_index, cav_ids)
      #future_object_ids = load_gt_object_ids(npy_save_path, future_timestamp_index)
      #print('future_gt_boxes_3d: ', future_gt_boxes_3d)
      #print('future_gt_object_ids: ', future_gt_object_ids)
      for i in range(num_notable_gts):
       notable_gt_object_id = notable_gts[i, 7]
       #print('notable_gt_object_id: '), notable_gt_object_id
       found_matching_gt = False
       for future_i in range(future_gt_object_ids.shape[0]):
         if future_gt_object_ids[future_i] == notable_gt_object_id:
           # found matching gt
           # get 2d location and transform coordinate
           found_matching_gt = True
           future_location = np.array([future_gt_boxes_3d[future_i][3], future_gt_boxes_3d[future_i][5], 0])
           #print('future_location: ', future_location)
           future_location = np.expand_dims(future_location, axis=0)
           #print('future_location: ', future_location)
           future_location_in_current_ego = project_points_by_matrix_torch(future_location, future_lidar_pose_ego_in_current_ego)
           future_location_in_current_ego = future_location_in_current_ego[0][:2]
           #print('future_location_in_current_ego: ', future_location_in_current_ego)
           notable_gts_future_trajectory[i, future_timestamp_index - global_timestamp_index - 1] = future_location_in_current_ego.copy()
           #assert False 
           break
       if not found_matching_gt:
         #print('future notable gt not found')
         # out of gt annotation region or gt annotation error
         # assume that notable object is static
         if future_timestamp_index - global_timestamp_index - 1 > 0:
           notable_gts_future_trajectory[i, future_timestamp_index - global_timestamp_index - 1] = notable_gts_future_trajectory[i, future_timestamp_index - global_timestamp_index - 2]
         #assert False

    #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
    #assert False
    return notable_gts_future_trajectory


def generate_3d_grounding_qa_dataset_nq5(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints, double_cavs, with_context, context_list, output_file, context_list_from_gt):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  NQ5: Prediction by Perception 
  Q: Where might the notable objects move in the future given my planned future trajectory [(x0, y0), (x1, y1), ...]? 
  A: There is a car [moving forward|turning left|turning right]. The predicted future trajecoty is [(x0, y0), ...]
     There is a pedestrian [moving forward|turning left|turning right]. The predicted future trajecoty is [(x0, y0), ...]


  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_nq5'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)
  exp_name += 'w' + str(num_future_waypoints)
  if double_cavs:
    exp_name += 'd'
  if with_context:
    exp_name += 'c'  


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []

  if with_context and context_list is not None:
    # generating temp qa for graph inference
    list_data_dict_save_file = output_file
    list_data_dict_save_file_p1 = list_data_dict_save_file[:-5] + '_p1.json'
    list_data_dict_save_file_p2 = list_data_dict_save_file[:-5] + '_p2.json'
    list_data_dict_save_file_p3 = list_data_dict_save_file[:-5] + '_p3.json'
    list_data_dict_save_file_p4 = list_data_dict_save_file[:-5] + '_p4.json'
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)
  else:
    # generating training and validation qa
    output_save_path = llm_data_path
    list_data_dict_save_file = os.path.join(output_save_path, exp_name + '.json')
    list_data_dict_save_file_p1 = os.path.join(output_save_path, exp_name + '_p1.json')
    list_data_dict_save_file_p2 = os.path.join(output_save_path, exp_name + '_p2.json')
    list_data_dict_save_file_p3 = os.path.join(output_save_path, exp_name + '_p3.json')
    list_data_dict_save_file_p4 = os.path.join(output_save_path, exp_name + '_p4.json')
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)


  data_sample_id = 0

  motion_names = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(4)
  qa_sub_type_no_notable_object_counter = 0
  qa_sub_type_notable_object_counter = 0
  future_time_stats = []
  distance_to_waypoint_stats = []

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    # MY_DEBUG
    #if scenario_index != 4:
    #  continue  

    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  

      # MY_DEBUG
      #if global_timestamp_index != 1962:
      #  continue  

      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, gt_object_ids, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)
      #print('gt_object_ids: ', gt_object_ids)
      #assert False

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)

      # notable gts in ego coordinate system
      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      asker_notable_gts_future_trajectory_dict = dict()
      asker_initial_location_dict = dict()
      for asker_cav_id in cav_ids:
        # get both cav's current location in cav_ego's current coordinate
        lidar_pose_in_ego_initial_frame = x1_to_x2(initial_lidar_pose[asker_cav_id], initial_lidar_pose['ego'])
        asker_initial_location = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
        asker_initial_location_dict[asker_cav_id] = asker_initial_location.copy()
        #print('asker_cav_id: ', asker_cav_id)
        #print('asker_initial_location: ', asker_initial_location)

        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, gt_object_ids, max_num_answer_objects, asker_initial_location)
        #print('notable_gts: ', notable_gts)
        # (N, 7+1+1+1), box parameters, gt_object_id, future time step, dist to cav
        #assert False

        notable_gts_future_trajectory = get_notable_gts_future_trajectory(notable_gts, npy_save_path, global_timestamp_index, num_future_frames, cav_ids)
        #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
        #assert False

        asker_notable_gts_dict[asker_cav_id] = notable_gts
        asker_notable_gts_future_trajectory_dict[asker_cav_id] = notable_gts_future_trajectory

        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        # Old Reason
        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict



      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id
      for asker_cav_id in cav_ids:

        # if we are generating temp qa for graph inference
        if with_context and context_list is not None:
          context_sample_list = []
          for context in context_list:
            context_sample = context[data_sample_id]
            assert(context_sample['id'] == data_sample_id)
            assert(context_sample['global_timestamp_index'] == global_timestamp_index)
            assert(context_sample['asker_cav_id'] == asker_cav_id)
            context_sample_list.append(context_sample)
        else:
          context_sample_list = None

        qa_sample_data_dict, qa_sub_type, future_time, distance_to_waypoint = generate_3d_grounding_qa_sample_nq5(
            asker_cav_id, initial_lidar_pose,
            asker_notable_gts_dict[asker_cav_id],
            asker_notable_gts_future_trajectory_dict[asker_cav_id],
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],    
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
            num_future_waypoints, with_context, asker_initial_location_dict[asker_cav_id],
            context_sample_list, context_list_from_gt, 
            asker_initial_location_dict, double_cavs_future_trajectory_in_ego_current)
        qa_sub_type = np.array(qa_sub_type)
        #assert False
        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        if qa_sub_type[0] == -1:
          qa_sub_type_no_notable_object_counter += 1
        else:  
          qa_sub_type_notable_object_counter += len(qa_sub_type)  
          #print('qa_sub_type: ', qa_sub_type)
          for idx in range(len(qa_sub_type)):
             qa_sub_type_counter[int(qa_sub_type[idx])] += 1

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  print('qa_sub_type_notable_object_counter: ', qa_sub_type_notable_object_counter)
  print('qa_sub_type_no_notable_object_counter: ', qa_sub_type_no_notable_object_counter)
  #distance_to_waypoint_stats = sorted(distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats: ', distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats[:10]: ', distance_to_waypoint_stats[:10])
  #print('distance_to_waypoint_stats[-10:]: ', distance_to_waypoint_stats[-10:])
  #future_time_stats = sorted(future_time_stats)
  #print('future_time_stats: ', future_time_stats)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return



def generate_3d_grounding_qa_dataset_nq7(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints, double_cavs, with_context, context_list, output_file, context_list_from_gt):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  NQ7: Prediction
  Q: Where might the notable objects move in the future given my planned future trajectory [(x0, y0), (x1, y1), ...]? 
  A: There is a car at (x0, y0) [moving forward|turning left|turning right]. The predicted future trajecoty is [(x0, y0), ...]
     There is a pedestrian at (x0, y0) [moving forward|turning left|turning right]. The predicted future trajecoty is [(x0, y0), ...]


  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_nq7'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)
  exp_name += 'w' + str(num_future_waypoints)
  if double_cavs:
    exp_name += 'd'
  if with_context:
    exp_name += 'c'  

  # what is this?
  # cav_1's answer exclude cav_1 and include cav_ego
  #exp_name += 'new'  


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []

  if with_context and context_list is not None:
    # generating temp qa for graph inference
    list_data_dict_save_file = output_file
    list_data_dict_save_file_p1 = list_data_dict_save_file[:-5] + '_p1.json'
    list_data_dict_save_file_p2 = list_data_dict_save_file[:-5] + '_p2.json'
    list_data_dict_save_file_p3 = list_data_dict_save_file[:-5] + '_p3.json'
    list_data_dict_save_file_p4 = list_data_dict_save_file[:-5] + '_p4.json'
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)
  else:
    # generating training and validation qa
    output_save_path = llm_data_path
    list_data_dict_save_file = os.path.join(output_save_path, exp_name + '.json')
    list_data_dict_save_file_p1 = os.path.join(output_save_path, exp_name + '_p1.json')
    list_data_dict_save_file_p2 = os.path.join(output_save_path, exp_name + '_p2.json')
    list_data_dict_save_file_p3 = os.path.join(output_save_path, exp_name + '_p3.json')
    list_data_dict_save_file_p4 = os.path.join(output_save_path, exp_name + '_p4.json')
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)


  data_sample_id = 0

  motion_names = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(4)
  qa_sub_type_no_notable_object_counter = 0
  qa_sub_type_notable_object_counter = 0
  future_time_stats = []
  distance_to_waypoint_stats = []

  another_cav_is_notable_object_counter = 0

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    # MY_DEBUG
    #if scenario_index != 4:
    #  continue  

    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  

      # MY_DEBUG
      #if global_timestamp_index != 1962:
      #  continue  

      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, gt_object_ids, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)
      #print('gt_object_ids: ', gt_object_ids)
      #assert False

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)

      # notable gts in ego coordinate system
      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      asker_notable_gts_future_trajectory_dict = dict()
      asker_initial_location_dict = dict()
      for asker_cav_id in cav_ids:
        # get both cav's current location in cav_ego's current coordinate
        lidar_pose_in_ego_initial_frame = x1_to_x2(initial_lidar_pose[asker_cav_id], initial_lidar_pose['ego'])
        asker_initial_location = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
        asker_initial_location_dict[asker_cav_id] = asker_initial_location.copy()
        #print('asker_cav_id: ', asker_cav_id)
        #print('asker_initial_location: ', asker_initial_location)

        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, gt_object_ids, max_num_answer_objects, asker_initial_location)
        #print('notable_gts: ', notable_gts)
        # (N, 7+1+1+1), box parameters, gt_object_id, future time step, dist to cav
        #assert False

        notable_gts_future_trajectory = get_notable_gts_future_trajectory(notable_gts, npy_save_path, global_timestamp_index, num_future_frames, cav_ids)
        #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
        #assert False

        asker_notable_gts_dict[asker_cav_id] = notable_gts
        asker_notable_gts_future_trajectory_dict[asker_cav_id] = notable_gts_future_trajectory

        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        # Old Reason
        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict



      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id
      for asker_cav_id in cav_ids:

        # if we are generating temp qa for graph inference
        if with_context and context_list is not None:
          context_sample_list = []
          for context in context_list:
            context_sample = context[data_sample_id]
            assert(context_sample['id'] == data_sample_id)
            assert(context_sample['global_timestamp_index'] == global_timestamp_index)
            assert(context_sample['asker_cav_id'] == asker_cav_id)
            context_sample_list.append(context_sample)
        else:
          context_sample_list = None

        qa_sample_data_dict, qa_sub_type, future_time, distance_to_waypoint, another_cav_is_notable_object = generate_3d_grounding_qa_sample_nq7(
            asker_cav_id, initial_lidar_pose,
            asker_notable_gts_dict[asker_cav_id],
            asker_notable_gts_future_trajectory_dict[asker_cav_id],
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],    
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
            num_future_waypoints, with_context, asker_initial_location_dict[asker_cav_id],
            cav_ids, double_cavs_future_trajectory_in_ego_current, asker_initial_location_dict,
            context_sample_list, context_list_from_gt)
        qa_sub_type = np.array(qa_sub_type)
        #assert False
        list_data_dict.append(qa_sample_data_dict)

        if another_cav_is_notable_object >= 0:
          another_cav_is_notable_object_counter += 1

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        if qa_sub_type[0] == -1:
          qa_sub_type_no_notable_object_counter += 1
        else:  
          qa_sub_type_notable_object_counter += len(qa_sub_type)  
          #print('qa_sub_type: ', qa_sub_type)
          for idx in range(len(qa_sub_type)):
             qa_sub_type_counter[int(qa_sub_type[idx])] += 1

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  print('qa_sub_type_notable_object_counter: ', qa_sub_type_notable_object_counter)
  print('qa_sub_type_no_notable_object_counter: ', qa_sub_type_no_notable_object_counter)
  #distance_to_waypoint_stats = sorted(distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats: ', distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats[:10]: ', distance_to_waypoint_stats[:10])
  #print('distance_to_waypoint_stats[-10:]: ', distance_to_waypoint_stats[-10:])
  #future_time_stats = sorted(future_time_stats)
  #print('future_time_stats: ', future_time_stats)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)

  print('another_cav_is_notable_object_counter: ', another_cav_is_notable_object_counter)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return




def generate_3d_grounding_qa_dataset_nq6(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints, double_cavs, with_context, context_list, output_file, context_list_from_gt):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  NQ6: Prediction by Planning
  Q: I am CAV_EGO at (x0, y0). Where might other CAVs move in the future given their planned future trajectories? 
     Context: CAV_EGO is at (x0, y0). Its planned future trajectory is [(x0, y0), (x1, y1) ...].
              CAV_1 is at (x0, y0). Its planned future trajectory is [(x0, y0), (x1, y1) ...].
  A: CAV_1 is at (x0, y0) moving forward. Its planned future trajectory is [(x0, y0), (x1, y1) ...].


  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_nq6'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)
  exp_name += 'w' + str(num_future_waypoints)
  if double_cavs:
    exp_name += 'd'
  if with_context:
    exp_name += 'c'  

  # what is this?
  # cav_1's answer exclude cav_1 and include cav_ego
  #exp_name += 'new'  


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []

  if with_context and context_list is not None:
    # generating temp qa for graph inference
    list_data_dict_save_file = output_file
    list_data_dict_save_file_p1 = list_data_dict_save_file[:-5] + '_p1.json'
    list_data_dict_save_file_p2 = list_data_dict_save_file[:-5] + '_p2.json'
    list_data_dict_save_file_p3 = list_data_dict_save_file[:-5] + '_p3.json'
    list_data_dict_save_file_p4 = list_data_dict_save_file[:-5] + '_p4.json'
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)
  else:
    # generating training and validation qa
    output_save_path = llm_data_path
    list_data_dict_save_file = os.path.join(output_save_path, exp_name + '.json')
    list_data_dict_save_file_p1 = os.path.join(output_save_path, exp_name + '_p1.json')
    list_data_dict_save_file_p2 = os.path.join(output_save_path, exp_name + '_p2.json')
    list_data_dict_save_file_p3 = os.path.join(output_save_path, exp_name + '_p3.json')
    list_data_dict_save_file_p4 = os.path.join(output_save_path, exp_name + '_p4.json')
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)


  data_sample_id = 0

  motion_names = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(num_motions)
  future_time_stats = []
  distance_to_waypoint_stats = []

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    # MY_DEBUG
    #if scenario_index != 4:
    #  continue  

    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  

      # MY_DEBUG
      #if global_timestamp_index != 1962:
      #  continue  

      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, gt_object_ids, _, _ = load_gt_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)
      #print('gt_object_ids: ', gt_object_ids)
      #assert False

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)

      # notable gts in ego coordinate system
      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      asker_notable_gts_future_trajectory_dict = dict()
      asker_initial_location_dict = dict()
      for asker_cav_id in cav_ids:
        # get both cav's current location in cav_ego's current coordinate
        lidar_pose_in_ego_initial_frame = x1_to_x2(initial_lidar_pose[asker_cav_id], initial_lidar_pose['ego'])
        asker_initial_location = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
        asker_initial_location_dict[asker_cav_id] = asker_initial_location.copy()
        #print('asker_cav_id: ', asker_cav_id)
        #print('asker_initial_location: ', asker_initial_location)

        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, gt_object_ids, max_num_answer_objects, asker_initial_location)
        #print('notable_gts: ', notable_gts)
        # (N, 7+1+1+1), box parameters, gt_object_id, future time step, dist to cav
        #assert False

        notable_gts_future_trajectory = get_notable_gts_future_trajectory(notable_gts, npy_save_path, global_timestamp_index, num_future_frames, cav_ids)
        #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
        #assert False

        asker_notable_gts_dict[asker_cav_id] = notable_gts
        asker_notable_gts_future_trajectory_dict[asker_cav_id] = notable_gts_future_trajectory

        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        # Old Reason
        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict



      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id
      for asker_cav_id in cav_ids:

        # if we are generating temp qa for graph inference
        if with_context and context_list is not None:
          context_sample_list = []
          for context in context_list:
            context_sample = context[data_sample_id]
            assert(context_sample['id'] == data_sample_id)
            assert(context_sample['global_timestamp_index'] == global_timestamp_index)
            assert(context_sample['asker_cav_id'] == asker_cav_id)
            context_sample_list.append(context_sample)
        else:
          context_sample_list = None

        qa_sample_data_dict, qa_sub_type = generate_3d_grounding_qa_sample_nq6(
            asker_cav_id, initial_lidar_pose,
            asker_notable_gts_dict[asker_cav_id],
            asker_notable_gts_future_trajectory_dict[asker_cav_id],
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],    
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
            num_future_waypoints, with_context, asker_initial_location_dict[asker_cav_id],
            cav_ids, double_cavs_future_trajectory_in_ego_current, asker_initial_location_dict,
            context_sample_list, context_list_from_gt)
        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        for t in qa_sub_type:
          qa_sub_type_counter[t] += 1  

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return



def generate_3d_grounding_qa_dataset_nq4(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints, double_cavs, with_context, context_list, output_file, context_list_from_gt):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  NQ4: Notable object identification 
  Q: I am CAV_EGO. Is there anything I need to be aware of if my planned future trajectory is [(x0, y0), (x1, y1), ...]? 
  A: There is a car at (x0, y0).
     There is a pedestrian at (x0, y0).


  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_nq4'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)
  exp_name += 'w' + str(num_future_waypoints)
  if double_cavs:
    exp_name += 'd'
  if with_context:
    exp_name += 'c'  


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')

  list_data_dict = []
  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []

  if with_context and context_list is not None:
    # generating temp qa for graph inference
    list_data_dict_save_file = output_file
    list_data_dict_save_file_p1 = list_data_dict_save_file[:-5] + '_p1.json'
    list_data_dict_save_file_p2 = list_data_dict_save_file[:-5] + '_p2.json'
    list_data_dict_save_file_p3 = list_data_dict_save_file[:-5] + '_p3.json'
    list_data_dict_save_file_p4 = list_data_dict_save_file[:-5] + '_p4.json'
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)
  else:
    # generating training and validation qa
    output_save_path = llm_data_path
    list_data_dict_save_file = os.path.join(output_save_path, exp_name + '.json')
    list_data_dict_save_file_p1 = os.path.join(output_save_path, exp_name + '_p1.json')
    list_data_dict_save_file_p2 = os.path.join(output_save_path, exp_name + '_p2.json')
    list_data_dict_save_file_p3 = os.path.join(output_save_path, exp_name + '_p3.json')
    list_data_dict_save_file_p4 = os.path.join(output_save_path, exp_name + '_p4.json')
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)


  data_sample_id = 0

  motion_names = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(4)
  qa_sub_type_no_notable_object_counter = 0
  qa_sub_type_notable_object_counter = 0
  future_time_stats = []
  distance_to_waypoint_stats = []

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    # MY_DEBUG
    #if scenario_index != 4:
    #  continue  

    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  

      # MY_DEBUG
      #if global_timestamp_index != 1962:
      #  continue  

      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, gt_object_ids, visible_gt_object_ids_dict, invisible_gt_object_ids_dict = load_gt_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)
      #print('gt_object_ids: ', gt_object_ids)
      #print('visible_gt_object_ids_dict: ', visible_gt_object_ids_dict)
      #print('invisible_gt_object_ids_dict: ', invisible_gt_object_ids_dict)
      #assert False

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)

      # notable gts in ego coordinate system
      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      asker_notable_gts_future_trajectory_dict = dict()
      asker_initial_location_dict = dict()
      for asker_cav_id in cav_ids:
        # get both cav's current location in cav_ego's current coordinate
        lidar_pose_in_ego_initial_frame = x1_to_x2(initial_lidar_pose[asker_cav_id], initial_lidar_pose['ego'])
        asker_initial_location = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
        asker_initial_location_dict[asker_cav_id] = asker_initial_location.copy()
        #print('asker_cav_id: ', asker_cav_id)
        #print('asker_initial_location: ', asker_initial_location)

        occluding_gt_ids_in_gt_boxes, occluded_gt_ids_in_gt_boxes = get_occluding_gt_ids_in_gt_boxes(gt_boxes, gt_box_corners, asker_initial_location, max_num_answer_objects)
        #print('occluding_gt_ids_in_gt_boxes: ', occluding_gt_ids_in_gt_boxes) # at most 6
        #print('occluded_gt_ids_in_gt_boxes: ', occluded_gt_ids_in_gt_boxes) # can be a lot
        #assert False

        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, gt_object_ids, max_num_answer_objects, asker_initial_location)
        #print('notable_gts: ', notable_gts)
        # (N, 7+1+1+1), box parameters, gt_object_id, future time step, dist to cav
        #assert False

        notable_gts_future_trajectory = get_notable_gts_future_trajectory(notable_gts, npy_save_path, global_timestamp_index, num_future_frames, cav_ids)
        #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
        #assert False

        asker_notable_gts_dict[asker_cav_id] = notable_gts
        asker_notable_gts_future_trajectory_dict[asker_cav_id] = notable_gts_future_trajectory

        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        # Old Reason
        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict



      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id
      for asker_cav_id in cav_ids:

        # if we are generating temp qa for graph inference
        if with_context and context_list is not None:
          context_sample_list = []
          for context in context_list:
            context_sample = context[data_sample_id]
            assert(context_sample['id'] == data_sample_id)
            assert(context_sample['global_timestamp_index'] == global_timestamp_index)
            assert(context_sample['asker_cav_id'] == asker_cav_id)
            context_sample_list.append(context_sample)
        else:
          context_sample_list = None

        qa_sample_data_dict, qa_sub_type, future_time, distance_to_waypoint = generate_3d_grounding_qa_sample_nq4(
            asker_cav_id, initial_lidar_pose,
            asker_notable_gts_dict[asker_cav_id],
            asker_notable_gts_future_trajectory_dict[asker_cav_id],
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],    
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
            num_future_waypoints, with_context,
            visible_gt_object_ids_dict, invisible_gt_object_ids_dict,
            occluding_gt_ids_in_gt_boxes, occluded_gt_ids_in_gt_boxes, gt_boxes, asker_initial_location_dict[asker_cav_id],
            context_sample_list, context_list_from_gt,
            det_box_scores_dict[asker_cav_id],
            asker_initial_location_dict)
        qa_sub_type = np.array(qa_sub_type)
        #assert False
        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        if qa_sub_type[0] == -1:
          qa_sub_type_no_notable_object_counter += 1
        else:  
          qa_sub_type_notable_object_counter += len(qa_sub_type)  
          #print('qa_sub_type: ', qa_sub_type)
          for idx in range(len(qa_sub_type)):
             qa_sub_type_counter[int(qa_sub_type[idx])] += 1

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  print('qa_sub_type_notable_object_counter: ', qa_sub_type_notable_object_counter)
  print('qa_sub_type_no_notable_object_counter: ', qa_sub_type_no_notable_object_counter)
  #distance_to_waypoint_stats = sorted(distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats: ', distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats[:10]: ', distance_to_waypoint_stats[:10])
  #print('distance_to_waypoint_stats[-10:]: ', distance_to_waypoint_stats[-10:])
  #future_time_stats = sorted(future_time_stats)
  #print('future_time_stats: ', future_time_stats)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return



def generate_3d_grounding_qa_dataset_nq3(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints, double_cavs, with_context, context_list, output_file, context_list_from_gt):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  NQ3: Invisible Notable object identification 
  Q: I am CAV_EGO. What are the notable objects invisible to me near my planned future trajectory [(x0, y0), (x1, y1), ...]? 
  A: There is a car at (x0, y0) invisible to you.
     There is a pedestrian at (x0, y0) invisible to you.

  if with_context, and context_list is None, use gt context
  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_nq3'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)
  exp_name += 'w' + str(num_future_waypoints)
  if double_cavs:
    exp_name += 'd'
  if with_context:
    exp_name += 'c'  


  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')

  list_data_dict = []
  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []

  if with_context and context_list is not None:
    # generating temp qa for graph inference
    list_data_dict_save_file = output_file
    list_data_dict_save_file_p1 = list_data_dict_save_file[:-5] + '_p1.json'
    list_data_dict_save_file_p2 = list_data_dict_save_file[:-5] + '_p2.json'
    list_data_dict_save_file_p3 = list_data_dict_save_file[:-5] + '_p3.json'
    list_data_dict_save_file_p4 = list_data_dict_save_file[:-5] + '_p4.json'
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)
  else:
    # generating training and validation qa
    output_save_path = llm_data_path
    list_data_dict_save_file = os.path.join(output_save_path, exp_name + '.json')
    list_data_dict_save_file_p1 = os.path.join(output_save_path, exp_name + '_p1.json')
    list_data_dict_save_file_p2 = os.path.join(output_save_path, exp_name + '_p2.json')
    list_data_dict_save_file_p3 = os.path.join(output_save_path, exp_name + '_p3.json')
    list_data_dict_save_file_p4 = os.path.join(output_save_path, exp_name + '_p4.json')
    print('list_data_dict_save_file: ', list_data_dict_save_file)
    #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json
    print('list_data_dict_save_file_p1: ', list_data_dict_save_file_p1)

  data_sample_id = 0

  motion_names = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(4)
  qa_sub_type_no_notable_object_counter = 0
  qa_sub_type_notable_object_counter = 0
  future_time_stats = []
  distance_to_waypoint_stats = []
  total_num_invisible_notable_gts_stats_dict = {cav_id: 0 for cav_id in cav_ids}
  total_num_notable_gts_stats_dict = {cav_id: 0 for cav_id in cav_ids}

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    # MY_DEBUG
    #if scenario_index != 4:
    #  continue  

    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  

      # MY_DEBUG
      #if global_timestamp_index != 1962:
      #  continue  

      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, gt_object_ids, visible_gt_object_ids_dict, invisible_gt_object_ids_dict = load_gt_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)
      #print('gt_object_ids: ', gt_object_ids)
      #print('visible_gt_object_ids_dict: ', visible_gt_object_ids_dict)
      #print('invisible_gt_object_ids_dict: ', invisible_gt_object_ids_dict)
      #assert False

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)

      # notable gts in ego coordinate system
      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      asker_notable_gts_future_trajectory_dict = dict()
      asker_initial_location_dict = dict()
      for asker_cav_id in cav_ids:
        # get both cav's current location in cav_ego's current coordinate
        lidar_pose_in_ego_initial_frame = x1_to_x2(initial_lidar_pose[asker_cav_id], initial_lidar_pose['ego'])
        asker_initial_location = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
        asker_initial_location_dict[asker_cav_id] = asker_initial_location.copy()
        #print('asker_cav_id: ', asker_cav_id)
        #print('asker_initial_location: ', asker_initial_location)

        occluding_gt_ids_in_gt_boxes, occluded_gt_ids_in_gt_boxes = get_occluding_gt_ids_in_gt_boxes(gt_boxes, gt_box_corners, asker_initial_location, max_num_answer_objects)
        #print('occluding_gt_ids_in_gt_boxes: ', occluding_gt_ids_in_gt_boxes) # at most 6
        #print('occluded_gt_ids_in_gt_boxes: ', occluded_gt_ids_in_gt_boxes) # can be a lot
        #assert False

        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, gt_object_ids, max_num_answer_objects, asker_initial_location)
        #print('notable_gts: ', notable_gts)
        # (N, 7+1+1+1), box parameters, gt_object_id, future time step, dist to cav
        #assert False

        notable_gts_future_trajectory = get_notable_gts_future_trajectory(notable_gts, npy_save_path, global_timestamp_index, num_future_frames, cav_ids)
        #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
        #assert False

        asker_notable_gts_dict[asker_cav_id] = notable_gts
        asker_notable_gts_future_trajectory_dict[asker_cav_id] = notable_gts_future_trajectory

        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        # Old Reason
        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict



      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id
      for asker_cav_id in cav_ids:

        # if we are generating temp qa for graph inference
        if with_context and context_list is not None:
          context_sample_list = []
          for context in context_list:
            context_sample = context[data_sample_id]
            assert(context_sample['id'] == data_sample_id)
            assert(context_sample['global_timestamp_index'] == global_timestamp_index)
            assert(context_sample['asker_cav_id'] == asker_cav_id)
            context_sample_list.append(context_sample)
        else:
          context_sample_list = None

        qa_sample_data_dict, qa_sub_type, future_time, distance_to_waypoint, num_invisible_notable_gts = generate_3d_grounding_qa_sample_nq3(
            asker_cav_id, initial_lidar_pose,
            asker_notable_gts_dict[asker_cav_id],
            asker_notable_gts_future_trajectory_dict[asker_cav_id],
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],    
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
            num_future_waypoints, with_context,
            visible_gt_object_ids_dict, invisible_gt_object_ids_dict, 
            occluding_gt_ids_in_gt_boxes, occluded_gt_ids_in_gt_boxes, gt_boxes, asker_initial_location_dict[asker_cav_id],
            context_sample_list, context_list_from_gt,
            asker_initial_location_dict)
        qa_sub_type = np.array(qa_sub_type)
        total_num_invisible_notable_gts_stats_dict[asker_cav_id] += num_invisible_notable_gts
        total_num_notable_gts_stats_dict[asker_cav_id] += asker_notable_gts_dict[asker_cav_id].shape[0] if asker_notable_gts_dict[asker_cav_id] is not None else 0
        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        if qa_sub_type[0] == -1:
          qa_sub_type_no_notable_object_counter += 1
        else:  
          qa_sub_type_notable_object_counter += len(qa_sub_type)  
          #print('qa_sub_type: ', qa_sub_type)
          for idx in range(len(qa_sub_type)):
             qa_sub_type_counter[int(qa_sub_type[idx])] += 1

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  print('qa_sub_type_notable_object_counter: ', qa_sub_type_notable_object_counter)
  print('qa_sub_type_no_notable_object_counter: ', qa_sub_type_no_notable_object_counter)
  #distance_to_waypoint_stats = sorted(distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats: ', distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats[:10]: ', distance_to_waypoint_stats[:10])
  #print('distance_to_waypoint_stats[-10:]: ', distance_to_waypoint_stats[-10:])
  #future_time_stats = sorted(future_time_stats)
  #print('future_time_stats: ', future_time_stats)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)
  print('total_num_notable_gts_stats_dict: ', total_num_notable_gts_stats_dict)
  print('total_num_invisible_notable_gts_stats_dict: ', total_num_invisible_notable_gts_stats_dict)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return



def generate_3d_grounding_qa_dataset_nq1(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints, double_cavs, with_context):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  NQ1: Visible Notable object identification 
  Q: I am CAV_EGO. What are the notable objects visible to me near my planned future trajectory [(x0, y0), (x1, y1), ...]? 
  A: There is a car at (x0, y0) visible to you.
     There is a pedestrian at (x0, y0) visible to you.


  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_nq1'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)
  exp_name += 'w' + str(num_future_waypoints)
  if double_cavs:
    exp_name += 'd'
  if with_context:
    exp_name += 'c'  

  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, exp_name + '.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json

  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []
  list_data_dict_save_file_p1 = os.path.join(llm_data_path, exp_name + '_p1.json')
  list_data_dict_save_file_p2 = os.path.join(llm_data_path, exp_name + '_p2.json')
  list_data_dict_save_file_p3 = os.path.join(llm_data_path, exp_name + '_p3.json')
  list_data_dict_save_file_p4 = os.path.join(llm_data_path, exp_name + '_p4.json')


  data_sample_id = 0

  motion_names = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(4)
  qa_sub_type_no_notable_object_counter = 0
  qa_sub_type_notable_object_counter = 0
  future_time_stats = []
  distance_to_waypoint_stats = []
  total_num_visible_notable_gts_stats_dict = {cav_id: 0 for cav_id in cav_ids}
  total_num_notable_gts_stats_dict = {cav_id: 0 for cav_id in cav_ids}

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    # MY_DEBUG
    #if scenario_index != 4:
    #  continue  

    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  

      # MY_DEBUG
      #if global_timestamp_index != 1962:
      #  continue  

      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, gt_object_ids, visible_gt_object_ids_dict, invisible_gt_object_ids_dict = load_gt_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)
      #print('gt_object_ids: ', gt_object_ids)
      #print('visible_gt_object_ids_dict: ', visible_gt_object_ids_dict)
      #print('invisible_gt_object_ids_dict: ', invisible_gt_object_ids_dict)
      #assert False

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)

      # notable gts in ego coordinate system
      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      asker_notable_gts_future_trajectory_dict = dict()
      asker_initial_location_dict = dict()
      for asker_cav_id in cav_ids:
        # get both cav's current location in cav_ego's current coordinate
        lidar_pose_in_ego_initial_frame = x1_to_x2(initial_lidar_pose[asker_cav_id], initial_lidar_pose['ego'])
        asker_initial_location = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
        asker_initial_location_dict[asker_cav_id] = asker_initial_location.copy()
        #print('asker_cav_id: ', asker_cav_id)
        #print('asker_initial_location: ', asker_initial_location)

        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, gt_object_ids, max_num_answer_objects, asker_initial_location)
        #print('notable_gts: ', notable_gts)
        # (N, 7+1+1+1), box parameters, gt_object_id, future time step, dist to cav
        #assert False

        notable_gts_future_trajectory = get_notable_gts_future_trajectory(notable_gts, npy_save_path, global_timestamp_index, num_future_frames, cav_ids)
        #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
        #assert False

        asker_notable_gts_dict[asker_cav_id] = notable_gts
        asker_notable_gts_future_trajectory_dict[asker_cav_id] = notable_gts_future_trajectory

        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        # Old Reason
        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict



      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id
      for asker_cav_id in cav_ids:
        qa_sample_data_dict, qa_sub_type, future_time, distance_to_waypoint, num_visible_notable_gts = generate_3d_grounding_qa_sample_nq1(
            asker_cav_id, initial_lidar_pose,
            asker_notable_gts_dict[asker_cav_id],
            asker_notable_gts_future_trajectory_dict[asker_cav_id],
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],    
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
            num_future_waypoints, with_context,
            visible_gt_object_ids_dict, invisible_gt_object_ids_dict,
            asker_initial_location_dict[asker_cav_id],
            asker_initial_location_dict)
        qa_sub_type = np.array(qa_sub_type)
        total_num_visible_notable_gts_stats_dict[asker_cav_id] += num_visible_notable_gts
        total_num_notable_gts_stats_dict[asker_cav_id] += asker_notable_gts_dict[asker_cav_id].shape[0] if asker_notable_gts_dict[asker_cav_id] is not None else 0
        #assert False
        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        if qa_sub_type[0] == -1:
          qa_sub_type_no_notable_object_counter += 1
        else:  
          qa_sub_type_notable_object_counter += len(qa_sub_type)  
          #print('qa_sub_type: ', qa_sub_type)
          for idx in range(len(qa_sub_type)):
             qa_sub_type_counter[int(qa_sub_type[idx])] += 1

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  print('qa_sub_type_notable_object_counter: ', qa_sub_type_notable_object_counter)
  print('qa_sub_type_no_notable_object_counter: ', qa_sub_type_no_notable_object_counter)
  #distance_to_waypoint_stats = sorted(distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats: ', distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats[:10]: ', distance_to_waypoint_stats[:10])
  #print('distance_to_waypoint_stats[-10:]: ', distance_to_waypoint_stats[-10:])
  #future_time_stats = sorted(future_time_stats)
  #print('future_time_stats: ', future_time_stats)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)
  print('total_num_notable_gts_stats_dict: ', total_num_notable_gts_stats_dict)
  print('total_num_visible_notable_gts_stats_dict: ', total_num_visible_notable_gts_stats_dict)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return


def generate_3d_grounding_qa_dataset_nq2(len_record, npy_save_path, cav_ids, downsample_negatives, simplified, max_num_answer_objects, num_future_waypoints, double_cavs, with_context):
  '''
  # Note that in my dmstrack coordinate system
  # [x, z] is the ground plane spatial location
  # https://github.com/eddyhkchiu/DMSTrack/issues/1

  NQ2: Occlusing object identification 
  Q: I am CAV_EGO. What objects might obstruct my view? 
  A: There is a car at (x0, y0) obstructing your view.
     There is a pedestrian at (x0, y0) obstructing your view.


  '''
  exp_name = 'v2v4real_3d_grounding_qa_dataset_nq2'
  if downsample_negatives:
    exp_name += 'b'
  if simplified:
    exp_name += 's'  
  exp_name += 'm' + str(max_num_answer_objects)
  exp_name += 'w' + str(num_future_waypoints)
  if double_cavs:
    exp_name += 'd'
  if with_context:
    exp_name += 'c'  

  print('len_record: ', len_record)
  #print('npy_save_path: ', npy_save_path)
  #print('cav_id_set: ', cav_id_set)

  llm_data_path = os.path.join(npy_save_path, 'co_llm')
  list_data_dict = []
  list_data_dict_save_file = os.path.join(llm_data_path, exp_name + '.json')
  print('list_data_dict_save_file: ', list_data_dict_save_file)
  #  /home/hsukuangc/dataset/V2V4Real_official_models/no_fusion_keep_all/npy/co_llm/v2v4real_3d_grounding_qa_dataset_v6.json

  # 4 more splitted files for batch_size 1 inference, based on global_timestamp_index
  # val total 1993 frames, split by 500
  list_data_dict_p1 = []
  list_data_dict_p2 = []
  list_data_dict_p3 = []
  list_data_dict_p4 = []
  list_data_dict_save_file_p1 = os.path.join(llm_data_path, exp_name + '_p1.json')
  list_data_dict_save_file_p2 = os.path.join(llm_data_path, exp_name + '_p2.json')
  list_data_dict_save_file_p3 = os.path.join(llm_data_path, exp_name + '_p3.json')
  list_data_dict_save_file_p4 = os.path.join(llm_data_path, exp_name + '_p4.json')


  data_sample_id = 0

  motion_names = ['moving forward', 'turning left', 'turning right', 'staying at the same location']
  num_motions = len(motion_names)
  qa_sub_type_counter = np.zeros(4)
  qa_sub_type_no_notable_object_counter = 0
  qa_sub_type_notable_object_counter = 0
  future_time_stats = []
  distance_to_waypoint_stats = []
  total_num_occluding_gts_stats_dict = {cav_id: 0 for cav_id in cav_ids}

  
  time_horizon = 3
  frame_rate = 10
  num_future_frames = frame_rate * time_horizon
  max_num_notable_gts_stats = 0


  for scenario_index in range(len(len_record)):
    # MY_DEBUG
    #if scenario_index != 4:
    #  continue  

    if scenario_index == 0:
      start_global_timestamp_index = 0
    else:
      start_global_timestamp_index = len_record[scenario_index - 1]
    end_global_timestamp_index = len_record[scenario_index] - 1

    # for each frame (each data sample) that has at least num_future_frames in the sequence
    for global_timestamp_index in range(start_global_timestamp_index, end_global_timestamp_index + 1 - num_future_frames):
      print('global_timestamp_index: ', global_timestamp_index)  

      # MY_DEBUG
      #if global_timestamp_index != 1962:
      #  continue  

      local_timestamp_index = global_timestamp_index - start_global_timestamp_index

      gt_boxes, gt_box_corners, gt_object_ids, visible_gt_object_ids_dict, invisible_gt_object_ids_dict = load_gt_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('gt_boxes: ', gt_boxes)
      #print('gt_boxes.shape: ', gt_boxes.shape)
      # (2, 7)
      #print('gt_object_ids: ', gt_object_ids)
      #print('visible_gt_object_ids_dict: ', visible_gt_object_ids_dict)
      #print('invisible_gt_object_ids_dict: ', invisible_gt_object_ids_dict)
      #assert False

      det_box_scores_dict, det_box_corners_dict = load_det_boxes_3d(npy_save_path, global_timestamp_index, cav_ids)
      #print('det_box_scores_dict: ', det_box_scores_dict)

      double_cavs_future_trajectory_in_ego_current, double_cavs_future_trajectory_in_self_current, initial_lidar_pose = get_double_cavs_future_trajectory(npy_save_path, global_timestamp_index, start_global_timestamp_index, end_global_timestamp_index, num_future_frames, cav_ids)
      # (num_future_frames, 2)

      # notable gts in ego coordinate system
      asker_merged_reason_point_in_det_boxes_dict = dict()
      asker_notable_gts_dict = dict()
      asker_notable_gts_future_trajectory_dict = dict()
      asker_initial_location_dict = dict()
      for asker_cav_id in cav_ids:
        # get both cav's current location in cav_ego's current coordinate
        lidar_pose_in_ego_initial_frame = x1_to_x2(initial_lidar_pose[asker_cav_id], initial_lidar_pose['ego'])
        asker_initial_location = np.array([lidar_pose_in_ego_initial_frame[0, 3], lidar_pose_in_ego_initial_frame[1, 3]])
        asker_initial_location_dict[asker_cav_id] = asker_initial_location.copy()
        #print('asker_cav_id: ', asker_cav_id)
        #print('asker_initial_location: ', asker_initial_location)

        occluding_gt_ids_in_gt_boxes, occluded_gt_ids_in_gt_boxes = get_occluding_gt_ids_in_gt_boxes(gt_boxes, gt_box_corners, asker_initial_location, max_num_answer_objects)
        #print('occluding_gt_ids_in_gt_boxes: ', occluding_gt_ids_in_gt_boxes) # at most 6
        #print('occluded_gt_ids_in_gt_boxes: ', occluded_gt_ids_in_gt_boxes) # can be a lot
        #assert False

        notable_gts = get_notable_gts_near_cav_future_trajectory(double_cavs_future_trajectory_in_ego_current[asker_cav_id], gt_boxes, gt_box_corners, gt_object_ids, max_num_answer_objects, asker_initial_location)
        #print('notable_gts: ', notable_gts)
        # (N, 7+1+1+1), box parameters, gt_object_id, future time step, dist to cav
        #assert False

        notable_gts_future_trajectory = get_notable_gts_future_trajectory(notable_gts, npy_save_path, global_timestamp_index, num_future_frames, cav_ids)
        #print('notable_gts_future_trajectory: ', notable_gts_future_trajectory)
        #assert False

        asker_notable_gts_dict[asker_cav_id] = notable_gts
        asker_notable_gts_future_trajectory_dict[asker_cav_id] = notable_gts_future_trajectory

        if notable_gts is not None and notable_gts.shape[0] > max_num_notable_gts_stats:
          max_num_notable_gts_stats = notable_gts.shape[0]
          #print('notable_gts: ', notable_gts)

        # Old Reason
        # check whether we have positive samples
        if notable_gts is None:
          # if no notable_gts, check any cav detects something at a future waypoint 
          # may be just show 3 way points in question
          # the first one is 9th waypoint for 1 second in the future
          merged_reason_points = double_cavs_future_trajectory_in_ego_current[asker_cav_id][9:10, :]
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        else:
          # check any cav detects the notable_gts
          # for generating the reasons
          merged_reason_points = np.stack([
            notable_gts[:, 3],
            notable_gts[:, 5]
          ], axis=1)
          #print('merged_reason_points: ', merged_reason_points)
          # (1, 2)
          #assert False
        merged_reason_point_in_det_boxes_dict = dict()
        for cav_id in cav_ids:
          merged_reason_point_in_det_boxes_dict[cav_id] = check_point_in_rotated_rectangles(
            merged_reason_points, det_box_corners_dict[cav_id], det_box_scores_dict[cav_id])
        #if notable_gts is not None and notable_gts.shape[0] > 1:  
        #  print('notable_gts: ', notable_gts)  
        #  print('merged_reason_point_in_det_boxes_dict: ', merged_reason_point_in_det_boxes_dict)
        #  # {'ego': array([-1, -1]), '1': array([-1, -1])}
        #  assert False
        asker_merged_reason_point_in_det_boxes_dict[asker_cav_id] = merged_reason_point_in_det_boxes_dict


      # MY_DEBUG
      # temp go to next frame without generating QA sample
      #continue

      # at one frame we generate 2 qa pairs, each one is from one asker_cav_id
      for asker_cav_id in cav_ids:
        qa_sample_data_dict, qa_sub_type, future_time, distance_to_waypoint, num_occluding_gts = generate_3d_grounding_qa_sample_nq2(
            asker_cav_id, initial_lidar_pose,
            asker_notable_gts_dict[asker_cav_id],
            asker_notable_gts_future_trajectory_dict[asker_cav_id],
            double_cavs_future_trajectory_in_ego_current[asker_cav_id], double_cavs_future_trajectory_in_self_current[asker_cav_id],
            asker_merged_reason_point_in_det_boxes_dict[asker_cav_id],    
            data_sample_id, scenario_index, local_timestamp_index, global_timestamp_index, simplified,
            num_future_waypoints, with_context,
            visible_gt_object_ids_dict, invisible_gt_object_ids_dict,
            occluding_gt_ids_in_gt_boxes, occluded_gt_ids_in_gt_boxes, gt_boxes, asker_initial_location_dict[asker_cav_id])
        qa_sub_type = np.array(qa_sub_type)
        total_num_occluding_gts_stats_dict[asker_cav_id] += num_occluding_gts
        #assert False
        list_data_dict.append(qa_sample_data_dict)

        # MY_DEBUG
        #print('qa_sample_data_dict: ', qa_sample_data_dict)
        #with open(list_data_dict_save_file, 'w') as f:
        #  json.dump(list_data_dict, f)
        #if data_sample_id == 382:
        #  assert False

        data_sample_id += 1
        #print('qa_sub_type_counter: ', qa_sub_type_counter)
        #print('qa_sub_type: ', qa_sub_type)
        if qa_sub_type[0] == -1:
          qa_sub_type_no_notable_object_counter += 1
        else:  
          qa_sub_type_notable_object_counter += len(qa_sub_type)  
          #print('qa_sub_type: ', qa_sub_type)
          for idx in range(len(qa_sub_type)):
             qa_sub_type_counter[int(qa_sub_type[idx])] += 1

        # split to 4 
        if global_timestamp_index < 500:
            list_data_dict_p1.append(qa_sample_data_dict)
        elif global_timestamp_index < 1000:
            list_data_dict_p2.append(qa_sample_data_dict)
        elif global_timestamp_index < 1500:
            list_data_dict_p3.append(qa_sample_data_dict)
        else:
            list_data_dict_p4.append(qa_sample_data_dict)

  print('Total number of data samples: ', data_sample_id)
  print('qa_sub_type_counter: ', qa_sub_type_counter)
  print('qa_sub_type_notable_object_counter: ', qa_sub_type_notable_object_counter)
  print('qa_sub_type_no_notable_object_counter: ', qa_sub_type_no_notable_object_counter)
  #distance_to_waypoint_stats = sorted(distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats: ', distance_to_waypoint_stats)
  #print('distance_to_waypoint_stats[:10]: ', distance_to_waypoint_stats[:10])
  #print('distance_to_waypoint_stats[-10:]: ', distance_to_waypoint_stats[-10:])
  #future_time_stats = sorted(future_time_stats)
  #print('future_time_stats: ', future_time_stats)
  print('max_num_notable_gts_stats: ', max_num_notable_gts_stats)
  print('total_num_occluding_gts_stats_dict: ', total_num_occluding_gts_stats_dict)

  with open(list_data_dict_save_file, 'w') as f:  
    json.dump(list_data_dict, f)

  # split to 4
  with open(list_data_dict_save_file_p1, 'w') as f:  
    json.dump(list_data_dict_p1, f)
  with open(list_data_dict_save_file_p2, 'w') as f:  
    json.dump(list_data_dict_p2, f)
  with open(list_data_dict_save_file_p3, 'w') as f:  
    json.dump(list_data_dict_p3, f)
  with open(list_data_dict_save_file_p4, 'w') as f:  
    json.dump(list_data_dict_p4, f)

  return


def get_occluding_gt_ids_in_gt_boxes(gt_boxes, gt_box_corners, asker_initial_location, max_num_answer_objects):
  #print('gt_boxes: ', gt_boxes)
  # (N, 7), [h, w, l, x, y, z, rot_y]
  #print('gt_box_corners: ', gt_box_corners)
  # (N, 8, 3)
  #print('asker_initial_location: ', asker_initial_location)
  # [x0, z0]
  #print('max_num_answer_objects: ', max_num_answer_objects)
  # 6

  occluding_gt_ids_in_gt_boxes = []  
  occluded_gt_ids_in_gt_boxes = []

  # sort by dist to asker cav
  gt_ids = np.arange(gt_boxes.shape[0])
  #print('gt_ids: ', gt_ids)
  dists = np.sqrt((gt_boxes[:, 3] - asker_initial_location[0])**2 + (gt_boxes[:, 5] - asker_initial_location[1])**2)
  #print('dists: ', dists)
  gt_ids_dists = np.stack([
    gt_ids,
    dists
  ], axis=1)
  #print('gt_ids_dists: ', gt_ids_dists)
  # (N, 2)

  # sort the row by its second element (dist)
  gt_ids_dists = gt_ids_dists[gt_ids_dists[:, 1].argsort()]
  #print('gt_ids_dists: ', gt_ids_dists)

  # for each gt in the sorted list
  # if not mark as occluded, add it to occluding list
  #   create sector region, 
  #   mark all other gt in sector region as occluded
  for gt_id in gt_ids_dists[:, 0]:
    gt_id = int(gt_id)  
    #print('gt_id: ', gt_id)

    # check if this gt is asker cav itself
    #print('dists[gt_id]: ', dists[gt_id])
    if dists[gt_id] < 3:
      continue

    if gt_id not in occluded_gt_ids_in_gt_boxes:
      occluding_gt_ids_in_gt_boxes.append(gt_id)  
      if len(occluding_gt_ids_in_gt_boxes) == max_num_answer_objects:
        break

      sample_box_scores = np.expand_dims(gt_boxes[gt_id], axis=0)
      sample_box_corners = np.expand_dims(gt_box_corners[gt_id], axis=0)
      sector_regions = get_sector_regions_behind_det_box(asker_initial_location, sample_box_scores, sample_box_corners)
      #print('sector_regions: ', sector_regions)
      _, _, gt_ids_in_gt_boxes_in_each_region = get_boxes_in_sector_regions(sector_regions, gt_boxes)
      #print('gt_ids_in_gt_boxes_in_each_region: ', gt_ids_in_gt_boxes_in_each_region)
      # [[]]
      gt_ids_in_gt_boxes_in_this_region = gt_ids_in_gt_boxes_in_each_region[0]
      #print('gt_ids_in_gt_boxes_in_this_region: ', gt_ids_in_gt_boxes_in_this_region)
      for gt_id_occluded in gt_ids_in_gt_boxes_in_this_region:
         occluded_gt_ids_in_gt_boxes.append(gt_id_occluded)

  #print('occluding_gt_ids_in_gt_boxes: ', occluding_gt_ids_in_gt_boxes)
  #print('occluded_gt_ids_in_gt_boxes: ', occluded_gt_ids_in_gt_boxes)
  #if len(occluded_gt_ids_in_gt_boxes) > 0:
  #  assert False
  return occluding_gt_ids_in_gt_boxes, occluded_gt_ids_in_gt_boxes  


if __name__ == '__main__':
    main()
