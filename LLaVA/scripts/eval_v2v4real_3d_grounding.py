import os
import argparse
import json
import numpy as np
import cv2

from AB3DMOT.AB3DMOT_libs.box import Box3D
from AB3DMOT_libs.dist_metrics import iou
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_io import mkdir_if_missing

from V2V4Real.opencood.utils.transformation_utils import x1_to_x2
from V2V4Real.opencood.utils.box_utils import boxes_to_corners_3d, corner_to_center, project_points_by_matrix_torch

import matplotlib.pyplot as plt

import pickle
#import tqdm


def parse_multiple_box_parameters(text, simplified, max_num_answer_objects, skip_dummy_boxes=False):
    '''
    Input:
        text: str
        if not simplified:
          "some words [h, w, l, x, y, z, a],  [h, w, l, x, y, z, a], ... some words"
        else:
          "some words [x, z], [x, z], ... some words"
    Output:
        if not simplified:
          box: np shape [max_num_answer_objects, 8] 
            successful parsing: [h, w, l, x, y, z, a] + [1]
            failed parsing or num_boxes < max_num_answer_objects: dummy box [7] + [0]
            dummy_box [1,1,1,0,0,0,0]
        else:    
          box: np shape [max_num_answer_objects, 2] 
            successful parsing: [x, z] + [1]
            failed parsing or num_boxes < max_num_answer_objects: dummy box [2] + [0]
            dummy_box [0,0]
    '''
    all_boxes = []
    start_idx = text.find('[')
    end_idx = text.find(']')
    remaining_text = text[start_idx:]

    for i in range(max_num_answer_objects):
      box = parse_box_parameters(remaining_text, simplified)
      all_boxes.append(box)

      # if dummpy box, not need to do further parsing, fill dummpy box
      if box[-1] == 0:
        remaining_text = ''    
      else:  
        # prepare next parsing
        start_idx = remaining_text.find('[')
        end_idx = remaining_text.find(']')
        remaining_text = remaining_text[end_idx+1:]

    all_boxes = np.stack(all_boxes, axis=0)  

    if not skip_dummy_boxes:
      return all_boxes
    else:
      final_all_boxes = []
      for i in range(all_boxes.shape[0]):
        if all_boxes[i][-1] != 0:
          final_all_boxes.append(all_boxes[i])    
      if len(final_all_boxes) > 0:
        final_all_boxes = np.stack(final_all_boxes)
        return final_all_boxes
      else:
        return None  


def parse_box_parameters(text, simplified):
    '''
    Input:
        text: str
        if not simplified:
          "some words [h, w, l, x, y, z, a] some words"
        else:
          "some words [x, z] some words"
    Output:
        if not simplified:
          box: np shape 8, 
            successful parsing: [h, w, l, x, y, z, a] + [1]
            failed parsing: dummy box [7] + [0]
            dummy_box [1,1,1,0,0,0,0]
        else:    
          box: np shape 2, 
            successful parsing: [x, z] + [1]
            failed parsing: dummy box [2] + [0]
            dummy_box [0,0]
    '''
    # dummy box at origin, will not match with any gt
    # origin is occupied by cav ego, which is not
    # included in ground-truth box set
    if not simplified:
      num_parameters = 7
      dummy_box = np.array([1, 1, 1, 0, 0, 0, 0.0, 0])
    else:  
      num_parameters = 2
      dummy_box = np.array([0.0, 0, 0])


    start_idx = text.find('[')
    end_idx = text.find(']')

    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        #print('parsing fail text: ', text)
        return dummy_box

    box_str = text[start_idx + 1 : end_idx].strip()
    box_str = box_str.split(',')

    if len(box_str) != num_parameters:
        #print('parsing fail text: ', text)
        return dummy_box

    try:
        box_str = [float(s.strip()) for s in box_str]
    except:
        #print('parsing fail text: ', text)
        return dummy_box

    box = np.array(box_str + [1])

    # invalid box size
    if not simplified:
      if box[0] <= 0 or box[1] <= 0 or box[1] <= 0:
        return dummy_box
    
    return box


def eval_multiple_center_dist_2d(gt_boxes, output_boxes, simplified): 
    '''
    Using box center_dist_2d to calculate precision, recall, f1 for valid boxes

    Input:
      gt_boxes: (num_samples, max_num_answer_objects=3, 3)
        3: [x, z, is_valid]
      output_boxes: (num_samples, max_num_answer_objects=3, 3)
        3: [x, z, is_valid]
    '''
    gt_boxes = np.expand_dims(gt_boxes, axis=2)
    num_total_gt_boxes = np.sum(gt_boxes[:, :, :, -1] == 1)
    print('num_total_gt_boxes: ', num_total_gt_boxes)

    output_boxes = np.expand_dims(output_boxes, axis=1)
    num_total_output_boxes = np.sum(output_boxes[:, :, :, -1] == 1)
    print('num_total_output_boxes: ', num_total_output_boxes)

    if not simplified:
      center_dist_2d = np.sqrt((output_boxes[:, :, :, 3] - gt_boxes[:, :, :, 3]) ** 2 + (output_boxes[:, :, :, 5] - gt_boxes[:, :, :, 5]) ** 2) 
    else:
      center_dist_2d = np.sqrt((output_boxes[:, :, :, 0] - gt_boxes[:, :, :, 0]) ** 2 + (output_boxes[:, :, :, 1] - gt_boxes[:, :, :, 1]) ** 2) 

    valid_mask = np.logical_and(gt_boxes[:, :, :, -1], output_boxes[:, :, :, -1])

    center_dist_2d_padded_max = center_dist_2d.copy()
    center_dist_2d_padded_max[np.logical_not(valid_mask)] = 1e5
    dist_errors = []
    for i in range(center_dist_2d_padded_max.shape[0]):
      for output_idx in range(center_dist_2d_padded_max.shape[2]):
        if valid_mask[i, 0, output_idx]:
          dist_error = np.min(center_dist_2d_padded_max[i, :, output_idx]).item()    
          dist_errors.append(dist_error)
          if dist_error > 1e4:
            assert False
    avg_dist_error =  sum(dist_errors) / len(dist_errors)    
    print('avg_dist_error: ', avg_dist_error)

    for threshold in [0.5, 1, 2, 4]:
      print('threshold: ', threshold)
      distance_less_than_threshold = center_dist_2d <= threshold
      distance_less_than_threshold = np.logical_and(distance_less_than_threshold, valid_mask)

      output_close_to_at_least_one_gt = np.any(distance_less_than_threshold, axis=1)
      num_matched_output = np.sum(output_close_to_at_least_one_gt)

      gt_close_to_at_least_one_output = np.any(distance_less_than_threshold, axis=2)
      num_match_gt = np.sum(gt_close_to_at_least_one_output)

      precision = 1.0 * num_matched_output / (num_total_output_boxes)
      recall = 1.0 * num_match_gt / (num_total_gt_boxes)
      f1 = 2 * precision * recall / (precision + recall)
      print('precision: ', precision)
      print('recall: ', recall)
      print('f1: ', f1)

    return


def eval_center_dist_2d(gt_boxes, output_boxes, simplified): 
    num_boxes = gt_boxes.shape[0]

    if not simplified:
      center_dist_2d = np.sqrt((output_boxes[:, 3] - gt_boxes[:, 3]) ** 2 + (output_boxes[:, 5] - gt_boxes[:, 5]) ** 2) 
    else:
      center_dist_2d = np.sqrt((output_boxes[:, 0] - gt_boxes[:, 0]) ** 2 + (output_boxes[:, 1] - gt_boxes[:, 1]) ** 2) 

    for threshold in [0.5, 1, 2, 4]:
      num_matches = np.sum(center_dist_2d < threshold)
      accuracy = 1.0 * num_matches / num_boxes
      print('2D center distance accuracy @ %f meters: %f' % (threshold, accuracy))

    # average 2d distance error
    average_2d_error = np.average(center_dist_2d)
    print('2D center distance error in meters: %f' % average_2d_error)

    return


def eval_iou_3d(gt_boxes, output_boxes):
    # [h, w, l, x, y, z, a]

    num_boxes = gt_boxes.shape[0]

    # ab3dmot code will fail when iou is close to 1
    # add eps to avoid that without affecting iou value too much
    output_boxes += 1e-7

    # Box3D(x, y, z, h, w, l, ry, s)
    gt_boxes = [Box3D(b[3], b[4], b[5], b[0], b[1], b[2], b[6], 1) for b in gt_boxes]
    output_boxes = [Box3D(b[3], b[4], b[5], b[0], b[1], b[2], b[6], 1) for b in output_boxes]


    for iou_type in ['iou_2d', 'iou_3d']:
      iou_nd = [iou(output_boxes[i], gt_boxes[i], iou_type) for i in range(num_boxes)]
      iou_nd = sorted(iou_nd)
      iou_nd = np.array(iou_nd)
      # counting zero iou
      num_zero_iou = np.sum(iou_nd < 1e-7)
      print('num_zero_iou: ', num_zero_iou)
      for threshold in [0.5, 0.7]:
        num_matches = np.sum(iou_nd > threshold)
        accuracy = 1.0 * num_matches / num_boxes
        print('Accuracy: %.4f @ %s threshold %.1f' % (accuracy, iou_type, threshold))

    return


def eval_model_v2v4real_3d_grounding(args):
    with open(args.answers_file) as f:
        data = [json.loads(line) for line in f]

    num_boxes = len(data) # 31421
    num_parameters = 7

    gt_boxes = [parse_box_parameters(d['conversations'][1]['value']) for d in data]
    gt_boxes = np.stack(gt_boxes, axis=0)

    output_boxes = [parse_box_parameters(d['outputs']) for d in data]
    output_boxes = np.stack(output_boxes, axis=0)

    # counting parsing failures
    num_failed_parsing = num_boxes - np.sum(output_boxes[:, -1])
    print('num_failed_parsing: ', num_failed_parsing)
    output_boxes = output_boxes[:, :num_parameters]

    # eval metric 1: 2d distance < 0.5, nuscenes detection [0.5, 1, 2, 4] 
    eval_center_dist_2d(gt_boxes, output_boxes)

    # eval metric 2: 2d iou < 0.5
    eval_iou_3d(gt_boxes, output_boxes)
    return


def classify_answer_type(answer):
    '''
    follow the logic in data generation code
    https://github.com/eddyhkchiu/my_co_llm_driver/blob/32f2ce61303678d268f19dc72103f1ce357df821/DMSTrack/V2V4Real/opencood/tools/inference.py#L1172

    in gt box / there exist an object: [0, 1, 2, 3]
      0: in both
      1: in cav_ego only
      2: in cav_1 only
      3: out both
    out gt box / there is no object: [4, 5, 6, 7]
      4: in both
      5: in cav_ego only
      6: in cav_1 only
      7: out both
    '''
    if 'There is no' in answer:
        object_exist = 0
        if 'both connected autonomous vehicles' in answer:
            qa_sub_type = 4
        elif 'connected autonomous vehicle ego' in answer:
            qa_sub_type = 5
        elif 'connected autonomous vehicle 1' in answer:
            qa_sub_type = 6
        elif 'None of the connected autonomous vehicles' in answer:
            qa_sub_type = 7
        else:
            qa_sub_type = -1
    else:
        object_exist = 1
        if 'Both connected autonomous vehicles' in answer:
            qa_sub_type = 0
        elif 'Connected autonomous vehicle ego' in answer:
            qa_sub_type = 1
        elif 'Connected autonomous vehicle 1' in answer:
            qa_sub_type = 2
        elif 'None of the connected autonomous vehicles' in answer:
            qa_sub_type = 3
        else:
            qa_sub_type = -1

    return np.array([qa_sub_type, object_exist])


def evaluate_object_existance(gt_answer_types, output_answer_types):
    gt_object_exist = gt_answer_types[:,1]
    output_object_exist = output_answer_types[:,1]

    accuracy = np.sum(gt_object_exist == output_object_exist) / gt_object_exist.shape[0]
    print('accuracy: ', accuracy)

    TP = np.sum(np.logical_and(gt_object_exist, output_object_exist))
    print('TP: ', TP)
    FP = np.sum(np.logical_and(np.logical_not(gt_object_exist), output_object_exist))
    print('FP: ', FP)
    FN = np.sum(np.logical_and(gt_object_exist, np.logical_not(output_object_exist)))
    print('FN: ', FN)
    TN = np.sum(np.logical_and(np.logical_not(gt_object_exist), np.logical_not(output_object_exist)))
    print('TN: ', TN)

    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)

    return


def evaluate_reason(gt_answer_types, output_answer_types):
    '''
    For correct object exist output: TP and TN
    calculate the reason's 4-class classification accuracy
    '''
    gt_object_exist = gt_answer_types[:,1]
    output_object_exist = output_answer_types[:,1]

    TP_mask = np.logical_and(gt_object_exist, output_object_exist)
    TP_count = np.sum(TP_mask)
    correct_reason_count = np.sum(gt_answer_types[TP_mask][:,0] == output_answer_types[TP_mask][:,0])
    accuracy = 1.0 * correct_reason_count / TP_count
    print('TP reason accuracy: ', accuracy)

    TN_mask = np.logical_and(np.logical_not(gt_object_exist), np.logical_not(output_object_exist))
    TN_count = np.sum(TN_mask)
    correct_reason_count = np.sum(gt_answer_types[TN_mask][:,0] == output_answer_types[TN_mask][:,0])
    accuracy = 1.0 * correct_reason_count / TN_count
    print('TN reason accuracy: ', accuracy)

    return


def evaluate_multiple_box_accuracy(gt_answer_types, gt_boxes, output_answer_types, output_boxes, simplified, max_num_answer_objects):
    '''
    Input:
      gt_answer_types: (num_samples)
      gt_boxes: (num_samples, max_num_answer_objects=3, 3): 3: [x, z, is_valid]
      output_answer_types: (num_samples)
      output_boxes: (num_samples, max_num_answer_objects=3, 3): 3: [x, z, is_valid]

    We only evaluate TP cases: gt object exists, output answer also predict objects  
    '''

    if not simplified:
      num_parameters = 7
    else:
      num_parameters = 2

    # For TP, evaluate bounding box accuracy
    gt_object_exist = gt_answer_types[:,1]
    output_object_exist = output_answer_types[:,1]
    TP_mask = np.logical_and(gt_object_exist, output_object_exist)
    TP_count = np.sum(TP_mask)

    num_failed_parsing = np.sum(np.sum(output_boxes[TP_mask][:, :, -1], axis=1) == 0)
    print('num_failed_parsing: ', num_failed_parsing)
    parsing_success_rate = 1.0 * (TP_count - num_failed_parsing) / TP_count
    print('parsing_success_rate: ', parsing_success_rate)

    print('gt_boxes[TP_mask].shape: ', gt_boxes[TP_mask].shape)
    print('output_boxes[TP_mask].shape: ', output_boxes[TP_mask].shape)

    # eval metric 1: 2d distance < 0.5, nuscenes detection [0.5, 1, 2, 4] 
    eval_multiple_center_dist_2d(gt_boxes[TP_mask], output_boxes[TP_mask], simplified)

    if not simplified:
      print('not implemented')
      assert False
      eval_iou_3d(gt_boxes[TP_mask], output_boxes[TP_mask])

    return


def evaluate_box_accuracy(gt_answer_types, gt_boxes, output_answer_types, output_boxes, simplified):
    if not simplified:
      num_parameters = 7
    else:
      num_parameters = 2

    # For TP, evaluate bounding box accuracy
    gt_object_exist = gt_answer_types[:,1]
    output_object_exist = output_answer_types[:,1]
    TP_mask = np.logical_and(gt_object_exist, output_object_exist)
    TP_count = np.sum(TP_mask)

    # counting parsing failures
    num_failed_parsing = TP_count - np.sum(output_boxes[TP_mask][:, -1])
    print('num_failed_parsing: ', num_failed_parsing)
    parsing_success_rate = 1.0 * (TP_count - num_failed_parsing) / TP_count
    print('parsing_success_rate: ', parsing_success_rate)

    output_boxes = output_boxes[:, :num_parameters]
    eval_center_dist_2d(gt_boxes[TP_mask], output_boxes[TP_mask], simplified)

    if not simplified:
      # eval metric 2: 2d iou < 0.5
      eval_iou_3d(gt_boxes[TP_mask], output_boxes[TP_mask])

    return


def parse_behind_distance(answer):
  '''
  sample answer
  'Yes, there is a car 53 meters behind the object. Its bounding box parameters are [1.7, 1.9, 4.3, -80.7, -0.4, -6.7, 0.34]. Connected autonomous vehicle 1 detects it.'
  '''
  car_position = answer.find('car')
  meters_position = answer.find('meters')
  dist = answer[car_position+len('car '):meters_position].strip()
  try:
    dist = int(dist)
  except:
    return -1
  return dist


def select_and_sort_samples(selected_output_file, data, gt_answer_types, gt_boxes, output_answer_types, output_boxes):
  '''
  Select good samples for visualization
  good: 
    1. qa_sub_type == 2: gt box exists, cav_ego can not detect it, cav_1 can detects it
    2. LLM's answer is consistent with gt answer in classification part
    3. True positive cases
    4. LLM output box is close to GT box
  '''
  selected_result = []
  print('len(data): ', len(data))
  for i in range(len(data)):
    # keep the cases that gt exists, cav_ego can not detect it, cav_1 can
    if data[i]['qa_sub_type'] != 2:
      continue
    
    # keep the cases that LLM has correct object existance and reason classification answer
    if gt_answer_types[i][0] != output_answer_types[i][0]:
      continue

    # keep the cases that there exists a gt box, and LLM also thinks so
    if gt_answer_types[i][1] != 1 or output_answer_types[i][1] != 1:
      continue

    # keep the cases that LLM output box is not dummy one due to parsing error
    if output_boxes[i][3] == 0 and output_boxes[i][4] == 0 and output_boxes[i][5] == 0:
      continue

    # keep the cases that box accuracy l2_error < 0.5 meters
    l2_error = np.linalg.norm([gt_boxes[i][3] - output_boxes[i][3], gt_boxes[i][5] - output_boxes[i][5]])
    data[i]['l2_error'] = l2_error
    if l2_error > 4:
      continue

    # get behind distance from gt answer
    behind_distance = parse_behind_distance(data[i]['conversations'][1]['value'])
    data[i]['behind_distance'] = behind_distance

    selected_result.append(data[i])

  print('len(selected_result): ', len(selected_result))  

  # sort by box accuracy 2d center distance
  selected_result = sorted(selected_result, key=lambda x: x['l2_error'])

  # sort by behind_distance 
  #selected_result = sorted(selected_result, key=lambda x: x['behind_distance'])

  print('selected_result[:20]: ', selected_result[:20])

  with open(selected_output_file, 'w') as outfile:
    for result in selected_result:
      json.dump(result, outfile)    
      outfile.write('\n')

  return


def eval_model_v2v4real_3d_grounding_v2(args):
    with open(args.answers_file) as f:
        data = [json.loads(line) for line in f]
    #print('data[-1]: ', data[-1])
    # {"id": 0, "conversations": [{"from": "human", "value": "What is the object at the location [-20.5, -0.1]? What are its bounding box parameters?"}, {"from": "gpt", "value": "A car is at the location. Its bounding box parameters are [1.7, 2.1, 4.0, -20.5, -1.0, -0.1, 0.03]. Connected autonomous vehicle ego detects it."}], "scenario_index": 0, "local_timestamp_index": 0, "global_timestamp_index": 0, "qa_sub_type": 1, "outputs": "A car is at the location. Its bounding box parameters are [1.8, 2.2, 4.4, -20.5, -1.2, -0.1, 0.03]. Connected autonomous vehicle ego detects it."}

    num_boxes = len(data) # 31421

    gt_answer_types = np.stack([classify_answer_type(d['conversations'][1]['value']) for d in data], axis=0)
    print('data[0]: ', data[0])
    gt_boxes = [parse_box_parameters(d['conversations'][1]['value'], args.simplified) for d in data]
    gt_boxes = np.stack(gt_boxes, axis=0)

    output_answer_types = np.stack([classify_answer_type(d['outputs']) for d in data], axis=0)
    output_boxes = [parse_box_parameters(d['outputs'], args.simplified) for d in data]
    output_boxes = np.stack(output_boxes, axis=0)

    evaluate_object_existance(gt_answer_types, output_answer_types)
    # Note that we can ignore evaluate_reason result for v2xreal
    evaluate_reason(gt_answer_types, output_answer_types)
    evaluate_box_accuracy(gt_answer_types, gt_boxes, output_answer_types, output_boxes, args.simplified)

    #select_and_sort_samples(args.selected_output_file, data, gt_answer_types, gt_boxes, output_answer_types, output_boxes)

    return


def check_has_collision(future_cav_location_in_current_ego_coordinate, global_timestamp_index, future_global_timestamp_index, npy_save_path, asker_cav_id):
    '''
    Load cav_ego's lidar pose at global_timestamp_index, future_global_timestamp_index
    Transform cav_ego's future box parameters from current coordinate to future coordinate
    Check collision in future coordinate with future gt boxes
    '''
    current_ego_lidar_pose = np.load(os.path.join(npy_save_path, 'ego', '%04d_lidar_pose.npy' % global_timestamp_index))
    future_ego_lidar_pose = np.load(os.path.join(npy_save_path, 'ego', '%04d_lidar_pose.npy' % future_global_timestamp_index))

    current_ego_pose_in_future_coordinate = x1_to_x2(current_ego_lidar_pose, future_ego_lidar_pose)
    future_ego_pose_in_future_coordinate = current_ego_pose_in_future_coordinate.copy()
    future_ego_pose_in_future_coordinate[0, 3] += future_cav_location_in_current_ego_coordinate[0]
    future_ego_pose_in_future_coordinate[1, 3] += future_cav_location_in_current_ego_coordinate[1]
    # assume cav box has the following size
    future_ego_box = np.array([
      1.5, 2, 4, # h, w, l
      future_ego_pose_in_future_coordinate[0, 3], future_ego_pose_in_future_coordinate[1, 3], 0, # x, y, z
      0 # a
    ])
    future_cav_box = future_ego_box


    # New general approach for both CAVs
    # transform future_cav_location_in_current_ego_coordinate to future ego coordinate
    transformation_matrix = x1_to_x2(current_ego_lidar_pose, future_ego_lidar_pose)
    future_cav_location_in_current_ego_coordinate = np.expand_dims(future_cav_location_in_current_ego_coordinate, axis=0)
    future_cav_location_in_current_ego_coordinate = np.pad(future_cav_location_in_current_ego_coordinate, [[0, 0], [0, 1]])
    future_cav_location_in_future_ego_coordinate = project_points_by_matrix_torch(future_cav_location_in_current_ego_coordinate, transformation_matrix)
    future_cav_location_in_future_ego_coordinate = future_cav_location_in_future_ego_coordinate[0, :2]
    # assume cav box has the following size
    future_cav_box = np.array([
      1.5, 2, 4, # h, w, l
      future_cav_location_in_future_ego_coordinate[0], future_cav_location_in_future_ego_coordinate[1], 0, # x, y, z
      0 # a
    ])

    # get cav_1's gt future location at future ego coordinate
    # for collision check
    # for v2xreal, the other cav id is '2'
    # at some frame, there is no other cav
    # here we actualy want to get asker's gt future location in future ego's coordinate system
    future_asker_lidar_pose = np.load(os.path.join(npy_save_path, asker_cav_id, '%04d_lidar_pose.npy' % future_global_timestamp_index))
    future_asker_to_ego_transformation_matrix = x1_to_x2(future_asker_lidar_pose, future_ego_lidar_pose)
    cav_asker_gt_future_in_future_ego_coordinate = np.array([future_asker_to_ego_transformation_matrix[0, 3], future_asker_to_ego_transformation_matrix[1, 3]])

    # [h, w, l, x, y, z, a]
    # which coordinate?
    # v2v4real
    # x: car forward, figure left
    # y: car right, figure up
    # z: car up, figure screen to face

    gt_boxes_in_future_coordinate = np.load(os.path.join(npy_save_path, '%04d_gt.npy' % future_global_timestamp_index))
    #print('gt_boxes_in_future_coordinate: ', gt_boxes_in_future_coordinate)
    # (N, 8, 3), 3: x, y, z in v2v4real coordinate
    gt_boxes_in_future_coordinate = corner_to_center(gt_boxes_in_future_coordinate, order='hwl')
    #print('gt_boxes_in_future_coordinate: ', gt_boxes_in_future_coordinate)
    # (N, 7)
    # (N, [xyz, h, w, l, theta]) to (N, (h, w, l, x, y, z, rot_y))
    gt_boxes_in_future_coordinate = np.concatenate([
      gt_boxes_in_future_coordinate[:, 3:6], 
      gt_boxes_in_future_coordinate[:, 0:3], 
      gt_boxes_in_future_coordinate[:, 6:7]], 
    axis=1)
    #print('gt_boxes_in_future_coordinate: ', gt_boxes_in_future_coordinate)
    # [h, w, l, x, y, z, a]
    
    # check collision by calculating IOU
    # to AB3DMOT box format
    # Box3D(x, y, z, h, w, l, ry, s)
    gt_boxes_ab3dmot = [Box3D(b[3], b[4], b[5], b[0], b[1], b[2], b[6], 1) for b in gt_boxes_in_future_coordinate]
    future_cav_box_ab3dmot = [Box3D(b[3], b[4], b[5], b[0], b[1], b[2], b[6], 1) for b in [future_cav_box]]
    for i in range(gt_boxes_in_future_coordinate.shape[0]):
      iou_type = 'iou_3d'  
      iou_nd = iou(future_cav_box_ab3dmot[0], gt_boxes_ab3dmot[i], iou_type)
      if iou_nd > 0:
        # if checking collision for cav_ego and this colliding gt is cav_ego itself, it is not collision
        if asker_cav_id == 'ego':
          if np.linalg.norm([gt_boxes_in_future_coordinate[i][3], gt_boxes_in_future_coordinate[i][4]]) < 2:
            continue  
          else:
            return True  
        
        # if checking collision for cav_1 and this colliding gt is cav_1 itself, it is not collision
        if asker_cav_id == '1' or asker_cav_id == '2':
          if np.linalg.norm([gt_boxes_in_future_coordinate[i][3]-cav_asker_gt_future_in_future_ego_coordinate[0], gt_boxes_in_future_coordinate[i][4]-cav_asker_gt_future_in_future_ego_coordinate[1]]) < 2:
            continue  
          else:
            return True  

        # we should not hit this line
        assert False  

    return False


def evaluate_future_trajectory(num_future_waypoints, data, npy_save_path, qa_type_id):
    '''
    Evaluate L2 error between model output future waypoints and gt future trajectory
    Evaluate collision rate of model output waypoints and gt future boxes
    '''
    # data[0]:
    # {"id": 0, "conversations": [{"from": "human", "value": "What is the suggested future trajectory to avoid collision with nearby objects?"}, {"from": "gpt", "value": "The suggested future trajectory is [(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]."}], "scenario_index": 0, "local_timestamp_index": 0, "global_timestamp_index": 0, "qa_sub_type": [7], "distance_to_waypoint": null, "future_time": null, "gpt_reasoning_output": " None of the connected autonomous vehicles detects anything near your planned future trajectory.", "outputs": "The suggested future trajectory is [(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]."}

    # v7sd
    # {"id": 0, "conversations": [{"from": "human", "value": "I am CAV_EGO. What is the suggested future trajectory to avoid collision with nearby objects?"}, {"from": "gpt", "value": "The suggested future trajectory is [(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]."}], "scenario_index": 0, "local_timestamp_index": 0, "global_timestamp_index": 0, "qa_sub_type": [7], "distance_to_waypoint": null, "future_time": null, "gpt_reasoning_output": " None of the connected autonomous vehicles detects anything near your planned future trajectory.", "future_trajectory_str_in_ego": "[(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]", "future_trajectory_str_in_self": "[(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]", "asker_cav_id": "ego", "cav_ego_lidar_pose": [-0.8603860139846802, -0.5096380114555359, 0.0020271099638193846, -747.4990234375, 0.5096420049667358, -0.8603760004043579, 0.004268390126526356, 246.41400146484375, -0.0004312550008762628, 0.004705559927970171, 0.9999889731407166, 4.646309852600098, 0.0, 0.0, 0.0, 1.0], "cav_1_lidar_pose": [-0.6782050132751465, -0.7342339754104614, -0.03063499927520752, -685.0469970703125, 0.734870970249176, -0.6776940226554871, -0.026362700387835503, 203.3769989013672, -0.0014047699514776468, -0.04039210081100464, 0.9991829991340637, 7.019320011138916, 0.0, 0.0, 0.0, 1.0], "outputs": "The suggested future trajectory is [(8.9,0.2),(17.9,0.3),(27.0,0.3),(36.4,0.1),(45.9,-0.3),(55.6,-0.9)]."}

    l2_error_all_1s = []
    l2_error_all_2s = []
    l2_error_all_3s = []
    l2_error_max = 0
    collision_count_1s = 0
    collision_count_2s = 0
    collision_count_3s = 0
    ttc_collision_count = 0

    comfort_count = 0
    # https://github.com/motional/nuplan-devkit/blob/d60b4cd2071de9bb041509c43f5226dd22f248c0/docs/metrics_description.md
    min_lon_accel = -4.05 #m/s^2
    max_lon_accel = 2.40 #m/s^2
    max_abs_lat_accel = 4.89 #m/s^2
    max_abs_yaw_accel = 1.93 #rad/s^2
    max_abs_yaw_rate = 0.95 #rad/s
    max_abs_lon_jerk = 4.13 #m/s^3
    max_abs_mag_jerk = 8.37 #m/s^3

    sum_local_pdms = 0

    for data_sample_id in range(len(data)):

      data_sample = data[data_sample_id]      

      # backward compatible   
      if 'asker_cav_id' not in data_sample:
        data_sample['asker_cav_id'] = 'ego'  

      gt_answer = data_sample['conversations'][1]['value']  
      gt_future_waypoints = parse_planned_future_trajectory(gt_answer, num_future_waypoints, prepend_current_position=False) 
      output_future_waypoints = parse_planned_future_trajectory(data_sample['outputs'], num_future_waypoints, prepend_current_position=False)

      # In our latest V2V-GoT dataset, trajectory are already in CAV_EGO's coordinate system
      # But for V2V-LLM Q5, trajectory are from asker_cav's coordinate system
      # for backward compatibility, switch coordinate system if in V2V-LLM Q5
      # if asker is CAV_1, transform the waypoints from CAV_1's coordinates to CAV_EGO's
      if qa_type_id == 5 and 'asker_cav_id' in data_sample and (data_sample['asker_cav_id'] == '1' or data_sample['asker_cav_id'] == '2'):
        gt_future_waypoints = transform_waypoints_from_1_to_ego(gt_future_waypoints, data_sample)
        output_future_waypoints = transform_waypoints_from_1_to_ego(output_future_waypoints, data_sample)

      # NAVSIM PDMS calculation
      # calculate asker cav current position in cav_ego coordinate
      current_position_in_cav_asker = np.array([[0, 0]])
      if 'asker_cav_id' in data_sample and (data_sample['asker_cav_id'] != 'ego'):
        current_position_in_cav_ego = transform_waypoints_from_1_to_ego(current_position_in_cav_asker, data_sample)
      else:  
        current_position_in_cav_ego = current_position_in_cav_asker  

      ttc_output_future_waypoints = parse_planned_future_trajectory(data_sample['outputs'], num_future_waypoints, prepend_current_position=True, current_position=current_position_in_cav_ego)

      # Comfort
      velocity0 = (ttc_output_future_waypoints[1] - ttc_output_future_waypoints[0]) / 0.5
      velocity1 = (ttc_output_future_waypoints[2] - ttc_output_future_waypoints[1]) / 0.5
      velocity2 = (ttc_output_future_waypoints[3] - ttc_output_future_waypoints[2]) / 0.5
      accel0 = (velocity1 - velocity0) / 0.5
      accel1 = (velocity2 - velocity1) / 0.5
      jerk = (accel1 - accel0) / 0.5

      lon_accel = accel0[0]
      lat_accel = accel0[1]
      lon_jerk = jerk[0]
      mag_jerk = np.linalg.norm(jerk)

      angle0 = np.arctan2(velocity0[1], velocity0[0])
      angle1 = np.arctan2(velocity1[1], velocity1[0])
      angle2 = np.arctan2(velocity2[1], velocity2[0])
      yaw_rate0 = (angle1 - angle0) / 0.5
      yaw_rate1 = (angle2 - angle1) / 0.5
      yaw_rate = yaw_rate0
      yaw_accel = (yaw_rate1 - yaw_rate0) / 0.5
      local_comfort_count = 0
      if lon_accel >= min_lon_accel and lon_accel <= max_lon_accel and np.abs(lat_accel) <= max_abs_lat_accel and np.abs(yaw_accel) <= max_abs_yaw_accel and np.abs(yaw_rate) <= max_abs_yaw_rate and np.abs(lon_jerk) <= max_abs_lon_jerk and np.abs(mag_jerk) <= max_abs_mag_jerk:
        comfort_count += 1
        local_comfort_count = 1

      # TTC calculation
      # Projected future waypoint by constant velocity
      # at future 0.3, 0.6, 0.9 seconds
      origin = ttc_output_future_waypoints[0]
      velocity = (ttc_output_future_waypoints[1] - ttc_output_future_waypoints[0]) / 0.5
      projected_future_waypoints = np.stack([
        origin + velocity * 0.3, 
        origin + velocity * 0.6, 
        origin + velocity * 0.9 
      ], axis=0)

      local_ttc_collision_count = 0
      for i in range(projected_future_waypoints.shape[0]):
        global_timestamp_index = data_sample['global_timestamp_index']  
        future_global_timestamp_index = global_timestamp_index + 3 * (i+1)
        future_cav_location_in_current_ego_coordinate = projected_future_waypoints[i]
        has_collision = check_has_collision(future_cav_location_in_current_ego_coordinate, global_timestamp_index, future_global_timestamp_index, npy_save_path, data_sample['asker_cav_id'])
        if has_collision:
          ttc_collision_count += 1
          local_ttc_collision_count = 1
          break

      # L2 error
      l2_error = np.linalg.norm(output_future_waypoints - gt_future_waypoints, axis=1)
      l2_error_3s = np.average(l2_error)
      l2_error_all_3s.append(l2_error_3s)
      l2_error_2s = np.average(l2_error[:4])
      l2_error_all_2s.append(l2_error_2s)
      l2_error_1s = np.average(l2_error[:2])
      l2_error_all_1s.append(l2_error_1s)

      if l2_error_3s > l2_error_max:
        l2_error_max = l2_error_3s
      
      # Collision
      local_collision_count_3s = 0
      for i in range(num_future_waypoints):
        global_timestamp_index = data_sample['global_timestamp_index']  
        future_global_timestamp_index = global_timestamp_index + int(30 / num_future_waypoints) * (i+1)
        future_cav_location_in_current_ego_coordinate = output_future_waypoints[i]
        has_collision = check_has_collision(future_cav_location_in_current_ego_coordinate, global_timestamp_index, future_global_timestamp_index, npy_save_path, data_sample['asker_cav_id'])
        if has_collision:    
          if i < 2:
            collision_count_1s += 1
            collision_count_2s += 1
            collision_count_3s += 1
            local_collision_count_3s = 1
            break
          elif i < 4:
            collision_count_2s += 1
            collision_count_3s += 1
            local_collision_count_3s = 1
            break
          else:
            collision_count_3s += 1
            local_collision_count_3s = 1
            break

      # Local PDMS
      local_pdms = (1 - local_collision_count_3s) * ((1 - local_ttc_collision_count) * 5.0 + local_comfort_count * 2.0) / 7.0
      sum_local_pdms += local_pdms

    # end of loop over all data sample


    l2_error_avg_1s = sum(l2_error_all_1s) / len(l2_error_all_1s)
    print('l2_error_avg_1s: ', l2_error_avg_1s)
    l2_error_avg_2s = sum(l2_error_all_2s) / len(l2_error_all_2s)
    print('l2_error_avg_2s: ', l2_error_avg_2s)
    l2_error_avg_3s = sum(l2_error_all_3s) / len(l2_error_all_3s)
    print('l2_error_avg_3s: ', l2_error_avg_3s)
    l2_error_avg_all = (l2_error_avg_1s + l2_error_avg_2s + l2_error_avg_3s) / 3
    print('l2_error_avg_all: ', l2_error_avg_all)

    collision_rate_1s = 1.0 * collision_count_1s / len(data)
    print('collision_rate_1s: ', collision_rate_1s)
    collision_rate_2s = 1.0 * collision_count_2s / len(data)
    print('collision_rate_2s: ', collision_rate_2s)
    collision_rate_3s = 1.0 * collision_count_3s / len(data)
    print('collision_rate_3s: ', collision_rate_3s)
    collision_rate_avg_all = (collision_rate_1s + collision_rate_2s + collision_rate_3s) / 3
    print('collision_rate_avg_all: ', collision_rate_avg_all)

    # NAVSIM PDMS
    NC = 1 - collision_rate_3s
    DAC = 1
    EP = 1
    TTC = 1 - (1.0 * ttc_collision_count / len(data))
    C = 1.0 * comfort_count / len(data)

    PDMS = NC * (TTC * 5 + C * 2) / 7

    print('NC: ', NC)

    print('TTC: ', TTC)

    print('C: ', C)

    print('PDMS: ', PDMS)

    PDMS_sample_average = sum_local_pdms /  len(data)
    print('PDMS_sample_average: ', PDMS_sample_average)

    return


def parse_suggested_speed_steering_idx(answer):
    '''
    speed_classes = ['fast', 'moderate', 'slow', 'very slow', 'stop']
    steering_classes = ['left', 'slightly left', 'straight', 'slightly right', 'right']
    '''
    if 'fast' in answer:
      suggested_speed_idx = 0
    elif 'moderate' in answer:
      suggested_speed_idx = 1
    elif 'very slow' in answer:
      suggested_speed_idx = 3
    elif 'slow' in answer:
      suggested_speed_idx = 2
    elif 'stop' in answer: 
      suggested_speed_idx = 4
    else: # parse error 
      suggested_speed_idx = 4

    if 'straight' in answer:
      suggested_steering_idx = 2  
    elif 'slightly left' in answer:
      suggested_steering_idx = 1  
    elif 'slightly right' in answer:
      suggested_steering_idx = 3  
    elif 'left' in answer:
      suggested_steering_idx = 0  
    elif 'right' in answer:
      suggested_steering_idx = 4  
    else: # parse error 
      suggested_steering_idx = 2  

    return suggested_speed_idx, suggested_steering_idx


def evaluate_suggested_speed_steering(data, npy_save_path):
    '''
    Evaluate suggested speed and steering setting in this 5x5 classification problem
    speed_accuracy: 5
    sterring_accuracy: 5
    action_accuracy: 25
    edit_dist: average diff between gt idx and model output idx
    '''
    print('data[0]: ', data[0])
    #  {'id': 0, 'conversations': [{'from': 'human', 'value': 'I am CAV_EGO. What are the suggested speed and steering settings?'}, {'from': 'gpt', 'value': 'The suggested speed setting is: fast. The suggested steering setting is: straight.'}], 'scenario_index': 0, 'local_timestamp_index': 0, 'global_timestamp_index': 0, 'qa_sub_type': 2, 'suggested_speed_idx': 0, 'suggested_steering_idx': 2, 'dist': 8.768972905266613, 'angle': 0.6360308611218952, 'future_trajectory_str_in_ego': '[(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]', 'future_trajectory_str_in_self': '[(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]', 'asker_cav_id': 'ego', 'cav_ego_lidar_pose': [-0.8603860139846802, -0.5096380114555359, 0.0020271099638193846, -747.4990234375, 0.5096420049667358, -0.8603760004043579, 0.004268390126526356, 246.41400146484375, -0.0004312550008762628, 0.004705559927970171, 0.9999889731407166, 4.646309852600098, 0.0, 0.0, 0.0, 1.0], 'cav_1_lidar_pose': [-0.6782050132751465, -0.7342339754104614, -0.03063499927520752, -685.0469970703125, 0.734870970249176, -0.6776940226554871, -0.026362700387835503, 203.3769989013672, -0.0014047699514776468, -0.04039210081100464, 0.9991829991340637, 7.019320011138916, 0.0, 0.0, 0.0, 1.0], 'outputs': 'The suggested speed setting is: fast. The suggested steering setting is: straight.'}

    speed_correct_count = 0
    steering_correct_count = 0
    action_correct_count = 0

    speed_edit_dist = 0
    steering_edit_dist = 0
    action_edit_dist = 0

    non_straight_steering_count = 0

    for data_sample in data:
      gt_suggested_speed_idx, gt_suggested_steering_idx = parse_suggested_speed_steering_idx(data_sample['conversations'][1]['value'])  
      output_suggested_speed_idx, output_suggested_steering_idx = parse_suggested_speed_steering_idx(data_sample['outputs'])  

      if output_suggested_steering_idx != 2:
        non_straight_steering_count += 1  

      if gt_suggested_speed_idx == output_suggested_speed_idx:
        speed_correct_count += 1
      if gt_suggested_steering_idx == output_suggested_steering_idx:
        steering_correct_count += 1
      if (gt_suggested_speed_idx == output_suggested_speed_idx) and (gt_suggested_steering_idx == output_suggested_steering_idx):  
        action_correct_count += 1  

      speed_edit_dist += abs(gt_suggested_speed_idx - output_suggested_speed_idx)  
      steering_edit_dist += abs(gt_suggested_steering_idx - output_suggested_steering_idx)
      action_edit_dist += abs(gt_suggested_speed_idx - output_suggested_speed_idx) + abs(gt_suggested_steering_idx - output_suggested_steering_idx)

    speed_accuracy = 1.0 * speed_correct_count / len(data)
    steering_accuracy = 1.0 * steering_correct_count / len(data)
    action_accuracy = 1.0 * action_correct_count / len(data)

    speed_edit_dist /= 1.0 * len(data)
    steering_edit_dist /= 1.0 * len(data)
    action_edit_dist /= 1.0 * len(data)

    print('speed_accuracy: ', speed_accuracy)
    print('steering_accuracy: ', steering_accuracy)
    print('action_accuracy: ', action_accuracy)
    print('speed_edit_dist: ', speed_edit_dist)
    print('steering_edit_dist: ', steering_edit_dist)
    print('action_edit_dist: ', action_edit_dist)

    print('non_straight_steering_count: ', non_straight_steering_count)
    print('non_straight_steering_count %: ', non_straight_steering_count * 1.0 / len(data))
    return 


def parse_notable_object_prediction_answer(qa_type_id, gt_answer, max_num_answer_objects, num_future_waypoints):
    notable_gts_action_classes = ['moving forward', 'turning left', 'turning right', 'staying at the same location']

    #print('gt_answer: ', gt_answer)
    # 'There is a car at (-20.5,-0.1) moving forward. The predicted future trajectory is [(-11.5,0.3),(-2.5,0.5),(6.3,0.8),(15.2,1.1),(24.1,1.4),(32.9,1.6)]. There is a ...'
    if  qa_type_id == 16:
      # NQ6 prediction by CAV planning  
      num_answer_objects = gt_answer.count('CAV_')  
    else:  
      num_answer_objects = gt_answer.count('There is a')
    num_answer_objects = min(num_answer_objects, max_num_answer_objects)
    num_answer_objects_correctly_parsed = 0

    gts_location = np.zeros([num_answer_objects, 2])
    gts_action = np.zeros(num_answer_objects)
    gts_future_waypoints = np.zeros([num_answer_objects, num_future_waypoints, 2])

    has_parsing_location_error = False
    has_parsing_action_error = False
    has_parsing_trajectory_error = False

    start_idx = 0
    for i in range(num_answer_objects):
      # parse location  
      open_idx = gt_answer.find('(', start_idx)
      close_idx = gt_answer.find(')', start_idx)
      temp_str = gt_answer[open_idx+1: close_idx]
      temp_str = '[' + temp_str + ']'
      location = parse_box_parameters(temp_str, True)
      if location[2] != 0:
         gts_location[i] = location[:2].copy()
      else: 
        print('parse location error') 
        print('answer: ', gt_answer)
        has_parsing_location_error = True
        continue
        assert False

      # parse action

      if  qa_type_id == 16:
        # NQ6 prediction by CAV planning  
        next_there_index = gt_answer.find('CAV_', close_idx)
      else:  
        next_there_index = gt_answer.find('There', close_idx)
      temp_str = gt_answer[close_idx:next_there_index] #
      found_action = False
      for action_idx in range(len(notable_gts_action_classes)):
        if notable_gts_action_classes[action_idx] in temp_str:
          gts_action[i] = action_idx  
          found_action = True
      if not found_action:
        has_parsing_action_error = True  
        if qa_type_id in [15]: # only NQ5 needs parse action
          print('parse action error')
          print('answer: ', gt_answer)

      # parse future waypoint
      temp_str = gt_answer[close_idx+1:]
      future_waypoints, future_waypoints_has_parsing_error = parse_planned_future_trajectory(temp_str, num_future_waypoints, False, return_has_parsing_error = True)
      # if parsing trajectory error, set it to be static
      if future_waypoints_has_parsing_error:
        has_parsing_trajectory_error = True  
        print('parse trajectory error')
        print('answer: ', gt_answer)
        for j in range(num_future_waypoints):  
          gts_future_waypoints[i][j] = gts_location[i].copy()  
      else: # parse traj succeed    
        if future_waypoints is not None:  
          gts_future_waypoints[i] = future_waypoints.copy()

      # for next notable object parsing
      if  qa_type_id == 16:
        # NQ6 prediction by CAV planning  
        start_idx = gt_answer.find('CAV_', start_idx+1)
      else:  
        start_idx = gt_answer.find('There is a', start_idx+1)

      num_answer_objects_correctly_parsed += 1


    gts_location = gts_location[:num_answer_objects_correctly_parsed]
    gts_action = gts_action[:num_answer_objects_correctly_parsed]
    gts_future_waypoints = gts_future_waypoints[:num_answer_objects_correctly_parsed]

    return gts_location, gts_action, gts_future_waypoints, has_parsing_location_error, has_parsing_action_error, has_parsing_trajectory_error


# https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py#L360
def greedy_match(distance_matrix, match_threshold):
  '''
  Find the one-to-one matching using greedy allgorithm choosing small distance
  distance_matrix: (num_detections, num_tracks)
  '''
  matched_indices = []

  num_detections, num_tracks = distance_matrix.shape
  distance_1d = distance_matrix.reshape(-1)
  index_1d = np.argsort(distance_1d)
  index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
  detection_id_matches_to_tracking_id = [-1] * num_detections
  tracking_id_matches_to_detection_id = [-1] * num_tracks
  for sort_i in range(index_2d.shape[0]):
    detection_id = int(index_2d[sort_i][0])
    tracking_id = int(index_2d[sort_i][1])
    if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
      if distance_matrix[detection_id][tracking_id] < match_threshold:
        tracking_id_matches_to_detection_id[tracking_id] = detection_id
        detection_id_matches_to_tracking_id[detection_id] = tracking_id
        matched_indices.append([detection_id, tracking_id])

  matched_indices = np.array(matched_indices)
  return matched_indices



def match_gt_and_output(gts_location, outputs_location, match_threshold):
    '''
    gts_location:     (Ng, 2) 
    outputs_location: (No, 2) 
    '''

    Ng = gts_location.shape[0]
    No = outputs_location.shape[0]

    distance_matrix = np.zeros([Ng, No])
    for i in range(Ng):
      for j in range(No):
        distance_matrix[i][j] = np.sqrt((gts_location[i][0] - outputs_location[j][0])**2 + (gts_location[i][1] - outputs_location[j][1])**2)

    matched_indices = greedy_match(distance_matrix, match_threshold)
    return matched_indices, distance_matrix


def evaluate_notable_objects_prediction(max_num_answer_objects, num_future_waypoints, data, npy_save_path, qa_type_id, match_threshold):
    print('data[0]: ', data[0])
    # {'id': 0, 'conversations': [{'from': 'human', 'value': 'I am CAV_EGO. Where might those notable ojbects move in the future if my planned future trajectory is [(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]?'}, {'from': 'gpt', 'value': 'There is no notable object.'}], 'scenario_index': 0, 'local_timestamp_index': 0, 'global_timestamp_index': 0, 'qa_sub_type': [-1], 'distance_to_waypoint': None, 'future_time': None, 'future_trajectory_str_in_ego': '[(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]', 'future_trajectory_str_in_self': '[(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]', 'asker_cav_id': 'ego', 'cav_ego_lidar_pose': [-0.8603860139846802, -0.5096380114555359, 0.0020271099638193846, -747.4990234375, 0.5096420049667358, -0.8603760004043579, 0.004268390126526356, 246.41400146484375, -0.0004312550008762628, 0.004705559927970171, 0.9999889731407166, 4.646309852600098, 0.0, 0.0, 0.0, 1.0], 'cav_1_lidar_pose': [-0.6782050132751465, -0.7342339754104614, -0.03063499927520752, -685.0469970703125, 0.734870970249176, -0.6776940226554871, -0.026362700387835503, 203.3769989013672, -0.0014047699514776468, -0.04039210081100464, 0.9991829991340637, 7.019320011138916, 0.0, 0.0, 0.0, 1.0], 'outputs': 'There is no notable object.'}
    #print('data[1]: ', data[1])
    # {'id': 1, 'conversations': [{'from': 'human', 'value': 'I am CAV_1. Where might those notable ojbects move in the future if my planned future trajectory is [(-68.0,3.4),(-59.9,1.4),(-51.8,0.5),(-43.9,0.6),(-35.5,0.4),(-26.8,0.3)]?'}, {'from': 'gpt', 'value': 'There is a car at (-20.5,-0.1) moving forward. The predicted future trajectory is [(-11.5,0.3),(-2.5,0.5),(6.3,0.8),(15.2,1.1),(24.1,1.4),(32.9,1.6)]. There is a car at (-21.1,1.5) moving forward. The predicted future trajectory is [(-21.1,1.5),(-21.1,1.5),(-21.1,1.5),(-21.1,1.5),(-21.1,1.5),(26.8,3.1)]. '}], 'scenario_index': 0, 'local_timestamp_index': 0, 'global_timestamp_index': 0, 'qa_sub_type': [0.0, 0.0], 'distance_to_waypoint': [6.345685982198006, 5.8998676466399065], 'future_time': [29.0, 29.0], 'future_trajectory_str_in_ego': '[(-68.0,3.4),(-59.9,1.4),(-51.8,0.5),(-43.9,0.6),(-35.5,0.4),(-26.8,0.3)]', 'future_trajectory_str_in_self': '[(7.9,0.5),(16.2,1.0),(24.2,2.5),(31.8,4.8),(39.9,7.0),(48.2,9.4)]', 'asker_cav_id': '1', 'cav_ego_lidar_pose': [-0.8603860139846802, -0.5096380114555359, 0.0020271099638193846, -747.4990234375, 0.5096420049667358, -0.8603760004043579, 0.004268390126526356, 246.41400146484375, -0.0004312550008762628, 0.004705559927970171, 0.9999889731407166, 4.646309852600098, 0.0, 0.0, 0.0, 1.0], 'cav_1_lidar_pose': [-0.6782050132751465, -0.7342339754104614, -0.03063499927520752, -685.0469970703125, 0.734870970249176, -0.6776940226554871, -0.026362700387835503, 203.3769989013672, -0.0014047699514776468, -0.04039210081100464, 0.9991829991340637, 7.019320011138916, 0.0, 0.0, 0.0, 1.0], 'outputs': 'There is a car at (-20.5,-0.1) moving forward. The predicted future trajectory is [(-11.5,0.3),(-2.5,0.5),(6.3,0.8),(15.2,1.1),(24.1,1.4),(32.9,1.5)].'}


    # evaluate binary classification: at least notable object exist or not
    binary_tp = 0
    binary_fn = 0
    binary_fp = 0
    binary_tn = 0

    # evaluate notable object current location, f1, p, r
    thresholds = [0.5, 1, 2, 4]
    num_matched_gt_output = np.zeros(4)
    num_gts = 0
    num_outputs = 0

    # evaluate notable object action classification
    num_matched_gt_output_correct_action_count = 0
    num_matched_gt_output_incorrect_action_count = 0

    # evaluate notable object predicted future waypoint
    # 0s is error of current location
    l2_error_all_0s = 0
    l2_error_all_1s = 0
    l2_error_all_2s = 0
    l2_error_all_3s = 0
    l2_error_count = 0
    l2_error_max = 0

    # sanity check parse error
    gt_parse_error_count = 0
    output_parse_error_count = 0

    for data_sample_id in range(len(data)):
      data_sample = data[data_sample_id]  
      gt_answer = data_sample['conversations'][1]['value']
      output = data_sample['outputs']

      if 'There is a' in gt_answer or 'CAV_' in gt_answer:
        if 'There is a' in output or 'CAV_' in output:
          binary_tp += 1  
        else:  
          binary_fn += 1  
      else:
        if 'There is a' in output or 'CAV_' in output: 
          binary_fp += 1  
        else:  
          binary_tn += 1  


      # The following only consider there exist at least one gt and one output
      gts_location, gts_action, gts_future_waypoints, gts_has_parsing_location_error, gts_has_parsing_action_error, gts_has_parsing_trajectory_error = parse_notable_object_prediction_answer(qa_type_id, gt_answer, max_num_answer_objects, num_future_waypoints)
      num_gts += gts_location.shape[0]
      if gts_has_parsing_location_error: 
        # NQ4 or NQ5 need notable object identification 
        # NQ1, NQ2, NQ3 also need to parse location correctly
        gt_parse_error_count += 1  
        assert False
      elif qa_type_id == 15: # NQ5 also needs action and trajectory
        if gts_has_parsing_action_error or gts_has_parsing_trajectory_error:
          gt_parse_error_count += 1
          assert False


      outputs_location, outputs_action, outputs_future_waypoints, outputs_has_parsing_location_error, outputs_has_parsing_action_error, outputs_has_parsing_trajectory_error = parse_notable_object_prediction_answer(qa_type_id, output, max_num_answer_objects, num_future_waypoints)
      num_outputs += outputs_location.shape[0]
      if outputs_has_parsing_location_error: # NQ4 or NQ5 need notable object identification 
        output_parse_error_count += 1  
      elif qa_type_id == 15: # NQ5 also needs action and trajectory
        if outputs_has_parsing_action_error or outputs_has_parsing_trajectory_error:
          output_parse_error_count += 1

      # If there is at least one gt and at least one output, 
      # evaluate the predicted trajectory l2 error
      # otherwise no way to measure l2 error
      if len(gts_location) == 0 or len(outputs_location) == 0:
        continue  

      matched_indices, distance_matrix = match_gt_and_output(gts_location, outputs_location, match_threshold)

      for i in range(len(matched_indices)):
        # evaluate notable object current location, f1, p, r
        for threshold_id in range(len(thresholds)):
          if distance_matrix[matched_indices[i][0], matched_indices[i][1]] < thresholds[threshold_id]:
            num_matched_gt_output[threshold_id] += 1  
        l2_error_all_0s += distance_matrix[matched_indices[i][0], matched_indices[i][1]]

        # evaluate notable object action classification
        if gts_action[matched_indices[i][0]] == outputs_action[matched_indices[i][1]]:
          num_matched_gt_output_correct_action_count += 1  
        else:  
          num_matched_gt_output_incorrect_action_count += 1  
          
        # evaluate notable object predicted future waypoint
        if num_future_waypoints > 0:
          l2_error = np.linalg.norm(gts_future_waypoints[matched_indices[i][0]] - outputs_future_waypoints[matched_indices[i][1]], axis=1)
          l2_error_all_3s += np.average(l2_error)
          if num_future_waypoints == 6:
            l2_error_all_2s += np.average(l2_error[:4])
            l2_error_all_1s += np.average(l2_error[:2])

          l2_error_count += 1
          if np.average(l2_error) > l2_error_max:
            l2_error_max = np.average(l2_error)  

    # end of loop over all data samples

    print('num_future_waypoints: ', num_future_waypoints)
    if num_future_waypoints > 0:
      l2_error_avg_1s = l2_error_all_1s / l2_error_count
      l2_error_avg_2s = l2_error_all_2s / l2_error_count
      l2_error_avg_3s = l2_error_all_3s / l2_error_count
      print('l2_error_avg_1s: ', l2_error_avg_1s)
      print('l2_error_avg_2s: ', l2_error_avg_2s)
      print('l2_error_avg_3s: ', l2_error_avg_3s)
      l2_error_avg_123_all = (l2_error_avg_1s + l2_error_avg_2s + l2_error_avg_3s) / 3
      print('l2_error_avg_123_all: ', l2_error_avg_123_all)

      # for prediction task that we only predict one future waypoint
      l2_error_avg_0s = l2_error_all_0s / l2_error_count
      print('l2_error_avg_0s: ', l2_error_avg_0s)
      l2_error_avg_03_all = (l2_error_avg_0s + l2_error_avg_3s) / 2
      print('l2_error_avg_03_all: ', l2_error_avg_03_all)
    
    action_accuracy = 1.0 * num_matched_gt_output_correct_action_count / (num_matched_gt_output_correct_action_count + num_matched_gt_output_incorrect_action_count)
    print('action_accuracy: ', action_accuracy)

    for threshold_id in range(len(thresholds)):
      localization_precision = 1.0 * num_matched_gt_output[threshold_id] / num_outputs
      localization_recall = 1.0 * num_matched_gt_output[threshold_id] / num_gts
      localization_f1 = 2 * localization_precision * localization_recall / (localization_precision + localization_recall)
      print('localization_f1 @ %f: %f' % (thresholds[threshold_id], localization_f1))
      print('localization_precision @ %f: %f' % (thresholds[threshold_id], localization_precision))
      print('localization_recall@ %f: %f' % (thresholds[threshold_id], localization_recall))


    binary_precision = 1.0 * binary_tp / (binary_tp + binary_fp)
    binary_recall = 1.0 * binary_tp / (binary_tp + binary_fn)
    binary_f1 = 2 * binary_precision * binary_recall / (binary_precision + binary_recall)
    print('binary_f1: ', binary_f1)
    print('binary_precision: ', binary_precision)
    print('binary_recall: ', binary_recall)

    
    gt_parse_error_rate = 1.0 * gt_parse_error_count / len(data)
    output_parse_error_rate = 1.0 * output_parse_error_count / len(data)
    print('gt_parse_error_rate: ', gt_parse_error_rate)
    print('output_parse_error_rate: ', output_parse_error_rate)

    return


def evaluate_is_another_cav_notable_object(data):
    correct_count = 0
    for data_sample_id in range(len(data)):
      data_sample = data[data_sample_id]
      gt_answer = data_sample['conversations'][1]['value']
      output = data_sample['outputs']
      if 'not notable' in gt_answer and 'not notable' in output:
        correct_count += 1
      elif 'is a notable' in gt_answer and 'is a notable' in output:
        correct_count += 1

    accuracy = 1.0 * correct_count / len(data)
    print('evaluate_is_another_cav_notable_object')
    print('binary classification accuracy: ', accuracy)
    return


def eval_model_v2v4real_3d_grounding_v6(args):
    print('args.answers_file: ', args.answers_file)
    with open(args.answers_file) as f:
        data = [json.loads(line) for line in f]
    #print('data[-1]: ', data[-1])
    # {"id": 354, "conversations": [{"from": "human", "value": "Is there anything I need to be aware of if my planned future trajectory is [(15.0,0.4),(30.3,0.8),(44.9,1.2)]?"}, {"from": "gpt", "value": "Yes, there are cars at [19.6, -5.0], [19.7, -7.8], which are close to your planned future trajectory. Both connected autonomous vehicles detect them."}], "scenario_index": 3, "local_timestamp_index": 39, "global_timestamp_index": 444, "qa_sub_type": [1, 2], "distance_to_waypoint": [5.470780937448541, 8.344514214766985], "future_time": [12.0, 12.0], "outputs": "Yes, there are cars at [19.9, -2.0], [24.1, 1.0], which are close to your planned future trajectory. Connected autonomous vehicle ego detects them."}

    # v7sd
    # {"id": 0, "conversations": [{"from": "human", "value": "I am CAV_EGO. What is the suggested future trajectory to avoid collision with nearby objects?"}, {"from": "gpt", "value": "The suggested future trajectory is [(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]."}], "scenario_index": 0, "local_timestamp_index": 0, "global_timestamp_index": 0, "qa_sub_type": [7], "distance_to_waypoint": null, "future_time": null, "gpt_reasoning_output": " None of the connected autonomous vehicles detects anything near your planned future trajectory.", "future_trajectory_str_in_ego": "[(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]", "future_trajectory_str_in_self": "[(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]", "asker_cav_id": "ego", "cav_ego_lidar_pose": [-0.8603860139846802, -0.5096380114555359, 0.0020271099638193846, -747.4990234375, 0.5096420049667358, -0.8603760004043579, 0.004268390126526356, 246.41400146484375, -0.0004312550008762628, 0.004705559927970171, 0.9999889731407166, 4.646309852600098, 0.0, 0.0, 0.0, 1.0], "cav_1_lidar_pose": [-0.6782050132751465, -0.7342339754104614, -0.03063499927520752, -685.0469970703125, 0.734870970249176, -0.6776940226554871, -0.026362700387835503, 203.3769989013672, -0.0014047699514776468, -0.04039210081100464, 0.9991829991340637, 7.019320011138916, 0.0, 0.0, 0.0, 1.0], "outputs": "The suggested future trajectory is [(8.9,0.2),(17.9,0.3),(27.0,0.3),(36.4,0.1),(45.9,-0.3),(55.6,-0.9)]."}

    # v6sd
    #{"id": 0, "conversations": [{"from": "human", "value": "I am CAV_1. Is there anything I need to be aware of if my planned future trajectory is [(-68.0,3.4),(-59.9,1.4),(-51.8,0.5),(-43.9,0.6),(-35.5,0.4),(-26.8,0.3)]?"}, {"from": "gpt", "value": "Yes, there are cars at [-20.5, -0.1], [-21.1, 1.5], which are close to your planned future trajectory. Both connected autonomous vehicles detect them."}], "scenario_index": 0, "local_timestamp_index": 0, "global_timestamp_index": 0, "qa_sub_type": [1, 2], "distance_to_waypoint": [6.345685982198006, 5.8998676466399065], "future_time": [29.0, 29.0], "future_trajectory_str_in_ego": "[(-68.0,3.4),(-59.9,1.4),(-51.8,0.5),(-43.9,0.6),(-35.5,0.4),(-26.8,0.3)]", "future_trajectory_str_in_self": "[(7.9,0.5),(16.2,1.0),(24.2,2.5),(31.8,4.8),(39.9,7.0),(48.2,9.4)]", "asker_cav_id": "1", "cav_ego_lidar_pose": [-0.8603860139846802, -0.5096380114555359, 0.0020271099638193846, -747.4990234375, 0.5096420049667358, -0.8603760004043579, 0.004268390126526356, 246.41400146484375, -0.0004312550008762628, 0.004705559927970171, 0.9999889731407166, 4.646309852600098, 0.0, 0.0, 0.0, 1.0], "cav_1_lidar_pose": [-0.6782050132751465, -0.7342339754104614, -0.03063499927520752, -685.0469970703125, 0.734870970249176, -0.6776940226554871, -0.026362700387835503, 203.3769989013672, -0.0014047699514776468, -0.04039210081100464, 0.9991829991340637, 7.019320011138916, 0.0, 0.0, 0.0, 1.0], "outputs": "Yes, there is a car at [-20.0, -0.1], which is close to your planned future trajectory. Connected autonomous vehicle ego detects it."}

    # we have at most args.max_num_answer_objects output objects in gt answers
    #max_num_answer_objects = args.max_num_answer_objects

    if args.qa_type_id in [1, 2, 3, 4]:
      gt_answer_types = np.stack([classify_answer_type(d['conversations'][1]['value']) for d in data], axis=0)
      print('data[0]: ', data[0])
      gt_boxes = [parse_multiple_box_parameters(d['conversations'][1]['value'], args.simplified, args.max_num_answer_objects) for d in data]
      gt_boxes = np.stack(gt_boxes, axis=0)
      print('gt_boxes.shape: ', gt_boxes.shape)
      # (1723 samples/frames, 3 max_num_answer_objects, 3: [x, z, is_valid])

      output_answer_types = np.stack([classify_answer_type(d['outputs']) for d in data], axis=0)
      output_boxes = [parse_multiple_box_parameters(d['outputs'], args.simplified, args.max_num_answer_objects) for d in data]
      output_boxes = np.stack(output_boxes, axis=0)
      print('output_boxes.shape: ', output_boxes.shape)
      # (1723 samples/frames, 3 max_num_answer_objects, 3: [x, z, is_valid])

      evaluate_object_existance(gt_answer_types, output_answer_types)
      # Note that we can ignore evaluate_reason result for v2xreal
      evaluate_reason(gt_answer_types, output_answer_types)
      evaluate_multiple_box_accuracy(gt_answer_types, gt_boxes, output_answer_types, output_boxes, args.simplified, args.max_num_answer_objects)
      #select_and_sort_samples(args.selected_output_file, data, gt_answer_types, gt_boxes, output_answer_types, output_boxes)
      print('evaluate_multiple_box_accuracy finished')
    elif args.qa_type_id in [14]: # NQ4 to have a fair comparison to v2vllm
      # similart to above Q1~Q4, but need different parser
      print('data[0]: ', data[0])
      print('data[1]: ', data[1])
      gt_answer_types = np.stack([classify_answer_type(d['conversations'][1]['value']) for d in data], axis=0)
      print('gt_answer_types[0]: ', gt_answer_types[0])
      gt_boxes = [parse_multiple_box_parameters(d['conversations'][1]['value'].replace('(', '[').replace(')', ']'), args.simplified, args.max_num_answer_objects) for d in data]
      print('gt_boxes[0]: ', gt_boxes[0])
      print('gt_boxes[1]: ', gt_boxes[1])
      gt_boxes = np.stack(gt_boxes, axis=0)
      print('gt_boxes.shape: ', gt_boxes.shape)

      output_answer_types = np.stack([classify_answer_type(d['outputs']) for d in data], axis=0)
      output_boxes = [parse_multiple_box_parameters(d['outputs'].replace('(', '[').replace(')', ']'), args.simplified, args.max_num_answer_objects) for d in data]
      output_boxes = np.stack(output_boxes, axis=0)
      print('output_boxes.shape: ', output_boxes.shape)
      
      evaluate_object_existance(gt_answer_types, output_answer_types)
      evaluate_multiple_box_accuracy(gt_answer_types, gt_boxes, output_answer_types, output_boxes, args.simplified, args.max_num_answer_objects)
      
    elif args.qa_type_id in [5, 19]: # Q5, NQ9
      evaluate_future_trajectory(args.num_future_waypoints, data, args.npy_save_path, args.qa_type_id)  
      print('evaluate_future_trajectory finished')
    elif args.qa_type_id in [18]:  # NQ8
      evaluate_suggested_speed_steering(data, args.npy_save_path)
    #elif args.qa_type_id in [15, 14, 11, 12, 13, 17]:  # NQ5, NQ4, NQ1, NQ2, NQ3, NQ7
    elif args.qa_type_id in [15, 11, 12, 13, 17]:  # NQ5, NQ1, NQ2, NQ3, NQ7
    #elif args.qa_type_id in [15, 17]:  # NQ5, NQ7
      if args.qa_type_id in [15, 17]:
        match_threshold = 10000
      else:
        match_threshold = 10000
      evaluate_notable_objects_prediction(args.max_num_answer_objects, args.num_future_waypoints, data, args.npy_save_path, args.qa_type_id, match_threshold)
    elif args.qa_type_id in [16]: # NQ6
      evaluate_is_another_cav_notable_object(data)


    return


def parse_planned_future_trajectory(question, num_future_waypoints, prepend_current_position, current_position=np.array([[0, 0]]), return_has_parsing_error=False):
    if num_future_waypoints == 0:
      if return_has_parsing_error:  
        return None, False
      else:
        return None

    start_idx = question.find('[')
    end_idx = question.find(']')

    waypoint_str = question[start_idx+1: end_idx]
    waypoint_str = waypoint_str.replace('(', '[')
    waypoint_str = waypoint_str.replace(')', ']')

    planned_future_trajectory = parse_multiple_box_parameters(waypoint_str, True, num_future_waypoints)
    if np.any(planned_future_trajectory[:, 2] == 0):
      has_parsing_error = True
    else:  
      has_parsing_error = False
    planned_future_trajectory = planned_future_trajectory[:, :2]

    if prepend_current_position:
      planned_future_trajectory = np.concatenate([
        current_position,
        planned_future_trajectory
      ], axis=0)

    if not return_has_parsing_error:   
      return planned_future_trajectory
    else:
      return planned_future_trajectory, has_parsing_error

def parse_direction(question):
    '''
    One of ['front', 'front left', 'front right', 'back', 'back left', 'back right']
    '''
    direction = np.zeros(2)
    if 'front left' in question:
      direction = np.array([0.5, -0.5 * np.sqrt(3)])
    elif 'front right' in question:
      direction = np.array([0.5,  0.5 * np.sqrt(3)])
    elif 'front' in question:
      direction = np.array([1, 0])  
    elif 'back left' in question:
      direction = np.array([-0.5, -0.5 * np.sqrt(3)])
    elif 'back right' in question:
      direction = np.array([-0.5,  0.5 * np.sqrt(3)])
    elif 'back' in question:
      direction = np.array([-1, 0])  
    else:  
      assert False  
    return direction  


def transform_waypoints_from_1_to_ego(waypoints, data_sample):
    '''
    cav_ego_lidar_pose can be list of 16 for v2v4real or list of 6 for v2xreal
    '''
    cav_ego_lidar_pose = data_sample['cav_ego_lidar_pose'] 
    if len(cav_ego_lidar_pose) == 16:
      cav_ego_lidar_pose = np.array(cav_ego_lidar_pose).reshape([4, 4])

    if data_sample['asker_cav_id'] == 'ego':
      cav_1_lidar_pose = data_sample['cav_ego_lidar_pose'] 
    elif data_sample['asker_cav_id'] == '1':
      cav_1_lidar_pose = data_sample['cav_1_lidar_pose'] 
    elif data_sample['asker_cav_id'] == '2':
      cav_1_lidar_pose = data_sample['cav_2_lidar_pose'] 
    else:  
      print('assert False asker_cav_id: ', data_sample['asker_cav_id'])  
      assert False

    if len(cav_1_lidar_pose) == 16:
      cav_1_lidar_pose = np.array(cav_1_lidar_pose).reshape([4, 4])
    transformation_matrix = x1_to_x2(cav_1_lidar_pose, cav_ego_lidar_pose)

    # set z to 0, using zero-pad
    waypoints = np.pad(waypoints, [[0, 0], [0, 1]])
    waypoints = project_points_by_matrix_torch(waypoints, transformation_matrix)
    waypoints = waypoints[:, :2]

    return waypoints



def visualize_single_frame(args, data_sample, visualization_folder, npy_save_path, max_num_answer_objects, num_future_waypoints, next_data_sample=None):

    #print('data_sample: ', data_sample)
    #assert False
    # {"id": 795, "conversations": [{"from": "human", "value": "Is there anything I need to be aware of if my planned future trajectory is [(10.3,0.3),(20.7,0.6),(31.1,0.9)]?"}, {"from": "gpt", "value": "Yes, there are cars at [21.1, -2.4], [27.4, 3.6], [29.4, -2.1], which are close to your planned future trajectory. Both connected autonomous vehicles detect them."}], "scenario_index": 5, "local_timestamp_index": 162, "global_timestamp_index": 945, "qa_sub_type": [2, 1, 0], "distance_to_waypoint": [3.096679389157216, 2.8152530562163554, 2.958256396518313], "future_time": [19.0, 26.0, 27.0], "outputs": "Yes, there are cars at [26.4, 0.2], [26.5, -5.9], [31.0, -5.8], which are close to your planned future trajectory. Both connected autonomous vehicles detect them."}

    # v7sd
    # {"id": 0, "conversations": [{"from": "human", "value": "I am CAV_EGO. What is the suggested future trajectory to avoid collision with nearby objects?"}, {"from": "gpt", "value": "The suggested future trajectory is [(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]."}], "scenario_index": 0, "local_timestamp_index": 0, "global_timestamp_index": 0, "qa_sub_type": [7], "distance_to_waypoint": null, "future_time": null, "gpt_reasoning_output": " None of the connected autonomous vehicles detects anything near your planned future trajectory.", "future_trajectory_str_in_ego": "[(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]", "future_trajectory_str_in_self": "[(8.6,0.2),(17.2,0.5),(26.0,0.7),(34.7,0.8),(43.6,0.8),(52.6,0.6)]", "asker_cav_id": "ego", "cav_ego_lidar_pose": [-0.8603860139846802, -0.5096380114555359, 0.0020271099638193846, -747.4990234375, 0.5096420049667358, -0.8603760004043579, 0.004268390126526356, 246.41400146484375, -0.0004312550008762628, 0.004705559927970171, 0.9999889731407166, 4.646309852600098, 0.0, 0.0, 0.0, 1.0], "cav_1_lidar_pose": [-0.6782050132751465, -0.7342339754104614, -0.03063499927520752, -685.0469970703125, 0.734870970249176, -0.6776940226554871, -0.026362700387835503, 203.3769989013672, -0.0014047699514776468, -0.04039210081100464, 0.9991829991340637, 7.019320011138916, 0.0, 0.0, 0.0, 1.0], "outputs": "The suggested future trajectory is [(8.9,0.2),(17.9,0.3),(27.0,0.3),(36.4,0.1),(45.9,-0.3),(55.6,-0.9)]."}

    # v6sd
    # {"id": 0, "conversations": [{"from": "human", "value": "I am CAV_1. Is there anything I need to be aware of if my planned future trajectory is [(-68.0,3.4),(-59.9,1.4),(-51.8,0.5),(-43.9,0.6),(-35.5,0.4),(-26.8,0.3)]?"}, {"from": "gpt", "value": "Yes, there are cars at [-20.5, -0.1], [-21.1, 1.5], which are close to your planned future trajectory. Both connected autonomous vehicles detect them."}], "scenario_index": 0, "local_timestamp_index": 0, "global_timestamp_index": 0, "qa_sub_type": [1, 2], "distance_to_waypoint": [6.345685982198006, 5.8998676466399065], "future_time": [29.0, 29.0], "future_trajectory_str_in_ego": "[(-68.0,3.4),(-59.9,1.4),(-51.8,0.5),(-43.9,0.6),(-35.5,0.4),(-26.8,0.3)]", "future_trajectory_str_in_self": "[(7.9,0.5),(16.2,1.0),(24.2,2.5),(31.8,4.8),(39.9,7.0),(48.2,9.4)]", "asker_cav_id": "1", "cav_ego_lidar_pose": [-0.8603860139846802, -0.5096380114555359, 0.0020271099638193846, -747.4990234375, 0.5096420049667358, -0.8603760004043579, 0.004268390126526356, 246.41400146484375, -0.0004312550008762628, 0.004705559927970171, 0.9999889731407166, 4.646309852600098, 0.0, 0.0, 0.0, 1.0], "cav_1_lidar_pose": [-0.6782050132751465, -0.7342339754104614, -0.03063499927520752, -685.0469970703125, 0.734870970249176, -0.6776940226554871, -0.026362700387835503, 203.3769989013672, -0.0014047699514776468, -0.04039210081100464, 0.9991829991340637, 7.019320011138916, 0.0, 0.0, 0.0, 1.0], "outputs": "Yes, there is a car at [-20.0, -0.1], which is close to your planned future trajectory. Connected autonomous vehicle ego detects it."}

    global_timestamp_index = data_sample['global_timestamp_index']

    visualization_file = os.path.join(visualization_folder, '%04d_%d_visualization_waypoint.png' % (global_timestamp_index, data_sample['id']))

    if args.visualization_poster:
      colors = {
        # regular      
        'ego': 'darkgoldenrod',
        '1': 'blue',
        '2': 'blue',
        'gt_boxes': 'grey',
        'gt_answer_both': 'lime',
        'gt_future_waypoints': 'lime',
        'planned_future_trajectory': 'magenta',
        'reference_location': 'magenta',
        'direction': 'magenta',
        'model_output': 'red',
        'output_future_waypoints': 'red',
        'unified_point_cloud': 'lightgrey',
        'asker_cav_location': 'magenta',
      }
    else: # old v2vllm paper color
      colors = {
        'ego': 'yellow',
        '1': 'cyan',
        '2': 'cyan',
        'gt_boxes': 'white',
        'gt_answer_both': 'lime',
        'gt_future_waypoints': 'lime',
        'planned_future_trajectory': 'magenta',
        'reference_location': 'magenta',
        'direction': 'magenta',
        'model_output': 'red',
        'output_future_waypoints': 'red',
        'unified_point_cloud': 'white',
        'asker_cav_location': 'magenta',
      }

    alpha = 1.0

    markers = {
      'ego': "x",
      '1': "x",
      '2': "x",
      'gt_ego': 'o',
      'gt_1': 'o',
      'gt_2': 'o',
    }

    fig = plt.figure(figsize=(24, 8), facecolor='black')
    ax = fig.add_subplot(111)
    if args.visualization_poster:
      ax.set_facecolor('white')
    else:
      ax.set_facecolor('black')
    ax.set_xlim([-150, 150])
    ax.set_ylim([-50, 50])

    # load lidar point cloud
    projected_lidar_dict = dict()
    for cav_id in['ego', '1', '2', '-1', '-2']:
      try:  
        projected_lidar_file = os.path.join(npy_save_path, cav_id, '%04d_projected_lidar.npy' % global_timestamp_index)  
        projected_lidar_dict[cav_id] = np.load(projected_lidar_file)
      except FileNotFoundError:
        continue  
      # v2v4real coordinate
      # plot lidar point cloud
      if args.visualization_point_only:
        # draw only one cav's point cloud in diff colors, and end this visualization right after drawing point cloud  
        if cav_id == data_sample['asker_cav_id']:
          ax.scatter(projected_lidar_dict[cav_id][:, 0], projected_lidar_dict[cav_id][:, 1], s=0.1, c=colors[cav_id])

      else: # regular, draw both cav's point cloud in white
        ax.scatter(projected_lidar_dict[cav_id][:, 0], projected_lidar_dict[cav_id][:, 1], s=0.1, c=colors['unified_point_cloud'])
    if args.visualization_point_only:
      # end this visualization  
      plt.gca().set_aspect('equal')
      plt.gca().invert_xaxis()
      fig.savefig(visualization_file, dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
      return visualization_file

    # load all gt boxes
    gt_boxes_file = os.path.join(npy_save_path, '%04d_gt.npy' % global_timestamp_index)
    gt_boxes = np.load(gt_boxes_file)
    # The v2v4real coordinate system is:
    # Left-hand coordinate system,
    # x: forward, y: right, z: up
    # x: 2d visualization left, y: 2d visualization up
    # theta: rotation around z-axis
    # plot all gt boxes
    for i in range(gt_boxes.shape[0]):
      ax.plot(gt_boxes[i, :, 0], gt_boxes[i, :, 1], c=colors['gt_boxes'])


    # load detection result boxes
    det_boxes_dict = dict()
    for cav_id in['ego', '1', '2', '-1', '-2']:
      try:
        det_boxes_file = os.path.join(npy_save_path, cav_id, '%04d_pred.npy' % global_timestamp_index)
        det_boxes_dict[cav_id] = np.load(det_boxes_file)
      except FileNotFoundError:
        continue  
      # v2v4real coordinate
      for i in range(det_boxes_dict[cav_id].shape[0]):
        pass  
        #ax.plot(det_boxes_dict[cav_id][i, :, 0], det_boxes_dict[cav_id][i, :, 1], c=colors[cav_id])
    
    # plot different reference info from question depending on qa_type_id
    question = data_sample['conversations'][0]['value']
    if args.qa_type_id == 2 or args.qa_type_id == 1:
      assert(args.simplified)   
      # for q2 v4
      # reference object location
      reference_location = parse_box_parameters(question, args.simplified)
      reference_location = np.expand_dims(reference_location, axis=0)    
      ax.scatter(reference_location[:, 0], reference_location[:, 1], c=colors['reference_location'], s=100)

    elif args.qa_type_id == 3:
      # for q3 v5
      # reference object location
      sample_box_center_str = data_sample['sample_box_center_str']
      reference_location = parse_box_parameters(sample_box_center_str, args.simplified)
      reference_location = np.expand_dims(reference_location, axis=0)    
      ax.scatter(reference_location[:, 0], reference_location[:, 1], c=colors['reference_location'], s=100)

      # reference direction
      direction = parse_direction(question)
      # plot 10 meters line
      direction *= 10
      direction = np.stack([
        np.array([0, 0]),
        direction
      ], axis=0)
      #ax.plot(direction[:, 0], direction[:, 1], c=colors['direction'])
    elif args.qa_type_id == 4 or args.qa_type_id == 14:
      # for q4 v6
      num_future_waypoints = 3
      # planned future trajectory from question
      # for v6sd 
      num_future_waypoints = 6
      # for q4 v6sd, cav_1 also use traj in cav_ego's coordinate
      # so the prepend_current_position need cav_1's current position in cav_ego's coordinate
      # we can not just prepend [0,0]
      question = data_sample['conversations'][0]['value']
      marker = markers[data_sample['asker_cav_id']]
      if data_sample['asker_cav_id'] == 'ego':
        planned_future_trajectory = parse_planned_future_trajectory(question, num_future_waypoints, prepend_current_position=True) 
      else: # data_sample['asker_cav_id'] == '1':
        current_position_in_cav_1 = np.array([[0, 0]])
        current_position_in_cav_ego = transform_waypoints_from_1_to_ego(current_position_in_cav_1, data_sample)
        planned_future_trajectory = parse_planned_future_trajectory(question, num_future_waypoints, prepend_current_position=True, current_position=current_position_in_cav_ego) 
      ax.plot(planned_future_trajectory[:, 0], planned_future_trajectory[:, 1], c=colors['planned_future_trajectory'])

      if next_data_sample is not None:
        marker = markers[next_data_sample['asker_cav_id']]  
        question = next_data_sample['conversations'][0]['value']
        if next_data_sample['asker_cav_id'] == 'ego':
          planned_future_trajectory = parse_planned_future_trajectory(question, num_future_waypoints, prepend_current_position=True) 
        else: # data_sample['asker_cav_id'] == '1':
          current_position_in_cav_1 = np.array([[0, 0]])
          current_position_in_cav__ego = transform_waypoints_from_1_to_ego(current_position_in_cav_1, next_data_sample)
          planned_future_trajectory = parse_planned_future_trajectory(question, num_future_waypoints, prepend_current_position=True, current_position=current_position_in_cav__ego) 
        ax.plot(planned_future_trajectory[:, 0], planned_future_trajectory[:, 1], c=colors['planned_future_trajectory'])


    # plot gt and output future waypoint
    elif args.qa_type_id == 5 or args.qa_type_id == 19:
      # for q5 v7
      # gt future waypoints
      gt_answer = data_sample['conversations'][1]['value']  

      if args.qa_type_id == 5:
        current_position = np.array([[0, 0]])
      else:
        question = data_sample['conversations'][0]['value']
        question_temp = question.replace('(', '[')
        question_temp = question_temp.replace(')', ']')
        current_position = parse_box_parameters(question_temp, True)
        current_position = np.array([[current_position[0], current_position[1]]])


      gt_future_waypoints = parse_planned_future_trajectory(gt_answer, num_future_waypoints, prepend_current_position=True, current_position=current_position) 
      output_future_waypoints = parse_planned_future_trajectory(data_sample['outputs'], num_future_waypoints, prepend_current_position=True, current_position=current_position)

      if args.qa_type_id == 5 and data_sample['asker_cav_id'] == '1' or data_sample['asker_cav_id'] == '2':
        gt_future_waypoints = transform_waypoints_from_1_to_ego(gt_future_waypoints, data_sample)  
        output_future_waypoints = transform_waypoints_from_1_to_ego(output_future_waypoints, data_sample)

      ax.plot(gt_future_waypoints[:, 0], gt_future_waypoints[:, 1], c=colors['gt_future_waypoints'])
      # plot last waypoint
      gt_marker =  markers['gt_' + data_sample['asker_cav_id']]
      ax.scatter(gt_future_waypoints[-1:, 0], gt_future_waypoints[-1:, 1], c=colors['gt_answer_both'], s=100, marker=gt_marker, alpha=alpha)


      if not args.visualization_gt_only:
        #ax.plot(output_future_waypoints[:, 0], output_future_waypoints[:, 1], c=colors['output_future_waypoints'])
        ax.plot(output_future_waypoints[:, 0], output_future_waypoints[:, 1], c=colors[data_sample['asker_cav_id']])
        # plot last waypoint
        marker = markers[data_sample['asker_cav_id']]
        ax.scatter(output_future_waypoints[-1:, 0], output_future_waypoints[-1:, 1], c=colors[data_sample['asker_cav_id']], s=1000, marker=marker, alpha=alpha)

      if next_data_sample is not None:
        assert(next_data_sample['asker_cav_id'] == '1')
        gt_answer = next_data_sample['conversations'][1]['value']

        if args.qa_type_id == 5:
          current_position = np.array([[0, 0]])
        else:
          question = next_data_sample['conversations'][0]['value']
          question_temp = question.replace('(', '[')
          question_temp = question_temp.replace(')', ']')
          current_position = parse_box_parameters(question_temp, True)
          current_position = np.array([[current_position[0], current_position[1]]])

        gt_future_waypoints = parse_planned_future_trajectory(gt_answer, num_future_waypoints, prepend_current_position=True, current_position=current_position)
        output_future_waypoints = parse_planned_future_trajectory(next_data_sample['outputs'], num_future_waypoints, prepend_current_position=True, current_position=current_position)

        if args.qa_type_id == 5 and next_data_sample['asker_cav_id'] == '1':
          gt_future_waypoints = transform_waypoints_from_1_to_ego(gt_future_waypoints, next_data_sample)
          output_future_waypoints = transform_waypoints_from_1_to_ego(output_future_waypoints, next_data_sample)

        ax.plot(gt_future_waypoints[:, 0], gt_future_waypoints[:, 1], c=colors['gt_future_waypoints'])
        # plot last waypoint
        gt_marker =  markers['gt_' + next_data_sample['asker_cav_id']]
        ax.scatter(gt_future_waypoints[-1:, 0], gt_future_waypoints[-1:, 1], c=colors['gt_answer_both'], s=100, marker=gt_marker, alpha=alpha)

        if not args.visualization_gt_only:
          #ax.plot(output_future_waypoints[:, 0], output_future_waypoints[:, 1], c=colors['output_future_waypoints'])
          ax.plot(output_future_waypoints[:, 0], output_future_waypoints[:, 1], c=colors[next_data_sample['asker_cav_id']])
          # plot last waypoint
          marker = markers[next_data_sample['asker_cav_id']]
          ax.scatter(output_future_waypoints[-1:, 0], output_future_waypoints[-1:, 1], c=colors[next_data_sample['asker_cav_id']], s=1000, marker=marker, alpha=alpha)

    else:
      print("Not implemented.")  
      #assert False


    # plot object in answer: q1, q2, q3, q4, nq4
    if args.qa_type_id in [1, 2, 3, 4, 14]:
      
      # for q1, q2, q3, set asker_cav_id to ego
      if 'asker_cav_id' not in data_sample:
        data_sample['asker_cav_id'] = 'ego'  

      marker = markers[data_sample['asker_cav_id']]
      gt_marker =  markers['gt_' + data_sample['asker_cav_id']]

      # gt notable object in gt answer
      gt_answer_str = data_sample['conversations'][1]['value']
      if args.qa_type_id in [1, 2, 3, 4]:
        gt_answer = parse_multiple_box_parameters(gt_answer_str, True, max_num_answer_objects, skip_dummy_boxes=True)
      else: # nq1, nq2, nq3, nq4
        # workaround, replace '(' with '[', ')' with ']'
        gt_answer_str_temp = gt_answer_str.replace('(', '[')
        gt_answer_str_temp = gt_answer_str_temp.replace(')', ']')
        gt_answer = parse_multiple_box_parameters(gt_answer_str_temp, True, max_num_answer_objects, skip_dummy_boxes=True)
      if gt_answer is not None:
        # if visualize merge, use different gt color for diff cav gt 
        if args.visualization_double == 'merge':
          ax.scatter(gt_answer[:, 0], gt_answer[:, 1], c=colors[data_sample['asker_cav_id']], s=100, marker=gt_marker, alpha=alpha)
        else: # use green as gt color 
          ax.scatter(gt_answer[:, 0], gt_answer[:, 1], c=colors['gt_answer_both'], s=100, marker=gt_marker, alpha=alpha)

      gt_answer_current_frame = gt_answer.copy() if gt_answer is not None else None
      # llm output
      output_str = data_sample['outputs']
      if args.qa_type_id in [1, 2, 3, 4]:
        output = parse_multiple_box_parameters(output_str, True, max_num_answer_objects, skip_dummy_boxes=True)
      else: # nq1234
        output_str_temp = output_str.replace('(', '[')
        output_str_temp = output_str_temp.replace(')', ']')
        output = parse_multiple_box_parameters(output_str_temp, True, max_num_answer_objects, skip_dummy_boxes=True)
      if output is not None:
        if not args.visualization_gt_only:  
          # if visualize merge or all, use different color for diff cav output  
          if args.visualization_double == 'merge' or args.visualization_double == 'all':  
            ax.scatter(output[:, 0], output[:, 1], c=colors[data_sample['asker_cav_id']], s=1000, marker=marker, alpha=alpha)
          else:  # use red for output 
            ax.scatter(output[:, 0], output[:, 1], c=colors['model_output'], s=1000, marker=marker, alpha=alpha)
        
      # q4 support both cavs  
      if next_data_sample is not None:  
        marker = markers[next_data_sample['asker_cav_id']]
        gt_marker =  markers['gt_' + next_data_sample['asker_cav_id']]
        # gt notable object in gt answer
        gt_answer_str = next_data_sample['conversations'][1]['value']
        gt_answer = parse_multiple_box_parameters(gt_answer_str, True, max_num_answer_objects, skip_dummy_boxes=True)

        if gt_answer_current_frame is None:
          gt_answer_new = gt_answer
          if gt_answer_new is not None:
            ax.scatter(gt_answer_new[:, 0], gt_answer_new[:, 1], c=colors[next_data_sample['asker_cav_id']], s=100, marker=gt_marker, alpha=alpha)

        elif gt_answer is not None: # check whether each gt is notable objects for both cavs or just the next_data_sample's aasker cav
          gt_answer_new = []
          gt_answer_both = []
          for gt_answer_id in range(gt_answer.shape[0]):
            diff = gt_answer[gt_answer_id:gt_answer_id+1, :] - gt_answer_current_frame
            diff = np.linalg.norm(diff, axis=1)
            if np.any(diff < 1e-5):
              gt_answer_both.append(gt_answer[gt_answer_id])  
            else:  
              gt_answer_new.append(gt_answer[gt_answer_id])  

          if len(gt_answer_new) > 0:    
            gt_answer_new = np.stack(gt_answer_new)    
            ax.scatter(gt_answer_new[:, 0], gt_answer_new[:, 1], c=colors[next_data_sample['asker_cav_id']], s=100, marker=gt_marker, alpha=alpha)

          if len(gt_answer_both) > 0:
            gt_answer_both = np.stack(gt_answer_both)
            ax.scatter(gt_answer_both[:, 0], gt_answer_both[:, 1], c=colors['gt_answer_both'], s=100, marker=gt_marker, alpha=alpha)

        # llm output
        output_str = next_data_sample['outputs']
        output = parse_multiple_box_parameters(output_str, True, max_num_answer_objects, skip_dummy_boxes=True)
        if output is not None:
          if not args.visualization_gt_only:  
            ax.scatter(output[:, 0], output[:, 1], c=colors[next_data_sample['asker_cav_id']], s=1000, marker=marker, alpha=alpha)  


    # prediction: nq7, notable object current location and future waypoint
    if args.qa_type_id in [17]: 

      marker = markers[data_sample['asker_cav_id']]
      gt_marker =  markers['gt_' + data_sample['asker_cav_id']]

      # gt notable object location
      gt_answer_str = data_sample['conversations'][1]['value']
      gts_location, gts_action, gts_future_waypoints, gts_has_parsing_location_error, gts_has_parsing_action_error, gts_has_parsing_trajectory_error = parse_notable_object_prediction_answer(args.qa_type_id, gt_answer_str, args.max_num_answer_objects, args.num_future_waypoints)
      # plot current location
      marker = markers[data_sample['asker_cav_id']]
      gt_marker =  markers['gt_' + data_sample['asker_cav_id']]
      if gts_location is not None:
        # if visualize merge, use different gt color for diff cav gt 
        if args.visualization_double == 'merge':
          ax.scatter(gts_location[:, 0], gts_location[:, 1], c=colors[data_sample['asker_cav_id']], s=100, marker=gt_marker, alpha=alpha)
        else: # use green as gt color 
          ax.scatter(gts_location[:, 0], gts_location[:, 1], c=colors['gt_answer_both'], s=100, marker=gt_marker, alpha=alpha)
      # gt all waypoints
      # [num_objects, num_waypoints, num_dims=2]
      gt_all_waypoints = np.stack([
        gts_location, 
        gts_future_waypoints[:, 0, :]
      ], axis=1)
      for i in range(gt_all_waypoints.shape[0]):
        ax.plot(gt_all_waypoints[i, :, 0], gt_all_waypoints[i, :, 1], c=colors['gt_future_waypoints'])
        # plot last waypoint
        ax.scatter(gt_all_waypoints[i, -1:, 0], gt_all_waypoints[i, -1:, 1], c=colors['gt_answer_both'], s=100, marker=gt_marker, alpha=alpha)
     

      
      # output notable object location
      output_str = data_sample['outputs']
      outputs_location, outputs_action, outputs_future_waypoints, outputs_has_parsing_location_error, outputs_has_parsing_action_error, outputs_has_parsing_trajectory_error = parse_notable_object_prediction_answer(args.qa_type_id, output_str, args.max_num_answer_objects, args.num_future_waypoints)
      if outputs_location is not None:
        if not args.visualization_gt_only:
          # if visualize merge or all, use different color for diff cav output  
          if args.visualization_double == 'merge' or args.visualization_double == 'all':
            ax.scatter(outputs_location[:, 0], outputs_location[:, 1], c=colors[data_sample['asker_cav_id']], s=1000, marker=marker, alpha=alpha)
          else:  # use red for output 
            ax.scatter(outputs_location[:, 0], outputs_location[:, 1], c=colors['model_output'], s=1000, marker=marker, alpha=alpha)
      # output all waypoints
      output_all_waypoints =  np.stack([
        outputs_location,
        outputs_future_waypoints[:, 0, :]
      ], axis=1)
      for i in range(output_all_waypoints.shape[0]):
        ax.plot(output_all_waypoints[i, :, 0], output_all_waypoints[i, :, 1], c=colors[data_sample['asker_cav_id']])
        # plot last waypoint
        gt_marker =  markers['gt_' + data_sample['asker_cav_id']]
        ax.scatter(output_all_waypoints[i, -1:, 0], output_all_waypoints[i, -1:, 1], c=colors[data_sample['asker_cav_id']], s=1000, marker=marker, alpha=alpha)

      

    # plot asker cav location
    if args.qa_type_id in [14, 17, 19]:
        question = data_sample['conversations'][0]['value']
        question_temp = question.replace('(', '[')
        question_temp = question_temp.replace(')', ']')
        current_position = parse_box_parameters(question_temp, True)
        current_position = np.array([[current_position[0], current_position[1]]])
  
        marker = markers[data_sample['asker_cav_id']]
        ax.scatter(current_position[:, 0], current_position[:, 1], c=colors['asker_cav_location'], s=1000, marker=marker, alpha=alpha)
      

        if next_data_sample is not None:
          question = next_data_sample['conversations'][0]['value']
          question_temp = question.replace('(', '[')
          question_temp = question_temp.replace(')', ']')
          current_position = parse_box_parameters(question_temp, True)
          current_position = np.array([[current_position[0], current_position[1]]])

          marker = markers[next_data_sample['asker_cav_id']]
          ax.scatter(current_position[:, 0], current_position[:, 1], c=colors['asker_cav_location'], s=1000, marker=marker, alpha=alpha)


    plt.gca().set_aspect('equal')
    plt.gca().invert_xaxis()
    #plt.axis('off')
    #print('visualization_file: ', visualization_file)
    fig.savefig(visualization_file, dpi=fig.dpi, bbox_inches='tight', pad_inches = 0)
    return visualization_file


def generate_single_video(image_list, video_file):
    frame = cv2.imread(image_list[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_file, fourcc, 10, (width, height))
    for image in image_list:
      video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    return

def visualize(args):
    print('args.answers_file: ', args.answers_file)
    with open(args.answers_file) as f:
      data = [json.loads(line) for line in f]

    end_idx = args.answers_file.rfind('/')
    visualization_folder = os.path.join(args.answers_file[:end_idx], 'visualization', args.visualization_output_folder)
    mkdir_if_missing(visualization_folder)

    image_list = []
    #for data_sample_id in range(len(data)):   

    # v2v-llm-graph sample
    #for data_sample_id in [1320]:

    # nq4
    #for data_sample_id in [1758]:
    # nq7
    #for data_sample_id in [759]:
    #for data_sample_id in [759, 758, 757, 756]:
    # nq9
    for data_sample_id in [801]:

    # merge 800 and 801
    #for data_sample_id in [800]:

    # video1
    #for data_sample_id in range(1572, 1824+1):

    # video2
    #for data_sample_id in range(838, 964+1):

    # video3
    #for data_sample_id in range(2860, 3040+1):

    # video4
    #for data_sample_id in range(584, 628+1):
    

    #for data_sample_id in [0]:    
    #for data_sample_id in range(10):    
    #for data_sample_id in [3442, 3444]:    
    # q1
    # attfuse, object_only
    #for data_sample_id in [61635]:    
    # v2vllm
    #for data_sample_id in [56412]:    
    # cobevt
    #for data_sample_id in [58485]:    
    # ego_only, early    
    #for data_sample_id in [56017]:    
    # v2xvit
    #for data_sample_id in [48827]:    
    # scene_only
    #for data_sample_id in [54439]:    
    # q2
    #for data_sample_id in [9197]:    
    # q3
    #for data_sample_id in [2124, 1799]:    
    # q4
    #for data_sample_id in [182, 1297]:    
    # q5
    #for data_sample_id in [1408, 1409, 2335, 942]:    
    # v2xreal
    # q5
    #for data_sample_id in [1116, 131]:    
    # q4
    #for data_sample_id in [833, 1263]:    
    # q3
    #for data_sample_id in [734]:    
    # q2
    #for data_sample_id in range(16477, 22152):    
    #for data_sample_id in [17582]:    
    # q1    
    #for data_sample_id in [118312-1, 118574-1, 123564]:    
    #for data_sample_id in [118574-1]:    
    #for data_sample_id in [118576-1]:    
    #for data_sample_id in [116965-1]:    
    #for data_sample_id in [121792-1]:    
    #for data_sample_id in [118574-1]:    
    # nq4, nq7
    #for data_sample_id in [181, 1296]:    

      print('data_sample_id: ', data_sample_id)  
      if args.visualization_double in ['all', 'even', 'odd']:
        if args.visualization_double == 'even':
          if data_sample_id % 2 == 1:
            continue  
        if args.visualization_double == 'odd':
          if data_sample_id % 2 == 0:
            continue  
        data_sample = data[data_sample_id]  


        visualization_file = visualize_single_frame(args, data_sample, visualization_folder, args.npy_save_path, args.max_num_answer_objects, args.num_future_waypoints)
        print('visualization_file: ', visualization_file)
        image_list.append(visualization_file)
      else: # visualiza_double, merge CAV_EGO and CAV_1's output in a single frame 
        assert(args.visualization_double == 'merge')  
        if data_sample_id % 2 == 0:  
          # current sample is CAV_EGO, visualize current and next samples' output
          data_sample = data[data_sample_id]  
          next_data_sample = data[data_sample_id+1]  
          visualization_file = visualize_single_frame(args, data_sample, visualization_folder, args.npy_save_path, args.max_num_answer_objects, args.num_future_waypoints, next_data_sample)
          image_list.append(visualization_file)



    # generate video
    #video_file = os.path.join(visualization_folder, 'visualization_all.mp4')
    #video_file = os.path.join(visualization_folder, 'visualization_video1.mp4')
    #video_file = os.path.join(visualization_folder, 'visualization_video2.mp4')
    #video_file = os.path.join(visualization_folder, 'visualization_video3.mp4')
    video_file = os.path.join(visualization_folder, 'visualization_video4.mp4')
    generate_single_video(image_list, video_file)

    print('Video generation complete.')
    return


def plot_histogram(values, bins=10, title="Histogram", xlabel="Values", ylabel="Frequency", color='blue', filename="histogram.png"):
    """
    Draws a histogram for a given list of values and saves it to a file.

    Parameters:
        values (list): The data to plot.
        bins (int): Number of bins in the histogram.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color (str): Color of the bars.
        filename (str): The name of the file to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins, color=color, edgecolor='black', alpha=0.7, density=True)
    #plt.title(title, fontsize=16)
    #plt.xlabel(xlabel, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename, dpi=300)  # Save the plot with high resolution
    plt.close()  # Close the plot to free memory


def plot_scatter(x_values, y_values, title="2D coordinate", xlabel="X", ylabel="Y", color='blue', filename="scatter.png"):
    """
    Draws a histogram for a given list of values and saves it to a file.

    Parameters:
        x_values (list): The data to plot.
        y_values (list): The data to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color (str): Color of the bars.
        filename (str): The name of the file to save the plot.
    """
    plt.scatter(x_values, y_values, s=1)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    #plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename, dpi=300)  # Save the plot with high resolution
    plt.close()  # Close the plot to free memory


def get_v2vqa_gt_data_for_stats(args, v2v_dataset_file_dict, v2x_dataset_file_dict):
    
    valid_gt_boxes = []

    v2v_train_dataset_file = os.path.join('../DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy/co_llm/', 'v2v4real_3d_grounding_qa_dataset_' + v2v_dataset_file_dict[args.qa_type_id] + '.json')
    v2v_test_dataset_file = os.path.join('../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/', 'v2v4real_3d_grounding_qa_dataset_' + v2v_dataset_file_dict[args.qa_type_id] + '.json')

    v2x_train_dataset_file = os.path.join('../V2X-Real/my_models/train_no_fusion_keep_all/npy/co_llm/', 'v2xreal_3d_grounding_qa_dataset_' + v2x_dataset_file_dict[args.qa_type_id] + '.json')
    v2x_test_dataset_file = os.path.join('../V2X-Real/my_models/no_fusion_keep_all/npy/co_llm/', 'v2xreal_3d_grounding_qa_dataset_' + v2x_dataset_file_dict[args.qa_type_id] + '.json')


    if 'v2xreal' in args.answers_file:
      train_dataset_file = v2x_train_dataset_file
      test_dataset_file = v2x_test_dataset_file
    else:  
      train_dataset_file = v2v_train_dataset_file
      test_dataset_file = v2v_test_dataset_file


    for dataset_file in [train_dataset_file, test_dataset_file]:

      print('dataset_file: ', dataset_file)
      with open(dataset_file) as f:
        data = json.load(f)
      print('len(data): ', len(data))
      print('args.max_num_answer_objects: ', args.max_num_answer_objects)


      # For q1 ~ q4, get the gt_boxes location in gt answers
      if args.qa_type_id in [1, 2, 3, 4]:

        gt_answer_types = np.stack([classify_answer_type(d['conversations'][1]['value']) for d in data], axis=0)
        # qa_pair/frame wise gt_object_exist 
        gt_object_exist = gt_answer_types[:,1]
        #print('gt_object_exist[:5]: ', gt_object_exist[:5])
        num_positive = np.sum(gt_object_exist)
        print('num_positive: ', num_positive)
        num_negative = gt_object_exist.shape[0] - num_positive
        print('num_negative: ', num_negative)
        # q1 test
        # num_positive:  76522
        # num_negative:  44861

        gt_boxes = [parse_multiple_box_parameters(d['conversations'][1]['value'], args.simplified, args.max_num_answer_objects) for d in data]
        gt_boxes = np.stack(gt_boxes, axis=0)
        #print('gt_boxes[:5]: ', gt_boxes[:5])
        print('gt_boxes.shape: ', gt_boxes.shape)    
        # (121383, 1, 3)
        # (num_qa_pairs, max_num_answer_objects, [x, y, is_valid])
        # q4 v6sdn
        # (12290, 3, 3)

        # only for Q4
        if args.qa_type_id == 4:
          cav_1_position_in_cav_ego = [transform_waypoints_from_1_to_ego(np.array([[0, 0]]), d) for d in data]
          print('cav_1_position_in_cav_ego[:5]: ', cav_1_position_in_cav_ego[:5])
          # list of data sample, np.array (1, 2)
          # if asker is CAV_1, shift gt_boxes (x, y) so that we cn get relative (x, y) to asker CAV, to get stats
          for i in range(gt_boxes.shape[0]):
              if  'asker_cav_id' in data[i] and (data[i]['asker_cav_id'] == '1' or data[i]['asker_cav_id'] == '2'):
                  gt_boxes[i][:, :2] -= cav_1_position_in_cav_ego[i][0]

        # only keep the valid gt_boxes
        for i in range(gt_boxes.shape[0]):
            for j in range(gt_boxes.shape[1]):
                if gt_boxes[i][j][2] > 0:
                    valid_gt_boxes.append(gt_boxes[i][j][:2])

      elif args.qa_type_id == 5:
        gt_future_waypoints = [parse_planned_future_trajectory(d['conversations'][1]['value'], args.num_future_waypoints, prepend_current_position=True) for d in data] 
        print('gt_future_waypoints[:5]: ', gt_future_waypoints[:5])
        # (num_qa_pairs, 7, 2)

        if not args.behavior:
          # add last waypoint to calculate data stats
          for i in range(len(gt_future_waypoints)):
            valid_gt_boxes.append(gt_future_waypoints[i][-1])
        else: 
          # add mean of diff between consecutive waypoints
          for i in range(len(gt_future_waypoints)):
            num_waypoints = 6  
            diff = gt_future_waypoints[i][1:7] - gt_future_waypoints[i][0:6]
            #print('diff: ', diff)
            #print('diff.shape: ', diff.shape)
            # (6, 2)
            diff = np.mean(diff, axis=0)
            #print('diff.shape: ', diff.shape)
            # (2, )
            valid_gt_boxes.append(diff)
            #assert False


    # end of for loop over all qa pairs
    return valid_gt_boxes




def get_stats(args):
    v2v_dataset_file_dict = {
      1: 'v2s',
      2: 'v4bs',
      3: 'v5bs',
      4: 'v6sm3doublenew',
      5: 'v7sm100w6double',
    }

    v2x_dataset_file_dict = {
      1: 'v2s',
      2: 'v4bs',
      3: 'v5bs',
      4: 'v6sm3doublenew',
      5: 'v7sm3w6double',
    }
    

    if 'nuscenes_planning' in args.answers_file: # rebuttal
      print('get stats for nuscenes planning')
      valid_gt_boxes = []

      # nuscene val
      gt_traj = open('./stp3_data/stp3_val/stp3_traj_gt.pkl','rb')
      gt_traj_traj = pickle.load(gt_traj)
      token = open('./stp3_data/stp3_val/filter_token.pkl','rb')
      token_filter = pickle.load(token)
      for token in token_filter:
        gt_trajectory = gt_traj_traj[token]['gt_trajectory']  
        valid_gt_boxes.append(np.array([gt_trajectory[-1][1], gt_trajectory[-1][0]]))


    else: # regular code path for v2vqa stats
      # for q1 ~ q4, it represent answer object's center location
      # for q5, it represents the cav gt future ending waypoint coordinates
      # for q5 behavior, it represents the mean diff of consecutive cav gt future waypoint
      valid_gt_boxes = get_v2vqa_gt_data_for_stats(args, v2v_dataset_file_dict, v2x_dataset_file_dict)


    valid_gt_boxes = np.stack(valid_gt_boxes, axis=0)            
    print('valid_gt_boxes.shape: ', valid_gt_boxes.shape)
    # stp3 nuscenes val (4819, 2) 
    # total include train and test
    # q1 test
    # (76522, 2)
    print('np.min(valid_gt_boxes[:, 0]): ', np.min(valid_gt_boxes[:, 0]))
    print('np.max(valid_gt_boxes[:, 0]): ', np.max(valid_gt_boxes[:, 0]))
    print('np.std(valid_gt_boxes[:, 0]): ', np.std(valid_gt_boxes[:, 0]))
    print('np.min(valid_gt_boxes[:, 1]): ', np.min(valid_gt_boxes[:, 1]))
    print('np.max(valid_gt_boxes[:, 1]): ', np.max(valid_gt_boxes[:, 1]))
    print('np.std(valid_gt_boxes[:, 1]): ', np.std(valid_gt_boxes[:, 1]))
    # v2vqa
    # x: -100 ~ 100
    # y: -40 ~ 40
    # nuscene
    # x: -1 ~ 40
    # y: -10 ~ 11

    angle = np.arctan2(valid_gt_boxes[:, 1], valid_gt_boxes[:, 0])
    #print('angle[:5]: ', angle[:5])
    angle *= 180 / np.pi
    # - pi ~ pi
    print('np.min(angle): ', np.min(angle))
    print('np.max(angle): ', np.max(angle))
    print('np.std(angle): ', np.std(angle))



    dist = np.linalg.norm(valid_gt_boxes, axis=1)
    #print('dist[:5]: ', dist[:5])
    print('np.min(dist): ', np.min(dist))
    print('np.max(dist): ', np.max(dist))
    print('np.std(dist): ', np.std(dist))
    # v2vqa: dist: 0 ~ 102.23
    # nuscenes : dist: 0 ~ 40

    stats_output_base_folder_name = os.path.join('./stats')
    if args.behavior:
      stats_output_base_folder_name = os.path.join('./stats', 'behavior')

    stats_output_base_filename = os.path.join(stats_output_base_folder_name, 'q' + str(args.qa_type_id))
    if 'v2x' in args.answers_file:
      stats_output_base_filename = os.path.join(stats_output_base_folder_name, 'v2xreal', 'q' + str(args.qa_type_id))
    if 'nuscenes_planning' in args.answers_file:
      stats_output_base_filename += 'nuscenes'

    print('stats_output_base_filename: ', stats_output_base_filename)


    stats_output_filename = stats_output_base_filename + '_x' + '.png'
    print('stats_output_filename: ', stats_output_filename)
    plot_histogram(valid_gt_boxes[:, 0], bins=10, title="Sample Histogram", xlabel="x (meters)", ylabel="probability density", color='#f9cb9c', filename=stats_output_filename)

    stats_output_filename = stats_output_base_filename + '_y' + '.png'
    print('stats_output_filename: ', stats_output_filename)
    plot_histogram(valid_gt_boxes[:, 1], bins=10, title="Sample Histogram", xlabel="y (meters)", ylabel="probability density", color='#ffe599', filename=stats_output_filename)

    stats_output_filename = stats_output_base_filename + '_dist' + '.png'
    print('stats_output_filename: ', stats_output_filename)
    plot_histogram(dist, bins=10, title="Sample Histogram", xlabel="distance (meters)", ylabel="probability density", color='#b6d7a8', filename=stats_output_filename)

    stats_output_filename = stats_output_base_filename + '_angle' + '.png'
    print('stats_output_filename: ', stats_output_filename)
    plot_histogram(angle, bins=10, title="Sample Histogram", xlabel="angle (degrees)", ylabel="probability density", color='#9fc5e8', filename=stats_output_filename)


    if args.behavior:
      stats_output_filename = stats_output_base_filename + '_trajectory_diff' + '.png'
      plot_scatter(valid_gt_boxes[:,0], valid_gt_boxes[:,1], title='Trajectory Diff', filename=stats_output_filename)
      stats_output_filename = stats_output_base_filename + '_trajectory_diff_dist_angle' + '.png'
      plot_scatter(dist, angle, title='Trajectory Diff', xlabel='Dist', ylabel='Angle', filename=stats_output_filename)


    print('Get dataset stats complete.')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--eval-result-file", type=str, default="result.txt")
    parser.add_argument("--selected-output-file", type=str, default="selected_output.jsonl")
    parser.add_argument("--simplified", action="store_true")
    parser.add_argument("--multiple-output", action="store_true")
    parser.add_argument("--max-num-answer-objects", type=int, default=3)
    parser.add_argument("--visualization-only", action="store_true")
    parser.add_argument("--visualization-double", type=str, default="all") # all, even, odd, merge
    parser.add_argument("--visualization-output-folder", type=str, default="visualiation") # 
    parser.add_argument("--visualization-gt-only", action="store_true")
    parser.add_argument("--visualization-point-only", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--qa-type-id", type=int, default=0)
    parser.add_argument("--num-future-waypoints", type=int, default=6)
    parser.add_argument("--npy-save-path", type=str, default="../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy")
    parser.add_argument("--behavior", action="store_true")
    parser.add_argument("--visualization-poster", action="store_true")
    args = parser.parse_args()

    if args.visualization_only:
      if 'gt_only' in args.visualization_output_folder:
        args.visualization_gt_only = True
      if 'poster' in args.visualization_output_folder:
        args.visualization_poster = True 
      if 'point_only' in args.visualization_output_folder:
        args.visualization_point_only = True

      visualize(args)  

    elif args.stats_only:
      get_stats(args)  


    else:
      if not args.multiple_output:
        eval_model_v2v4real_3d_grounding_v2(args)
      else:  
        eval_model_v2v4real_3d_grounding_v6(args)

