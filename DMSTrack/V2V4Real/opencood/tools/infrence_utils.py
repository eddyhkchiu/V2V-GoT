import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy


def inference_no_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.
    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset
    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        if cav_id == 'ego':
            output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)



    pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor


def inference_no_fusion_keep_all(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_dict, pred_score_dict, gt_box_tensor, gt_object_id_tensor, pred_feature_dict, pred_early_feature_dict,  spatial_features_dict, spatial_features_2d_dict, regression_map_dict, classification_map_dict, visible_gt_object_ids_dict, invisible_gt_object_ids_dict = \
        dataset.post_process(batch_data,
                             output_dict,
                             return_in_dict=True)

    return pred_box_dict, pred_score_dict, gt_box_tensor, gt_object_id_tensor, pred_feature_dict, pred_early_feature_dict, spatial_features_dict, spatial_features_2d_dict, regression_map_dict, classification_map_dict, visible_gt_object_ids_dict, invisible_gt_object_ids_dict


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    # MY_DEBUG
    # instead of calling post_process(), 
    # calling my new code post_process_return_in_dict()
    # to make return variables stored in dict['ego']
    #print('inference_early_fusion')
    #pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor = \
    #    dataset.post_process(batch_data,
    #                         output_dict)

    #return pred_box_tensor, pred_score, gt_box_tensor, gt_object_id_tensor

    pred_box_dict, pred_score_dict, gt_box_tensor, gt_object_id_tensor, pred_feature_dict, pred_early_feature_dict, spatial_features_dict, spatial_features_2d_dict, regression_map_dict, classification_map_dict = \
        dataset.post_process(batch_data,
                             output_dict,
                             return_in_dict=True)

    return pred_box_dict, pred_score_dict, gt_box_tensor, gt_object_id_tensor, pred_feature_dict, pred_early_feature_dict, spatial_features_dict, spatial_features_2d_dict, regression_map_dict, classification_map_dict


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset)


# Temp only save projected_lidar
def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path, co_llm_save_path,
      pred_score_tensor=None, gt_object_id_tensor=None, pred_feature=None,
      pred_early_feature=None, transformation_matrix=None,
      spatial_features=None, spatial_features_2d=None,
      regression_map=None, classification_map=None,
      projected_lidar=None, lidar_pose=None):

    if projected_lidar is not None:
      projected_lidar_np = torch_tensor_to_numpy(projected_lidar)
      np.save(os.path.join(save_path, '%04d_projected_lidar.npy' % timestamp), projected_lidar_np)
    return



def save_prediction_gt_temp(pred_tensor, gt_tensor, pcd, timestamp, save_path, co_llm_save_path, 
      pred_score_tensor=None, gt_object_id_tensor=None, pred_feature=None,
      pred_early_feature=None, transformation_matrix=None,
      spatial_features=None, spatial_features_2d=None,
      regression_map=None, classification_map=None,
      projected_lidar=None, lidar_pose=None):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)

    if pred_score_tensor is not None:
      pred_score_np = torch_tensor_to_numpy(pred_score_tensor)
      np.save(os.path.join(save_path, '%04d_pred_score.npy' % timestamp), pred_score_np)

    if gt_object_id_tensor is not None:
      gt_object_id_np = torch_tensor_to_numpy(gt_object_id_tensor)
      np.save(os.path.join(save_path, '%04d_gt_object_id.npy' % timestamp), gt_object_id_np)

    if pred_feature is not None:
      feature_np = torch_tensor_to_numpy(pred_feature)
      np.save(os.path.join(save_path, '%04d_feature.npy' % timestamp), feature_np)

    if pred_early_feature is not None:
      early_feature_np = torch_tensor_to_numpy(pred_early_feature)
      np.save(os.path.join(save_path, '%04d_early_feature.npy' % timestamp), early_feature_np)

    if transformation_matrix is not None:
      transformation_matrix_np = torch_tensor_to_numpy(transformation_matrix)
      np.save(os.path.join(save_path, '%04d_transformation_matrix.npy' % timestamp), transformation_matrix_np)
    
    if projected_lidar is not None:
      projected_lidar_np = torch_tensor_to_numpy(projected_lidar)
      np.save(os.path.join(save_path, '%04d_projected_lidar.npy' % timestamp), projected_lidar_np)

    if lidar_pose is not None:
      lidar_pose_np = torch_tensor_to_numpy(lidar_pose)
      np.save(os.path.join(save_path, '%04d_lidar_pose.npy' % timestamp), lidar_pose_np)


    # LLM
    if spatial_features is not None:
      spatial_features_np = torch_tensor_to_numpy(spatial_features)
      np.save(os.path.join(co_llm_save_path, '%04d_spatial_features.npy' % timestamp), spatial_features_np)

    if spatial_features_2d is not None:
      spatial_features_2d_np = torch_tensor_to_numpy(spatial_features_2d)
      np.save(os.path.join(co_llm_save_path, '%04d_spatial_features_2d.npy' % timestamp), spatial_features_2d_np)

    if regression_map is not None:
      regression_map_np = torch_tensor_to_numpy(regression_map)
      np.save(os.path.join(co_llm_save_path, '%04d_regression_map.npy' % timestamp), regression_map_np)

    if classification_map is not None:
      classification_map_np = torch_tensor_to_numpy(classification_map)
      np.save(os.path.join(co_llm_save_path, '%04d_classification_map.npy' % timestamp), classification_map_np)

    return


def save_visible_gt_ids(timestamp, save_path, visible_gt_object_ids_dict, invisible_gt_object_ids_dict):
    for cav_id in visible_gt_object_ids_dict:
      gt_object_ids = np.array(visible_gt_object_ids_dict[cav_id])
      np.save(os.path.join(save_path, '%04d_gt_object_id_visible_to_%s.npy' % (timestamp, cav_id)), gt_object_ids)
    for cav_id in invisible_gt_object_ids_dict:
      gt_object_ids = np.array(invisible_gt_object_ids_dict[cav_id])
      np.save(os.path.join(save_path, '%04d_gt_object_id_invisible_to_%s.npy' % (timestamp, cav_id)), gt_object_ids)
    return

