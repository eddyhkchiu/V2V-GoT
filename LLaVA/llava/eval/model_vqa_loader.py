import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

# MY_CODE
import numpy as np
import random
import time

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


# Custom dataset class
class V2V4RealCustomDataset(Dataset):
    def __init__(self, list_data_dict, image_folder, tokenizer, image_processor, model_config, simplified_object_feature, num_input_frames, num_latency_frames, positional_error_std):
        self.list_data_dict = list_data_dict
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.simplified_object_feature = simplified_object_feature
        self.num_input_frames = num_input_frames
        self.num_latency_frames = num_latency_frames
        self.last_global_timestamp_index = 1992 # v2v4real val set total  1993 frames
        self.positional_error_std = positional_error_std

        self.feature_source = args.feature_source
        self.llm_data_path = os.path.join('../DMSTrack/V2V4Real/official_models/', self.feature_source, 'npy/co_llm')

        # MY_DEBUG
        print('self.llm_data_path: ', self.llm_data_path)
        #assert False
        #print('self.simplified_object_feature: ', self.simplified_object_feature)
        #assert False
        # args is a global variable ???
        #print(args)
        self.__getitem__(0)
        self.__getitem__(self.__len__() - 1)
        #assert False


    def zero_padded_roll(self, array, shift, axis):
        result = np.zeros_like(array)

        if shift == 0:
            return array.copy()

        # positive shift
        if shift > 0:
            slicer_src = [slice(None)] * array.ndim
            slicer_dst = [slice(None)] * array.ndim
            slicer_src[axis] = slice(0, -shift)
            slicer_dst[axis] = slice(shift, None)
        else:  # negative shift
            slicer_src = [slice(None)] * array.ndim
            slicer_dst = [slice(None)] * array.ndim
            slicer_src[axis] = slice(-shift, None)
            slicer_dst[axis] = slice(0, shift)

        result[tuple(slicer_dst)] = array[tuple(slicer_src)]
        return result



    def load_single_frame_feature_map(self, cav_ids, feature_map_name, single_frame_global_timestamp_index):
        #print('feature_map_name: ', feature_map_name)
        #print('single_frame_global_timestamp_index: ', single_frame_global_timestamp_index)
        #print('self.num_latency_frames: ', self.num_latency_frames)
        non_ego_single_frame_global_timestamp_index = single_frame_global_timestamp_index - self.num_latency_frames
        if non_ego_single_frame_global_timestamp_index < 0 :
          non_ego_single_frame_global_timestamp_index = 0
        #print('non_ego_single_frame_global_timestamp_index: ', non_ego_single_frame_global_timestamp_index)    
        #assert False

        if self.positional_error_std > 0:
          non_ego_positional_shift_meters = np.random.normal(loc=0.0, scale=self.positional_error_std)
          feature_map_resolution = 0.4 * 4 # v2v4real 0.4 meters to 1 pixel, feature map down scale 4 
          non_ego_positional_shift_num_pixels = int(non_ego_positional_shift_meters / feature_map_resolution)
          non_ego_positional_shift_spatial_dim_idx = random.randint(2, 3)
        else:
          non_ego_positional_shift_num_pixels = 0
          non_ego_positional_shift_meters = 0
          non_ego_positional_shift_spatial_dim_idx = None
        #print('self.self.positional_error_std: ', self.positional_error_std)
        #print('non_ego_positional_shift_meters: ', non_ego_positional_shift_meters)
        #print('non_ego_positional_shift_num_pixels: ', non_ego_positional_shift_num_pixels)  
        #print('non_ego_positional_shift_spatial_dim_idx: ', non_ego_positional_shift_spatial_dim_idx)
        #assert False


        scene_spatial_features_2d_all = []
        #feature_shape = None
        feature_shape = [1, 1, 1, 1]
        for cav_id in cav_ids:

            #if cav_id == 'ego':
            if self.num_latency_frames == 0:
              scene_spatial_features_2d_file = os.path.join(self.llm_data_path, cav_id, '%04d_%s.npy' % (single_frame_global_timestamp_index, feature_map_name))
            else:  
              scene_spatial_features_2d_file = os.path.join(self.llm_data_path, cav_id, '%04d_%s.npy' % (non_ego_single_frame_global_timestamp_index, feature_map_name))

            #print('scene_spatial_features_2d_file: ', scene_spatial_features_2d_file)
            # v2xreal, not every frame has every agent
            try:
              scene_spatial_features_2d = np.load(scene_spatial_features_2d_file)
              feature_shape = scene_spatial_features_2d.shape
            except FileNotFoundError:
              scene_spatial_features_2d = np.zeros(feature_shape)  
            # (1, 256, 50, 88)


            #if cav_id != 'ego' and non_ego_positional_shift_num_pixels != 0:
            if non_ego_positional_shift_num_pixels != 0:
              # shift non_ego feature map  
              #scene_spatial_features_2d = np.roll(scene_spatial_features_2d, shift=non_ego_positional_shift_num_pixels, axis=non_ego_positional_shift_spatial_dim_idx)
              scene_spatial_features_2d = self.zero_padded_roll(scene_spatial_features_2d, non_ego_positional_shift_num_pixels, non_ego_positional_shift_spatial_dim_idx)



            scene_spatial_features_2d = torch.from_numpy(scene_spatial_features_2d[0])
            #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
            # [256, 50, 88]
            scene_spatial_features_2d_all.append(scene_spatial_features_2d)
        scene_spatial_features_2d = torch.stack(scene_spatial_features_2d_all, dim=0)
        #print('scene_spatial_features_2d.shape: ', scene_spatial_features_2d.shape)
        # [2, 256, 50, 88]: [cav_id, feature_size, spatial_dim_0, spatial_dim_1]
        # v2xreal [4, 384, 100, 176]

        return scene_spatial_features_2d


    def load_single_frame_object_features(self, cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav):
        # pick Option 3 for now and continue develop, 
        # we can still switch later

        non_ego_single_frame_global_timestamp_index = single_frame_global_timestamp_index - self.num_latency_frames
        if non_ego_single_frame_global_timestamp_index < 0:
          non_ego_single_frame_global_timestamp_index = 0

        # Option 1: BEV local feature map (from DMSTrack)
        object_features_all = []
        #feature_shape = None
        feature_shape = [1, 1]
        for cav_id in cav_ids:

            #if cav_id == 'ego':
            if self.num_latency_frames == 0:    
              object_features_file =  os.path.join(self.llm_data_path, '../', cav_id, '%04d_feature.npy' % single_frame_global_timestamp_index)
            else:
              object_features_file =  os.path.join(self.llm_data_path, '../', cav_id, '%04d_feature.npy' % non_ego_single_frame_global_timestamp_index)

            try:
              object_features = np.load(object_features_file)
              #print('object_features.shape: ', object_features.shape)
              # (num_boxes=20, 256, 5, 5)
              # just pick the center feature vector
              object_features = object_features[:, :, 2, 2]
              #print('object_features.shape: ', object_features.shape)
              # (num_boxes=20, 256)
              feature_shape = [max_num_boxes_per_cav, object_features.shape[1]]
            except FileNotFoundError:
              object_features = np.zeros(feature_shape)  

            if object_features.shape[0] > max_num_boxes_per_cav:
              object_features = object_features[:max_num_boxes_per_cav]  
            if object_features.shape[0] < max_num_boxes_per_cav:  
              object_features = np.pad(
                object_features,
                ((0, max_num_boxes_per_cav - object_features.shape[0]), (0, 0)),
                'constant', constant_values=(0, 0)
              )
            #print('object_features.shape: ', object_features.shape)
            # (50, 256): (max_num_boxes_per_cav, feature size)
            #print('object_features[:10]: ', object_features[:10])
            object_features = torch.from_numpy(object_features)
            object_features_all.append(object_features)

        object_features = torch.stack(object_features_all, dim=0)
        #print('object_features.shape: ', object_features.shape)
        # [2, 50, 256]:  (num_cavs, max_num_boxes_per_cav, feature_size)
        # v2xreal [4, 100 , 384]
        return object_features


    def load_single_frame_detection_box_score(self, cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav):

        non_ego_single_frame_global_timestamp_index = single_frame_global_timestamp_index - self.num_latency_frames
        if non_ego_single_frame_global_timestamp_index < 0:
          non_ego_single_frame_global_timestamp_index = 0


        detection_box_score_all = []
        feature_shape = None
        for cav_id in cav_ids:

            #if cav_id == 'ego':
            if self.num_latency_frames == 0:    
              detection_box_score_file = os.path.join(self.llm_data_path, cav_id, '%04d_detection_box_score.npy' % single_frame_global_timestamp_index)
            else:
              detection_box_score_file = os.path.join(self.llm_data_path, cav_id, '%04d_detection_box_score.npy' % non_ego_single_frame_global_timestamp_index)

            try:
              detection_box_score = np.load(detection_box_score_file)
              feature_shape = detection_box_score.shape
            except FileNotFoundError:
              detection_box_score = np.zeros([max_num_boxes_per_cav, feature_shape[1]])  
            #print('detection_box_score.shape: ', detection_box_score.shape)
            #print('detection_box_score: ', detection_box_score)
            # (num_boxes=20, 8)
            # [h, w, l, x, y, z, a, s]

            #if cav_id != 'ego':
            if self.positional_error_std > 0:
              #print('detection_box_score: ', detection_box_score)
              non_ego_positional_shift_meters = np.random.normal(loc=0.0, scale=self.positional_error_std, size=detection_box_score.shape)
              # [h, w, l, x, y, z]
              detection_box_score[:, :6] += non_ego_positional_shift_meters[:, :6]
              #print('detection_box_score: ', detection_box_score)


            # current values are in dmstrack coordinate system
            # need to swap y and z to make it in v2v4real coordinate system

            # in v2v4real coordinate system
            if args.simplified_object_feature == 8:
              # [h, w, l, x, y, z, a, s]
              # swap y, z
              detection_box_score = np.concatenate([
                detection_box_score[:, 0:4],
                detection_box_score[:, 5:6],
                detection_box_score[:, 4:5],
                detection_box_score[:, 6:8]
              ], axis=1)
              #print('detection_box_score: ', detection_box_score)
            elif args.simplified_object_feature == 3:
              # [x, y, s]
              detection_box_score = np.concatenate([
                detection_box_score[:, 3:4],
                detection_box_score[:, 5:6],
                detection_box_score[:, 7:8]
              ], axis=1)
              #print('detection_box_score: ', detection_box_score)
            elif args.simplified_object_feature == 2:
              # [x, y]
              # [x, y, s]
              detection_box_score = np.concatenate([
                detection_box_score[:, 3:4],
                detection_box_score[:, 5:6]
              ], axis=1)
              #print('detection_box_score: ', detection_box_score)

            if detection_box_score.shape[0] > max_num_boxes_per_cav:
              detection_box_score = detection_box_score[:max_num_boxes_per_cav]  
            if detection_box_score.shape[0] < max_num_boxes_per_cav:  
              detection_box_score = np.pad(
                detection_box_score,
                ((0, max_num_boxes_per_cav - detection_box_score.shape[0]), (0, 0)),
                'constant', constant_values=(0, 0)
              )
            #print('detection_box_score.shape: ', detection_box_score.shape)
            # (50, 8): (max_num_boxes_per_cav, self.data_args.simplified_object_feature)
            #print('detection_box_score[:10]: ', detection_box_score[:10])
            detection_box_score = torch.from_numpy(detection_box_score)
            detection_box_score_all.append(detection_box_score)

        detection_box_score = torch.stack(detection_box_score_all, dim=0)
        #print('detection_box_score.shape: ', detection_box_score.shape)
        # [2, 50, 8]:  (num_cavs, max_num_boxes_per_cav, self.data_args.simplified_object_feature)
        return detection_box_score





    def __getitem__(self, index):
        data_dict = self.list_data_dict[index]
        global_timestamp_index = data_dict['global_timestamp_index']
        local_timestamp_index = data_dict['local_timestamp_index']

        if args.ego_only == 'True' or args.ego_only == 'true':
            cav_ids = ['ego']
        else:    
            cav_ids = ['ego', '1']
        #print('cav_ids: ', cav_ids)    

        num_input_frames = self.num_input_frames

        load_data_dict = {}
        for feature_map_name in ['spatial_features_2d', 'regression_map', 'classification_map']:
            feature_map_all_frames = []
            for past_idx in range(num_input_frames):
                if local_timestamp_index < past_idx:  # no past frames:
                    single_frame_global_timestamp_index = global_timestamp_index
                else:
                    single_frame_global_timestamp_index = global_timestamp_index - past_idx

                single_frame_feature_map = self.load_single_frame_feature_map(cav_ids, feature_map_name, single_frame_global_timestamp_index)
                feature_map_all_frames.append(single_frame_feature_map)

            feature_map_all_frames = torch.stack(feature_map_all_frames, dim=0) 
            if feature_map_name == 'spatial_features_2d':
              load_data_dict['scene_point_feature_map'] = feature_map_all_frames
            else:  
              load_data_dict[feature_map_name] = feature_map_all_frames

        scene_point_feature_map = load_data_dict['scene_point_feature_map']      
        regression_map = load_data_dict['regression_map']
        classification_map = load_data_dict['classification_map']
        #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
        # [2, 2, 256, 50, 88]
        # [num_input_frames, num_cavs, feature_size, spatial_dim_0, spatial_dim_1]
        #print('regression_map.shape: ', regression_map.shape)
        # [2, 2, 14, 50, 88]
        #print('classification_map.shape: ', classification_map.shape)
        # [2, 2, 2, 50, 88]
       

        # Object-level feature vectors
        max_num_boxes_per_cav = 50

        # Option 1: BEV local feature map (from DMSTrack)
        object_features_all_frames = []
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:  # no past frames:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx

            single_frame_object_features = self.load_single_frame_object_features(cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav)
            object_features_all_frames.append(single_frame_object_features)

        object_features_all_frames = torch.stack(object_features_all_frames, dim=0)
        #print('object_features_all_frames.shape: ', object_features_all_frames.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=256]
        object_features = object_features_all_frames 
        #print('object_features.shape: ', object_features.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=256]


        # Option 3: box [h, w, l, x, y, z, a] and score [s]
        # Directly pad to max number of boxes per CAV (50)
        # [max_num_boxes_per_cav * num_cavs, 8]
        detection_box_score_all_frames = []
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:  # no past frames:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx

            single_frame_detection_box_score = self.load_single_frame_detection_box_score(cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav)
            detection_box_score_all_frames.append(single_frame_detection_box_score)

        detection_box_score_all_frames = torch.stack(detection_box_score_all_frames, dim=0)
        #print('detection_box_score_all_frames.shape: ', detection_box_score_all_frames.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=8]
        detection_box_score = detection_box_score_all_frames
        #print('detection_box_score.shape: ', detection_box_score.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=8]

        active_agent_mask = torch.from_numpy(np.ones([num_input_frames, 2]))


        qs = DEFAULT_IMAGE_TOKEN + '\n' +  data_dict['conversations'][0]['value']
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = torch.zeros(3, 336, 336)
        image_sizes=[3, 336, 336]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image_sizes, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask

    def __len__(self):
        return len(self.list_data_dict)


class V2XRealCustomDataset(V2V4RealCustomDataset):
    def __init__(self, list_data_dict, image_folder, tokenizer, image_processor, model_config, simplified_object_feature, num_input_frames, num_latency_frames, positional_error_std):
        super(V2XRealCustomDataset, self).__init__(list_data_dict, image_folder, tokenizer, image_processor, model_config, simplified_object_feature, num_input_frames, num_latency_frames, positional_error_std) 
        self.llm_data_path = os.path.join('../V2X-Real/my_models/', self.feature_source, 'npy/co_llm')
        print('self.llm_data_path: ', self.llm_data_path)

        active_agent_mask_save_file = os.path.join(self.llm_data_path, 'active_agent_mask.npy')
        active_agent_mask = np.load(active_agent_mask_save_file)
        self.active_agent_mask = active_agent_mask

        self.num_latency_frames = num_latency_frames
        self.last_global_timestamp_index = 1252 # v2xreal val set total 1253 frames
        self.positional_error_std = positional_error_std


        # MY_DEBUG
        #self.__getitem__(1)
        #assert False

    def __getitem__(self, index):
        data_dict = self.list_data_dict[index]
        global_timestamp_index = data_dict['global_timestamp_index']
        local_timestamp_index = data_dict['local_timestamp_index']

        if args.ego_only == 'True' or args.ego_only == 'true':
            cav_ids = ['ego']
        else:
            cav_ids = ['ego', '2', '-1', '-2']
        #print('cav_ids: ', cav_ids)

        num_input_frames = self.num_input_frames

        load_data_dict = {}
        for feature_map_name in ['spatial_features_2d', 'regression_map', 'classification_map']:
            feature_map_all_frames = []
            for past_idx in range(num_input_frames):
                if local_timestamp_index < past_idx:  # no past frames:
                    single_frame_global_timestamp_index = global_timestamp_index
                else:
                    single_frame_global_timestamp_index = global_timestamp_index - past_idx

                single_frame_feature_map = self.load_single_frame_feature_map(cav_ids, feature_map_name, single_frame_global_timestamp_index)
                feature_map_all_frames.append(single_frame_feature_map)

            feature_map_all_frames = torch.stack(feature_map_all_frames, dim=0)
            if feature_map_name == 'spatial_features_2d':
              load_data_dict['scene_point_feature_map'] = feature_map_all_frames
            else:
              load_data_dict[feature_map_name] = feature_map_all_frames

        scene_point_feature_map = load_data_dict['scene_point_feature_map']
        regression_map = load_data_dict['regression_map']
        classification_map = load_data_dict['classification_map']
        #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
        # [2, 2, 256, 50, 88]
        # [num_input_frames, num_cavs, feature_size, spatial_dim_0, spatial_dim_1]
        # v2xreal [1, 4, 384, 100, 176]
        #print('regression_map.shape: ', regression_map.shape)
        # [2, 2, 14, 50, 88]
        # v2xreal [1, 4, 42, 100, 176]
        #print('classification_map.shape: ', classification_map.shape)
        # [2, 2, 2, 50, 88]
        # v2xreal [1, 4, 18, 100, 176]


        # Object-level feature vectors
        max_num_boxes_per_cav = 100

        # Option 1: BEV local feature map (from DMSTrack)
        object_features_all_frames = []
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:  # no past frames:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx

            single_frame_object_features = self.load_single_frame_object_features(cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav)
            object_features_all_frames.append(single_frame_object_features)

        object_features_all_frames = torch.stack(object_features_all_frames, dim=0)
        #print('object_features_all_frames.shape: ', object_features_all_frames.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=256]
        object_features = object_features_all_frames
        #print('object_features.shape: ', object_features.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=256]
        # v2xreal [1, 4, 100, 384]


        # Option 3: box [h, w, l, x, y, z, a] and score [s]
        # Directly pad to max number of boxes per CAV (50)
        # [max_num_boxes_per_cav * num_cavs, 8]
        detection_box_score_all_frames = []
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:  # no past frames:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx

            single_frame_detection_box_score = self.load_single_frame_detection_box_score(cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav)
            detection_box_score_all_frames.append(single_frame_detection_box_score)

        detection_box_score_all_frames = torch.stack(detection_box_score_all_frames, dim=0)
        #print('detection_box_score_all_frames.shape: ', detection_box_score_all_frames.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=8]
        detection_box_score = detection_box_score_all_frames
        #print('detection_box_score.shape: ', detection_box_score.shape)
        # [num_input_frames=2, num_cavs=2, num_max_boxes_per_cav=50, feature_size=8]
        # v2xreal [1, 4, 100, 9]


        active_agent_mask_all_frames = []
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:  # no past frames:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx
            single_frame_active_agent_mask = self.active_agent_mask[single_frame_global_timestamp_index]
            single_frame_active_agent_mask = torch.from_numpy(single_frame_active_agent_mask)
            active_agent_mask_all_frames.append(single_frame_active_agent_mask)
        active_agent_mask_all_frames = torch.stack(active_agent_mask_all_frames, dim=0)
        active_agent_mask = active_agent_mask_all_frames
        #print('active_agent_mask: ', active_agent_mask)
        # [[ True,  True, False, False]]

        qs = DEFAULT_IMAGE_TOKEN + '\n' +  data_dict['conversations'][0]['value']
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = torch.zeros(3, 336, 336)
        image_sizes=[3, 336, 336]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image_sizes, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask







def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def v2v4real_collate_fn(batch):
    input_ids, image_tensors, image_sizes, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)

    scene_point_feature_map = torch.stack(scene_point_feature_map, dim=0)
    regression_map = torch.stack(regression_map, dim=0)
    classification_map = torch.stack(classification_map, dim=0)

    detection_box_score = torch.stack(detection_box_score, dim=0)

    object_features = torch.stack(object_features, dim=0)

    active_agent_mask = torch.stack(active_agent_mask, dim=0)

    return input_ids, image_tensors, image_sizes, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask


def v2xreal_collate_fn(batch):
    input_ids, image_tensors, image_sizes, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)

    scene_point_feature_map = torch.stack(scene_point_feature_map, dim=0)
    regression_map = torch.stack(regression_map, dim=0)
    classification_map = torch.stack(classification_map, dim=0)

    detection_box_score = torch.stack(detection_box_score, dim=0)

    object_features = torch.stack(object_features, dim=0)

    active_agent_mask = torch.stack(active_agent_mask, dim=0)

    return input_ids, image_tensors, image_sizes, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def create_data_loader_v2v4real(list_data_dict, image_folder, tokenizer, image_processor, model_config, simplified_object_feature, num_input_frames, num_latency_frames, positional_error_std, batch_size=1, num_workers=64):
    assert batch_size == 1, "batch_size must be 1"
    dataset = V2V4RealCustomDataset(list_data_dict, image_folder, tokenizer, image_processor, model_config, simplified_object_feature, num_input_frames, num_latency_frames, positional_error_std)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=v2v4real_collate_fn)
    return data_loader

def create_data_loader_v2xreal(list_data_dict, image_folder, tokenizer, image_processor, model_config, simplified_object_feature, num_input_frames, num_latency_frames, positional_error_std, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = V2XRealCustomDataset(list_data_dict, image_folder, tokenizer, image_processor, model_config, simplified_object_feature, num_input_frames, num_latency_frames, positional_error_std)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=v2xreal_collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes, ), line in tqdm(zip(data_loader, questions), total=len(questions)):


        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()


def inference_v2v4real_3d_grounding(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    my_model_config = {
      'mm_scene_projector_input_size': args.mm_scene_projector_input_size,
      'scene_level_only': True if args.scene_level_only == 'True' or args.scene_level_only == 'true' else False,
      'object_level_only': True if args.object_level_only == 'True' or args.object_level_only == 'true' else False,
      'scene_feature_mode': args.scene_feature_mode,
      'object_feature_mode': args.object_feature_mode,
      'num_input_frames': args.num_input_frames,
      'ego_only': True if args.ego_only == 'True' or args.ego_only == 'true' else False,
      'feature_source': args.feature_source,
      'dataset_source': args.dataset_source,
      'num_latency_frames': args.num_latency_frames,
      'positional_error_std': args.positional_error_10_std * 0.1,
    }
    if args.feature_source in ['cobevt', 'early', 'v2xvit', 'attfuse', 'no_fusion']: # and other intermediate fusion, early fusion
        assert(args.ego_only == 'True' or args.ego_only == 'true')
    if my_model_config['scene_level_only']:
        assert(not my_model_config['object_level_only'])
    if my_model_config['object_level_only']:
        assert(not my_model_config['scene_level_only'])

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, my_model_config=my_model_config)

    #questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    list_data_dict = json.load(open(args.question_file, "r"))

    # MY_DEBUG
    #list_data_dict = list_data_dict[:16]

    list_data_dict = get_chunk(list_data_dict, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    if args.dataset_source == 'v2v4real':
      data_loader = create_data_loader_v2v4real(list_data_dict, args.image_folder, tokenizer, image_processor, model.config, args.simplified_object_feature, args.num_input_frames, args.num_latency_frames, args.positional_error_10_std * 0.1)
    else:  
      data_loader = create_data_loader_v2xreal(list_data_dict, args.image_folder, tokenizer, image_processor, model.config, args.simplified_object_feature, args.num_input_frames, args.num_latency_frames, args.positional_error_10_std * 0.1)

  
    inference_time_sum = 0
    inference_time_count = 0

    for (input_ids, image_tensor, image_sizes, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask), data_dict in tqdm(zip(data_loader, list_data_dict), total=len(list_data_dict)):

        #print('input_ids: ', input_ids)
        #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
        # [1, 256*2, 50, 88]
        #print('detection_box_score.shape: ', detection_box_score.shape)
        # [1, 50*2, 8]
        #print('data_dict: ', data_dict)
        # {'id': 0, 'conversations': [{'from': 'human', 'value': 'What is the object at the location [-20.5, -0.1]? What are its bounding box parameters? \n'}, {'from': 'gpt', 'value': 'A car is at the location. Its bounding box parameters are [1.7, 2.1, 4.0, -20.5, -1.0, -0.1, 0.03]. \n'}], 'scenario_index': 0, 'local_timestamp_index': 0, 'global_timestamp_index': 0}

        scene_point_feature_map = scene_point_feature_map.to(dtype=torch.float16, device='cuda', non_blocking=True)
        regression_map = regression_map.to(dtype=torch.float16, device='cuda', non_blocking=True)
        classification_map = classification_map.to(dtype=torch.float16, device='cuda', non_blocking=True)
        detection_box_score = detection_box_score.to(dtype=torch.float16, device='cuda', non_blocking=True)
        object_features = object_features.to(dtype=torch.float16, device='cuda', non_blocking=True)
        active_agent_mask = active_agent_mask.to(dtype=torch.bool, device='cuda', non_blocking=True)

        #idx = line["question_id"]
        #cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)


        #start_time = time.time()
        with torch.inference_mode():
            #output_ids, inference_time = model.generate(
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                scene_point_feature_map=scene_point_feature_map,
                regression_map=regression_map,
                classification_map=classification_map,
                detection_box_score=detection_box_score,
                object_features=object_features,
                active_agent_mask=active_agent_mask)

        #print('outputs: ', outputs)
        #end_time = time.time()
        #inference_time = end_time - start_time
        #inference_time_sum += inference_time
        #inference_time_count += 1
        #print('inference_time: ', inference_time)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


        #ans_id = shortuuid.uuid()
        #ans_file.write(json.dumps({"question_id": idx,
        #                           "prompt": cur_prompt,
        #                           "text": outputs,
        #                           "answer_id": ans_id,
        #                           "model_id": model_name,
        #                           "metadata": {}}) + "\n")
        data_dict['outputs'] = outputs
        ans_file.write(
            json.dumps(data_dict)
            + "\n")
        # ans_file.flush()
    ans_file.close()

    #inference_time_average = inference_time_sum / inference_time_count
    #print('inference_time_average: ', inference_time_average)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    # MY_CODE
    # v2v4real 
    #parser.add_argument("--llm-data-path", type=str, default=None)
    parser.add_argument("--simplified-object-feature", type=int, default=0)
    parser.add_argument("--mm_scene_projector_input_size", type=int, default=None)
    parser.add_argument("--scene_level_only", type=str, default='False')
    parser.add_argument("--object_level_only", type=str, default='False')
    parser.add_argument("--scene_feature_mode", type=str, default='shallow')
    parser.add_argument("--object_feature_mode", type=str, default='shallow')
    parser.add_argument("--num_input_frames", type=int, default=1)
    parser.add_argument("--ego_only", type=str, default='False')
    parser.add_argument("--feature_source", type=str, default='no_fusion_keep_all')
    # v2xreal
    parser.add_argument("--dataset_source", type=str, default='v2v4real')
    # dynamic environment
    parser.add_argument("--num_latency_frames", type=int, default=0)
    parser.add_argument("--positional_error_10_std", type=int, default=0)

    args = parser.parse_args()


    # MY_CODE
    #eval_model(args)
    inference_v2v4real_3d_grounding(args)
