#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector, build_scene_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

import math

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            print('config: ', config) # here
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            self.mm_scene_projector = build_scene_vision_projector(config)

            for p in self.mm_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

            for p in self.mm_scene_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

            #self.initialize_mm_scene_projector()


            for p in self.mm_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

            for p in self.mm_scene_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
            #assert False

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                print('here unpad')
                # not hit
                assert False
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

                
    # MY_CODE
    def initialize_mm_scene_projector(self):            
        # https://github.com/eddyhkchiu/my_co_llm_driver/blob/bb7af2486e61886311454ec186005e5dad0f2d87/LLaVA/llava/model/builder.py#L76
        non_lora_trainables = torch.load('checkpoints/llava-v1.5-7b-task-lora/llava-v1.5-7b-task-lora_v2v4real_3d_grounding_v6sdup3_init_scene_and_object/non_lora_trainables.bin', map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}

        #if any(k.startswith('model.model.') for k in non_lora_trainables):
        #    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}


        non_lora_trainables = {(k[31:] if k.startswith('model.model.mm_scene_projector.') else k): v for k, v in non_lora_trainables.items()}

        print('non_lora_trainables: ', non_lora_trainables)
            
        self.mm_scene_projector.load_state_dict(non_lora_trainables, strict=False)
        assert False




    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        #print('model_args: ', model_args)

        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            print('here 1')
            # train from vicuna checkpoint will hit here
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
            #assert False    
        else:
            print('here 2') 
            # hit here when train from llava checkpoint
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

            for p in self.mm_scene_projector.parameters():
                p.requires_grad = True
                #print('p: ', p)
                # tensor([], device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

            
            #self.initialize_mm_scene_projector()
            # does not work
            # torch.Size([0])
            #assert False    

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print('here: pretrain_mm_mlp_adapter')
            # so far no hit
            assert False


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        #print('images.shape: ', images.shape)
        # [1, 3, 336, 336]
        image_features = self.get_model().get_vision_tower()(images)
        #print('image_features.shape: ', image_features.shape)
        # [1, 576, 1024]
        image_features = self.get_model().mm_projector(image_features)
        #print('image_features.shape: ', image_features.shape)
        # [1, 576, 4096]
        #assert False
        return image_features

    def generate_scene_level_features_option_3(self, scene_point_feature_map):
        #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
        # [32, 2, 2, 256, 50, 88]
        # [batch_size, num_input_frames, num_cavs, feature_size, spatial_left_right, spatial_forward_backward]


        # pad spatial_left_right to 51
        # patch size [3, 4] to generate 17 * 22 = 374 tokens
        # inside each patch, linearize the feature
        # each token's feature size is 256 * 3 * 4 = 3072
        # then apply projector to have feature size 1024
        point_features = scene_point_feature_map
        # [32, 2, 2, 256, 50, 88]

        # Step 1, padding spatial_left_right to 51
        p2d = (0, 0, 0, 1)
        point_features = nn.functional.pad(point_features, p2d, "constant", 0)
        #print('point_features.shape: ', point_features.shape)
        # [32, 2, 2, 256, 51, 88]

        # Step 2, patch [51, 88] to [3, 4] * [17, 22]
        # currently only 4-D tensors are supported
        batch_size, num_input_frames, num_cavs, feature_size, spatial_dim_0, spatial_dim_1 = point_features.shape
        point_features = point_features.reshape([batch_size * num_input_frames * num_cavs, feature_size, spatial_dim_0, spatial_dim_1])
        #print('point_features.shape: ', point_features.shape)
        # [32 * 2 * 2, 256, 51, 88]
        point_features = nn.functional.unfold(point_features, (3, 4), stride=(3,4))
        #print('point_features.shape: ', point_features.shape)
        # [32 * 2 * 2, 256 * 3 * 4, 17 * 22]
        # [32 * 2 * 2, 3072, 374]

        _, new_feature_size, num_tokens = point_features.shape
        point_features = point_features.reshape([batch_size, num_input_frames, num_cavs, new_feature_size, num_tokens])
        #print('point_features.shape: ', point_features.shape)
        # [32, 2, 2, 3072, 374]
        


        return point_features


    def generate_scene_level_features(self, my_model_config, scene_point_feature_map, regression_map, classification_map, active_agent_mask):
        '''
        Input:
          regression_map
            [batch_size, num_input_frames=2, num_cavs=2, feature_size=14, spatial_dim_0=50, spatial_dim_1=88]
          classification_map
            [batch_size, num_input_frames=2, num_cavs=2, feature_size=2, spatial_dim_0=50, spatial_dim_1=88]
          active_agent_mask
            [batch_size, num_input_frames=2, num_cavs=4] bool indicate whether agent i is active
        Output:
          scene_level_features: list of scene_level_feature, list size=num_cavs
            scene_level_feature: [batch_size, num_tokens=num_patches=220, feature_size=4096]
        '''
        # Scene-level

        if my_model_config['scene_feature_mode'] == 'deep':

          # New approach: deep features from scene_point_feature_map
          #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
          # [32, 2, 2, 256, 50, 88]
          # [32, 2, 1, 256, 48, 128] # cobevt
          scene_level_features = self.generate_scene_level_features_option_3(scene_point_feature_map)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 3072, 374]
          # [32, 2, 2, 256 * 3 * 4, 17 * 22]
          # [batch_size, num_input_frames, num_cavs, feature_size, num_tokens]
          # [32, 2, 2, 3072, 512] cobevt
          #assert False

          # swap axis
          #scene_level_features = torch.permute(scene_level_features, (0, 1, 3, 2)) 
          scene_level_features = torch.swapaxes(scene_level_features, -1, -2)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 374, 3072] # new option 3
          # [batch_size, num_input_frames, num_cavs, num_tokens, feature_size]
          #assert False


        # [batch_size, feature_size, spatial_left_right, spatial_forward_backward]
        # patch size [2, 2] to generate 25 * 44 = 1100 tokens
        # inside each patch, linearize the feature
        # each token's feature size is 256 * 2 * 2 = 1024
        # then apply projector to have feature size 4096

        ####point_features = scene_point_feature_map
        # Step 2, patch [50, 88] to [2, 2] * [25, 44]
        ####point_features = nn.functional.unfold(point_features, (2, 2), stride=(2, 2))
        #print('point_features.shape: ', point_features.shape)
        # [32, 512*2*2=2048, 25*44=1100]
        # [2, 256 * 2 * 2, 25 * 44]
        # [batch_size, feature_size, num_tokens]

        # Step 3, swap axis to have the same format of image_features:
        ####point_features = torch.permute(point_features, (0, 2, 1))
        #print('point_features.shape: ', point_features.shape)
        # [32, 1100, 2048]
        # [2, 1100, 1024]
        # [batch_size, num_point_tokens, feature_size]

        # Step 4, apply projector
        ####point_features = self.get_model().mm_projector(point_features)
        #print('point_features.shape: ', point_features.shape)
        # [2, 1100, 4096]
        # [batch_size, num_point_tokens, feature_size]


        elif my_model_config['scene_feature_mode'] == 'shallow':
          #print('my_model_config: ', my_model_config)  
          # Old approach before 0926
          # Old approach: shallow features from reg and cls maps
          #print('regression_map.shape: ', regression_map.shape)
          # [32, 2, 14, 50, 88]
          # v2xreal [32, 1, 4, 42, 100, 176]
          #print('classification_map.shape: ', classification_map.shape)
          # [32, 2, 2, 50, 88]
          # v2xreal [32, 1, 4, 18, 100, 176]

          # split to separate map per cav
          # patch size [5, 4] to generate (50/5) * (88/4) = 220 tokens
          # feature size (14 + 2 = 16) * 5 * 4  =  320 for each cav
          # pad 320 to 1024 before applying projector
          # split to separate map per cav
          #regression_map = torch.chunk(regression_map, num_cavs, 1)
          #print('regression_map[0].shape: ', regression_map[0].shape)
          #classification_map = torch.chunk(classification_map, num_cavs, 1)
          #scene_level_features = [torch.cat([regression_map[i], classification_map[i]], dim=1) for i in range(num_cavs)]
          #print('scene_level_features[0].shape: ', scene_level_features[0].shape)
          #print('scene_level_features[1].shape: ', scene_level_features[1].shape)
          # [32, 16, 50, 88]


          scene_level_features = torch.cat([regression_map, classification_map], dim=3)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 16, 50, 88]
          # [32, 2, 1, 16, 48, 128] # cobevt
          # v2xreal [32, 1, 4, 60, 100, 176]
          batch_size, num_input_frames, num_cavs, feature_size, spatial_dim_0, spatial_dim_1 = scene_level_features.shape
          scene_level_features = scene_level_features.reshape([batch_size * num_input_frames * num_cavs, feature_size, spatial_dim_0, spatial_dim_1])
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32 * 2 * 2, 16, 50, 88]
          # v2xreal [128 = 32 * 1 * 4, 60 = 42 + 18, 100, 176]

          # For v2xreal, we need to reduce the spatial dimention
          if my_model_config['dataset_source'] == 'v2xreal':
            scene_level_features = nn.functional.avg_pool2d(scene_level_features, kernel_size=2, stride=2)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # v2xreal [128, 60, 50, 88]


          # only 4-D tensor is supported
          # patch size [5, 4] to generate (50/5) * (88/4) = 220 tokens
          patch_size = (5, 4) # v2v4real
          if my_model_config['dataset_source'] == 'v2xreal':
              patch_size = (4, 4)

          #scene_level_features = nn.functional.unfold(scene_level_features,  (5, 4), stride=(5, 4))
          scene_level_features = nn.functional.unfold(scene_level_features,  patch_size, stride=patch_size)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32 * 2 * 2, 16*5*4, 50/5 * 88/4]
          # v2xreal ([128=32*1*4, 960=60*4*4, 264=50//4 * 88/4]
          _, new_feature_size, num_tokens = scene_level_features.shape

          scene_level_features = scene_level_features.reshape([batch_size, num_input_frames, num_cavs, new_feature_size, num_tokens])
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 16*5*4, 50/5 * 88/4]
          # [32, 2, 2, 320, 220]
          # [32, 2, 1, 320, 288] cobevt [48, 128]
          # v2xreal [32, 1, 4, 960, 264]

          #assert False





          
          # swap axis
          #scene_level_features = torch.permute(scene_level_features, (0, 1, 3, 2))
          scene_level_features = torch.swapaxes(scene_level_features, -1, -2)

          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 220, 320] # old shallow option 1
          # [batch_size, num_input_frames, num_cavs, num_tokens, feature_size]
          # v2xreal [32, 1, 4, 264, 960]

          # pad 320 to 1024 before applying projector
          # pre-pad, different from object level append-pad
          mm_projector_input_size = 1024
          scene_level_features = nn.functional.pad(
              scene_level_features,
              (mm_projector_input_size - scene_level_features.shape[-1], 0),
              'constant',
              0
          )
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [32, 2, 2, 220, 1024]
          # v2xreal [32, 1, 4, 264, 1024]
          #assert False

        else:
          print('not implemented')  
          assert False
        

        # apply projector
        # MY_DEBUG
        if my_model_config['scene_feature_mode'] == 'shallow':
          scene_level_features = self.get_model().mm_projector(scene_level_features)
        elif my_model_config['scene_feature_mode'] == 'deep':  
          # New approach: do not do padding, directly set mm_scene_projector's input size to 3072
          scene_level_features = self.get_model().mm_scene_projector(scene_level_features)
        else:  
          assert False  

        #print('scene_level_features.shape: ', scene_level_features.shape)
        # [32, 2, 2, 220, 4096] # old shallow option 1
        # [32, 2, 2, 374, 4096] # new deep option 3
        # 
        # cobevt
        # [32, 2, 1, 288, 4096] # old shallow option 1
        # [32, 2, 1, 512, 4096] # new deep option 3
        #
        # v2xreal
        # [32, 1, 4, 264, 4096]

        return scene_level_features

    
    def get_positional_embedding(self, detection_box_score, hidden_dim):
        '''
        Similar to
        https://github.com/eddyhkchiu/DMSTrack/blob/master/DMSTrack/model.py#L202

        Input:
          detection_box_score:
            [batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, 7 + 1 parameters]
          hidden_dim: 
            constant for positional coding

        Output:
          positional_embedding: [batch_size, max_num_boxes_per_cav * num_cavs, feature_size=8*hidden_dim]
        '''
        positional_embedding = None
        #print('detection_box_score.shape: ', detection_box_score.shape)
        # [32, 2, 2, 50, 8]
        # 8: [h, w, l, x, y, z, a, s]
        batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, num_parameters = detection_box_score.shape

        #print('hidden_dim: ', hidden_dim)
        # 128

        positional_feature = torch.reshape(detection_box_score, [batch_size * num_input_frames * num_cavs *  max_num_boxes_per_cav, num_parameters])
        #print('positional_feature.shape: ', positional_feature.shape)

        # normalize all distance by dividing by max distance 200 meters,
        # before applying sin cos positional embedding
        # TODO: move this constant to dataset dependent config
        max_distance = 200
        positional_feature[:, :6] /= max_distance
        #print('positional_feature: ', positional_feature)
        # after this normalization, the range of distances is [-1, 1]

        half_hidden_dim = hidden_dim // 2
        # because the range of distance is [-1, 1]
        # we want scale = math.pi
        # the original code scale = 2 * math.pi, is for range [0, 1]
        scale = math.pi

        dim_t = torch.arange(half_hidden_dim, dtype=positional_feature.dtype, device=positional_feature.device)
        dim_t = 2 ** (2 * dim_t / hidden_dim)
        #print('dim_t[:5]: ', dim_t[:5])
        # (64)

        positional_embedding = positional_feature.unsqueeze(dim=2)
        #print('positional_embedding.shape: ', positional_embedding.shape)
        # [3200, 8, 1]
        # (batch_size * max_num_boxes, num_parameters, 1)

        positional_embedding = positional_embedding * scale / dim_t
        #print('positional_embedding.shape: ', positional_embedding.shape)
        # [3200, 8, 64]
        # (batch_size * max_num_boxes, num_parameters, half_hidden_dim)
        #print('positional_embedding[0, -1, :5]: ', positional_embedding[0, -1, :5])


        positional_embedding = torch.cat([positional_embedding.sin(), positional_embedding.cos()], dim=2)
        #print('positional_embedding.shape: ', positional_embedding.shape)
        # [3200, 8, 128]
        # (batch_size * max_num_boxes, num_parameters, hidden_dim)
        # print encoded x
        #print('positional_embedding[:, 3, :]: ', positional_embedding[:, 3, :])

        # final reshape
        positional_embedding = positional_embedding.reshape([batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, num_parameters * hidden_dim])
        #print('positional_embedding.shape: ', positional_embedding.shape)
        # [32, 2, 2, 50, 8*128=1024]

        #assert False
        return positional_embedding


    def generate_object_level_features(self, my_model_config, detection_box_score, object_features):
        '''
        Input:
          detection_box_score:
            [batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, 7 + 1 parameters]
          object_features:
            [batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, 256 feature values]

        Output:
          object_level_feature: [batch_size, num_input_frames, num_cavs, num_tokens=max_num_objects=50, feature_size=4096]
        '''
        #print('object_features.shape: ', object_features.shape)
        # [32, 2, 2, 50, 256]
        # [batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, feature_size]
        batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, feature_size = object_features.shape


        if my_model_config['object_feature_mode'] == 'deep':
          # New approach:
          # concat detection_box_score and object_features
          object_level_features = torch.cat([
            detection_box_score,
            object_features
          ], dim=-1)
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, 2, 2, 50, 264]
          #assert False

          mm_projector_input_size = 1024
          object_level_features = nn.functional.pad(
              object_level_features,
              (0, mm_projector_input_size - object_level_features.shape[-1]),
              'constant',
              0
          )
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, 2, 2, 50, 1024]
          #assert False


        elif my_model_config['object_feature_mode'] == 'shallow':
          # Old approach 0928
          # detection_box_score only

          # Object-level
          # detection_box_score
          # list of batch_sample
          # each has (num_boxes, box_score_feature_size=8)
          # box_score_feature: [h, w, l, x, y, z, a, s]
          #print('detection_box_score.shape: ', detection_box_score.shape) 
          # [32, 2, 2, 50, 8] [batch_size, num_cavs, max_num_boxes_per_cav, 7 + 1 parameters]
          # pad zero to feature size mm_projector_input_size (original 1024)
          mm_projector_input_size = 1024
          detection_box_score = nn.functional.pad(
              detection_box_score, 
              (0, mm_projector_input_size - detection_box_score.shape[-1]), 
              'constant',
              0
          )
          #print('detection_box_score.shape: ', detection_box_score.shape) 
          # [32, 2, 2, 50, 1024]
          object_level_features = detection_box_score
  
          # TODO: better way is to use different projector to make feature size from 7 to 4096
          #object_level_features = torch.zeros([batch_size, 50, 1024], dtype=detection_box_score[0].dtype, device=detection_box_score[0].device)
          #for i in range(batch_size):
          #    object_level_features[i, :detection_box_score[i].shape[0], :detection_box_score[i].shape[1]] = detection_box_score[i]
          #print('object_level_features.shape: ', object_level_features.shape)
          #print('object_level_features[0, :10, :10]: ', object_level_features[0, :10, :10])
          #print('object_level_features[1, :10, :10]: ', object_level_features[1, :10, :10])

        elif my_model_config['object_feature_mode'] == 'pos128':
          #print('detection_box_score.shape: ', detection_box_score.shape) 
          # [32, 2, 50, 8] [batch_size, num_cavs, max_num_boxes_per_cav, 7 + 1 parameters]  
          hidden_dim = 128  
          positional_embedding = self.get_positional_embedding(detection_box_score, hidden_dim)  
          positional_feature_size = positional_embedding.shape[-1]

          object_level_features = positional_embedding
          #print('object_level_features.shape: ', object_level_features.shape)
          # [batch_size=32, num_input_frames=2, num_cavs=2, max_num_boxes_per_cav=50, 8*128=1024]

        else:
          print('not implemented')  
          assert False


        # TODO: use different projector
        object_level_features = self.get_model().mm_projector(object_level_features)
        #print('object_level_features.shape: ', object_level_features.shape)
        # [32, 2, 2, 50, 4096]
        # [batch_size=32, num_input_frames=2, num_cavs=2, max_num_boxes_per_cav=50, llm_feature_size=4096]
        # v2vreal [32, 1, 4, 100, 4096]
        #assert False

        return object_level_features


    def concat_features_original(self, scene_level_features, object_level_features):
        '''
        Original approach: 
          [num_input_frames, cav_id, scene_or_object]
          f_0
            cav_ego_scene, cav_ego_object, cav_1_scene, cav_1_object
          f_1
            cav_ego_scene, cav_ego_object, cav_1_scene, cav_1_object

        Input:
          scene_level_features: [batch_size, num_input_frames, num_cavs, num_tokens, feature_size=4096]
          object_level_features: [batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, feature_size=4096]
        Output:
          point_features: [batch_size, final_num_tokens, feature_size=4096]
        '''
        batch_size, num_input_frames, num_cavs, _, feature_size = scene_level_features.shape

        point_features_all_frames = []
        for f in range(num_input_frames):

          
          if num_cavs == 4:
            single_frame_point_features = torch.cat([
              scene_level_features[:, f, 0, :, :],  
              object_level_features[:, f, 0, :, :],  
              scene_level_features[:, f, 1, :, :],  
              object_level_features[:, f, 1, :, :],  
              scene_level_features[:, f, 2, :, :],  
              object_level_features[:, f, 2, :, :],  
              scene_level_features[:, f, 3, :, :],  
              object_level_features[:, f, 3, :, :],  
            ],  dim=1)
          elif num_cavs == 3:
            single_frame_point_features = torch.cat([
              scene_level_features[:, f, 0, :, :],  
              object_level_features[:, f, 0, :, :],  
              scene_level_features[:, f, 1, :, :],  
              object_level_features[:, f, 1, :, :],  
              scene_level_features[:, f, 2, :, :],  
              object_level_features[:, f, 2, :, :],  
            ],  dim=1)
          elif num_cavs == 2:
            single_frame_point_features = torch.cat([
              scene_level_features[:, f, 0, :, :],  
              object_level_features[:, f, 0, :, :],  
              scene_level_features[:, f, 1, :, :],  
              object_level_features[:, f, 1, :, :],  
            ],  dim=1)
          else:  
            assert(num_cavs == 1)
            single_frame_point_features = torch.cat([
              scene_level_features[:, f, 0, :, :],  
              object_level_features[:, f, 0, :, :],  
            ],  dim=1)
 
          point_features_all_frames.append(single_frame_point_features)

        point_features = torch.cat(point_features_all_frames, dim=1)
        #print('point_features.shape: ', point_features.shape)
        # num_cavs == 2
        # [32, 540*num_input_frames , 4096] # shallow
        # [32, 848*num_input_frames , 4096] # deep
        # num_cavs == 1
        # [32, 270*num_input_frames , 4096] # shallow
        # [32, 424*num_input_frames , 4096] # deep
        # 
        # v2xreal
        # num_cavs == 4
        # [32, 1456*num_input_frames, 4096]
        #assert False
        return point_features


    def generate_point_features(self, my_model_config, scene_point_feature_map, regression_map, classification_map, detection_box_score, object_features, active_agent_mask):
        #print("my_model_config['ego_only']: ", my_model_config['ego_only'])
        if my_model_config['ego_only']:
          cav_ids = ['ego']
        else:  
          cav_ids = ['ego', '1']  
        #print('cav_ids: ', cav_ids)

        scene_level_only = my_model_config['scene_level_only']
        object_level_only = my_model_config['object_level_only']
        if scene_level_only:
          scene_level_features = self.generate_scene_level_features(my_model_config, scene_point_feature_map, regression_map, classification_map, active_agent_mask)
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [batch_size, num_input_frames, num_cavs, num_tokens, feature_size=4096]
          # [32, 2, 2, 220, 4096] shallow
          # [32, 2, 2, 374, 4096] deep
          batch_size, num_input_frames, num_cavs, num_tokens, feature_size = scene_level_features.shape
          scene_level_features = scene_level_features.reshape([batch_size, num_input_frames * num_cavs * num_tokens, feature_size])
          #print('scene_level_features.shape: ', scene_level_features.shape)
          #assert False
          return scene_level_features
        elif object_level_only:
          # MY_DEBUG
          # only use object-level features
          object_level_features = self.generate_object_level_features(my_model_config, detection_box_score, object_features)
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, num_input_frames=2, num_cavs=2, num_tokens=50, feature_size=4096]
          batch_size, num_input_frames, num_cavs, max_num_boxes_per_cav, feature_size = object_level_features.shape
          object_level_features = object_level_features.reshape([batch_size, num_input_frames * num_cavs * max_num_boxes_per_cav, feature_size])
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, 2*2*50, 4096]
          #assert False
          return object_level_features
        else: # both scene and object
          scene_level_features = self.generate_scene_level_features(my_model_config, scene_point_feature_map, regression_map, classification_map, active_agent_mask)  
          #print('scene_level_features.shape: ', scene_level_features.shape)
          # [batch_size, num_input_frames, num_cavs, num_tokens, feature_size=4096]
          # [32, 2, 2, 220, 4096] shallow
          # [32, 2, 2, 374, 4096] deep
          # 
          # cobevt
          # [32, 2, 1, 288, 4096] # old shallow option 1
          # [32, 2, 1, 512, 4096] # new deep option 3
          # 
          # v2xreal
          # [32, 1, 4, 264, 4096]
          #assert False

          object_level_features = self.generate_object_level_features(my_model_config, detection_box_score, object_features)
          #print('object_level_features.shape: ', object_level_features.shape)
          # [32, num_input_frames=2, num_cavs=2, max_num_boxes_per_cav=50, feature_size=4096]
          # v2xreal [32, 1, 4, 100, 4096])

          # Concat scene-level tokens and object-level tokens for each cav
          point_features = self.concat_features_original(scene_level_features, object_level_features)

          #print('point_features.shape: ', point_features.shape)
          # no fusion, two cavs
          # [32, 540 * num_input_frames , 4096] # shallow
          # [32, 848 * num_input_frames , 4096] # deep
          # [batch_size, final_num_tokens, feature_size]
          #
          # cobevt, one merged
          # [32, 338 * num_input_frames , 4096] # shallow
          # [32, 562 * num_input_frames , 4096] # deep
          # [batch_size, final_num_tokens, feature_size]
          #
          # v2xreal [32, 1456 * num_input_frames, 4096]

          return point_features


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, my_model_config=None, scene_point_feature_map=None, regression_map=None, classification_map=None, detection_box_score=None, object_features=None, active_agent_mask=None
    ):


        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # MY_CODE
            # for v2v4real experiment
            # use point cloud feature map and new projector to generate
            # point cloud feature tokens with shape [batch_size, num_point_feature_tokens, 4096]
            # make sure the new projector is trainable and inside the model checkpoint
            # TODO: use config arg to determine whether in llava image code path or v2v4real point code path
            if scene_point_feature_map is not None or detection_box_score is not None:
                #print('my_model_config: ', my_model_config)
                #assert False
                #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
                point_features = self.generate_point_features(my_model_config, scene_point_feature_map, regression_map, classification_map,  detection_box_score, object_features, active_agent_mask)
                # and still call it image_features for now, 
                # so that we do not need to change the remaining code in this function
                image_features = point_features
                #print('point_features.shape: ', point_features.shape)
                # [2, 374, 4096]
                # [16, 1150, 4096]
                # [16, 50 , 4096]
                # [batch_size, num_tokens, feature_size]
            else: # regular llava code path using image
                #print('simple image encoder code') # here
                image_features = self.encode_images(images)
                #print('image_features.shape: ', image_features.shape)
                # [1, 576, 4096]
                # This assert is just to check whether we accidentally comes to llava image code path
                assert False
            

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        #print('input_ids: ', input_ids)
        #print('labels: ', labels)
        #print('attention_mask: ', attention_mask)
        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        #print('input_ids[0][:10]: ', input_ids[0][:10])
        # [tensor([    1,  -200,   447,   688,  3391,   373,   263,   521,  8233,  4315,
        # 10348,  2909,    13], device='cuda:0')]
        
        #print('labels[0][:10]: ', labels[0][:10])
        # [tensor([ -100,  -100,   447,   688,  3391,   373,   263,   521,  8233,  4315,
        # 10348,  2909,    13], device='cuda:0')]
        #assert False

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # MY_DEBUG
                # TODO: trace the text-only code path
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            #print('image_token_indices: ', image_token_indices)
            # [-1, 1, 13]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            #print('cur_input_ids_noim: ', cur_input_ids_noim)    
            # [tensor([1], device='cuda:0'), tensor([  447,   688,  3391,   373,   263,   521,  8233,  4315, 10348,  2909,
            # 13], device='cuda:0')]
            #print('cur_labels_noim: ', cur_labels_noim)
            #  [tensor([-100], device='cuda:0'), tensor([  447,   688,  3391,   373,   263,   521,  8233,  4315, 10348,  2909,
            # 13], device='cuda:0')]
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            #print('split_sizes: ', split_sizes)
            # [1, 11]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            #print('cur_input_embeds.shape: ', cur_input_embeds.shape)
            # [12, 4096]
            #print('len(cur_input_embeds_no_im): ', len(cur_input_embeds_no_im))
            #print('cur_input_embeds_no_im[0].shape: ', cur_input_embeds_no_im[0].shape)
            # [1, 4096]
            #print('cur_input_embeds_no_im[1].shape: ', cur_input_embeds_no_im[1].shape)
            # [11, 4096]
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                #print('cur_new_input_embeds[i].shape: ', cur_new_input_embeds[i].shape)
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    #print('cur_image_features.shape: ', cur_image_features.shape)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    #print('cur_new_input_embeds[-1].shape: ', cur_new_input_embeds[-1].shape)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            #print('cur_new_input_embeds.shape: ', cur_new_input_embeds.shape)
            # [1 + 576 + 11, 4096] == [588, 4096]
            cur_new_labels = torch.cat(cur_new_labels)
            #print('cur_new_labels.shape: ', cur_new_labels.shape)
            # [588]

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                #print('left')
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                #print('right') # here
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)


        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        #print('position_ids: ', position_ids) # None
        #print('attention_mask: ', attention_mask) # [1, 588]
        #print('past_key_values: ', past_key_values) # None
        #print('new_input_embeds.shape: ', new_input_embeds.shape) # [1, 588, 4096]
        #print('new_labels.shape: ', new_labels.shape) # [1, 588]
        #assert False
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # MY_CODE
        # TODO: trace this function and see 
        # if we need a similar one for point cloud feature

        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                # MY_CODE
                # currently not hit
                assert False
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
