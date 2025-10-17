import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

# MY_CODE
import numpy as np
import os


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        #print('prompt: ', prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        #print('input_ids: ', input_ids)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


        # MY_CODE
        cav_ids = ['ego', '1']
        if args.llm_data_path is not None:
            scene_point_feature_map_all = []
            for cav_id in cav_ids:
                scene_point_feature_map_file = os.path.join(args.llm_data_path, cav_id, '%04d_spatial_features_2d.npy' % args.global_timestamp_index)
                scene_point_feature_map = np.load(scene_point_feature_map_file)
                scene_point_feature_map = torch.from_numpy(scene_point_feature_map[0])
                scene_point_feature_map = scene_point_feature_map.to(model.device, dtype=torch.float16)
                scene_point_feature_map_all.append(scene_point_feature_map)
            scene_point_feature_map = torch.cat(scene_point_feature_map_all, dim=0)
            #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
            # (256*2, 50, 88)
            scene_point_feature_map = torch.unsqueeze(scene_point_feature_map, dim=0)
            #print('scene_point_feature_map.shape: ', scene_point_feature_map.shape)
            # (1, 256*2, 50, 88)
            #assert False
        else:
            scene_point_feature_map = None

        # regression_map
        if args.llm_data_path is not None:
            regression_map_all = []
            for cav_id in cav_ids:
                regression_map_file = os.path.join(args.llm_data_path, cav_id, '%04d_regression_map.npy' % args.global_timestamp_index)
                regression_map = np.load(regression_map_file)
                #print('regression_map.shape: ', regression_map.shape)
                # (1, 14, 50, 88)
                regression_map = torch.from_numpy(regression_map[0])
                #print('regression_map.shape: ', regression_map.shape)
                # [14, 50, 88]
                regression_map = regression_map.to(model.device, dtype=torch.float16)
                regression_map_all.append(regression_map)
            regression_map = torch.cat(regression_map_all, dim=0)
            #print('regression_map.shape: ', regression_map.shape)
            # [14 * 2, 50, 88]
            regression_map = torch.unsqueeze(regression_map, dim=0)
        else:
            regression_map = None

        # classification map
        if args.llm_data_path is not None:
            classification_map_all = []
            for cav_id in cav_ids:
                classification_map_file = os.path.join(args.llm_data_path, cav_id, '%04d_classification_map.npy' % args.global_timestamp_index)
                classification_map = np.load(classification_map_file)
                #print('classification_map.shape: ', classification_map.shape)
                # (1, 2, 50, 88)
                classification_map = torch.from_numpy(classification_map[0])
                #print('classification_map.shape: ', classification_map.shape)
                # [2, 50, 88]
                classification_map = classification_map.to(model.device, dtype=torch.float16)
                classification_map_all.append(classification_map)
            classification_map = torch.cat(classification_map_all, dim=0)
            #print('classification_map.shape: ', classification_map.shape)
            # [2 * 2, 50, 88]
            classification_map = torch.unsqueeze(classification_map, dim=0)
        else:
            classification_map = None

        if args.llm_data_path is not None:
            max_num_boxes_per_cav = 50
            detection_box_score_all = []
            for cav_id in cav_ids:
                detection_box_score = os.path.join(args.llm_data_path, cav_id, '%04d_detection_box_score.npy' % args.global_timestamp_index)
                detection_box_score = np.load(detection_box_score)
                detection_box_score = np.pad(
                    detection_box_score,
                    ((0, max_num_boxes_per_cav-detection_box_score.shape[0]), (0, 0)),
                    'constant', constant_values=(0, 0)
                )
                detection_box_score = torch.from_numpy(detection_box_score)
                detection_box_score = detection_box_score.to(model.device, dtype=torch.float16)
                #print('detection_box_score.shape: ', detection_box_score)
                detection_box_score_all.append(detection_box_score)

            detection_box_score = torch.cat(detection_box_score_all, dim=0)
            #print('detection_box_score.shape: ', detection_box_score.shape)
            # [100, 8]
            detection_box_score = torch.unsqueeze(detection_box_score, dim=0)
            #print('detection_box_score.shape: ', detection_box_score.shape)
            # [1, 100, 8]
            #assert False
        else:
            detection_box_score = None


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                # MY_CODE
                scene_point_feature_map=scene_point_feature_map,
                regression_map=regression_map,
                classification_map=classification_map,
                detection_box_score=detection_box_score)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    # MY_CODE
    parser.add_argument("--llm-data-path", type=str, default=None)
    parser.add_argument("--global-timestamp-index", type=int, default=0)

    args = parser.parse_args()
    main(args)
