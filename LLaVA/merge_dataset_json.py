import os
import json


def merge_v2vqa():
  data_file_names = [
    'v2v4real_3d_grounding_qa_dataset_v2s.json',      
    'v2v4real_3d_grounding_qa_dataset_v5bs.json',      
    'v2v4real_3d_grounding_qa_dataset_v4bs.json',      
    'v2v4real_3d_grounding_qa_dataset_v6sm3doublenew.json',      
    'v2v4real_3d_grounding_qa_dataset_v7sm100w6double.json'
  ]
  npy_save_path = '../DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy/co_llm/'

  all_data = []
  for data_file_name in data_file_names:
    data_path = os.path.join(npy_save_path, data_file_name)
    print('data_path: ', data_path)
    list_data_dict = json.load(open(data_path, "r"))    
    print('len(list_data_dict): ', len(list_data_dict))
    all_data += list_data_dict
  
  print('len(all_data): ', len(all_data))
  all_data_file = os.path.join(npy_save_path, 'v2v4real_3d_grounding_qa_dataset_all.json')
  with open(all_data_file, 'w') as f:
    json.dump(all_data, f)


def merge_v2vqa_graph(merged_dataset_name):
  if merged_dataset_name == 'allwc':
    data_file_names = [
      'v2v4real_3d_grounding_qa_dataset_nq1sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq2sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq3sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq5sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq6sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq7sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq8sm3w6dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json',      
    ]
  elif merged_dataset_name == 'all':
    data_file_names = [
      'v2v4real_3d_grounding_qa_dataset_nq1sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq2sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq3sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq3sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq5sm3w1d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq5sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq6sm3w1d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq6sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq7sm3w1d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq7sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq8sm3w6d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq8sm3w6dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq9sm3w6d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json',      
    ]
  elif merged_dataset_name == 'allwc9':
    data_file_names = [
      'v2v4real_3d_grounding_qa_dataset_nq1sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq2sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq3sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq5sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq6sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq7sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq8sm3w6dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq9sm3w6d.json',      
    ]
  elif merged_dataset_name == 'allsp3':
    data_file_names = [
      'v2v4real_3d_grounding_qa_dataset_nq1sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq2sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq3sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq5sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq6sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq7sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq8sm3w6dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_in_nq7sm3w1dc_out_nq9sm3w6dc.json',
    ]
  elif merged_dataset_name == 'allsp2':
    data_file_names = [
      'v2v4real_3d_grounding_qa_dataset_nq1sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq2sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq3sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq5sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq6sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq7sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq8sm3w6dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0d.json',      
    ]
  elif merged_dataset_name == 'spp4589':
    data_file_names = [
      'v2v4real_3d_grounding_qa_dataset_nq5sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq8sm3w6dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq9sm3w6dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0d.json',      
    ]
  elif merged_dataset_name == 'spp459':
    data_file_names = [
      'v2v4real_3d_grounding_qa_dataset_nq5sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_in_nq7sm3w1dc_out_nq9sm3w6dc.json',
    ]
  elif merged_dataset_name == 'spl':
    data_file_names = [
      'v2v4real_3d_grounding_qa_dataset_nq1sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq2sm3w0d.json',      
      'v2v4real_3d_grounding_qa_dataset_nq3sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq4sm3w0dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq5sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq6sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_nq7sm3w1dc.json',      
      'v2v4real_3d_grounding_qa_dataset_in_nq7sm3w1dc_out_nq9sm3w6dc.json',
    ]
  else:
    assert False
  
  for npy_save_path in ['../DMSTrack/V2V4Real/official_models/train_no_fusion_keep_all/npy/co_llm/', '../DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm/']:

    all_data = []
    for data_file_name in data_file_names:
      data_path = os.path.join(npy_save_path, data_file_name)
      print('data_path: ', data_path)
      list_data_dict = json.load(open(data_path, "r"))    
      #print('len(list_data_dict): ', len(list_data_dict))

      qa_source = data_file_name.split('_')[-1].split('.')[0]
      #print('qa_source: ', qa_source)
      qa_type_id = 10 + int(qa_source[2])
      #print('qa_type_id: ', qa_type_id)
      for data_sample in list_data_dict:
        #print('data_sample: ', data_sample)
        data_sample['qa_source'] = qa_source
        data_sample['qa_type_id'] = qa_type_id
        #print('data_sample: ', data_sample)

      all_data += list_data_dict
  
    print('len(all_data): ', len(all_data))
    all_data_file = os.path.join(npy_save_path, 'v2v4real_3d_grounding_qa_dataset_%s.json' % merged_dataset_name)
    print('all_data_file: ', all_data_file)
    with open(all_data_file, 'w') as f:
      json.dump(all_data, f)


def merge_v2xqa():
  data_file_names = [
    'v2xreal_3d_grounding_qa_dataset_v2s.json',      
    'v2xreal_3d_grounding_qa_dataset_v5bs.json',      
    'v2xreal_3d_grounding_qa_dataset_v4bs.json',      
    'v2xreal_3d_grounding_qa_dataset_v6sm3doublenew.json',      
    'v2xreal_3d_grounding_qa_dataset_v7sm3w6double.json'
  ]
  npy_save_path = '../V2X-Real/my_models/train_no_fusion_keep_all/npy/co_llm/'

  all_data = []
  for data_file_name in data_file_names:
    data_path = os.path.join(npy_save_path, data_file_name)
    print('data_path: ', data_path)
    list_data_dict = json.load(open(data_path, "r"))    
    print('len(list_data_dict): ', len(list_data_dict))
    all_data += list_data_dict
  
  print('len(all_data): ', len(all_data))
  all_data_file = os.path.join(npy_save_path, 'v2xreal_3d_grounding_qa_dataset_all.json')
  with open(all_data_file, 'w') as f:
    json.dump(all_data, f)


#merge_v2vqa()
#merge_v2xqa()

#merge_v2vqa_graph(with_context=False)
#merge_v2vqa_graph(with_context=True)

merge_v2vqa_graph(merged_dataset_name='all')
merge_v2vqa_graph(merged_dataset_name='allwc')
merge_v2vqa_graph(merged_dataset_name='allwc9')
merge_v2vqa_graph(merged_dataset_name='allsp3')
merge_v2vqa_graph(merged_dataset_name='allsp2')
merge_v2vqa_graph(merged_dataset_name='spp4589')
merge_v2vqa_graph(merged_dataset_name='spp459')
merge_v2vqa_graph(merged_dataset_name='spl')
