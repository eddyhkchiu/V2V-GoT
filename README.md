<div align="center">

# V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models and Graph-of-Thoughts

Hsu-kuang Chiu<sup>1,2</sup>, Ryo Hachiuma<sup>1</sup>, Chien-Yi Wang<sup>1</sup>, Yu-Chiang Frank Wang<sup>1</sup>, Min-Hung Chen<sup>1*</sup>, Stephen F. Smith<sup>2*</sup>

<sup>1</sup>NVIDIA, <sup>2</sup>Carnegie Mellon University, <sup>*</sup> equally advising



[project](https://eddyhkchiu.github.io/v2vgot.github.io/)  [arxiv](https://arxiv.org/abs/2509.18053)

<img src="images/V2V-LLM-Graph_Fig_1_0915_Fit.jpg" height=400px>

</div>

<b>V2V-GoT</b>: Graph-of-thoughts reasoning framework for vehicle-to-vehicle cooperative autonomous driving. All Connected Autonomous Vehicles (CAVs) share their perception features with the Multimodal Large Language Model (MLLM), as illustrated by the grey arrows. Any CAV can ask the MLLM to provide a suggested future trajectory or answer perception or prediction questions. The MLLM fuses the perception features from all CAVs and performs inference by following the graph-of-thoughts. If two QA nodes are connected by a directed edge in the graph, as illustrated by black arrows, the answer of the parent node QA is used as the input context of the child node QA.
  
## Overview

We propose the first graph-of-thoughts framework specifically designed for MLLM-based cooperative autonomous driving. Our graph-of-thoughts includes our proposed novel ideas of occlusion-aware perception and planning-aware prediction. We curate the <b>V2V-GoT-QA</b> dataset and develop the <b>V2V-GoT</b> model for training and testing the cooperative driving graph-of-thoughts. Our experimental results show that our method outperforms other baselines in cooperative perception, prediction, and planning tasks. For more details, please refer to our paper at <a href="https://arxiv.org/abs/2509.18053">arxiv</a>.
