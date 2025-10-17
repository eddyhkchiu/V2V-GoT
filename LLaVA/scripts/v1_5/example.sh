eval "$(conda shell.bash hook)"
pwd
ls
cd /home/hsukuangc/my_v2vllm_graph/LLaVA
pwd
ls
which conda
#conda activate llava
echo $CONDA_DEFAULT_ENV
source /root/miniconda3/bin/activate llava
nvidia-smi
export PYTHONPATH=/home/hsukuangc/my_v2vllm_graph/LLaVA:$PYTHONPATH
echo $PYTHONPATH
echo $CONDA_DEFAULT_ENV
