conda activate dmstrack

# for gcp
#export MY_HOME=${HOME}
# for psc
export MY_HOME=${PROJECT}

#export MY_PROJECT=my_co_llm_driver
export MY_PROJECT=my_v2vllm_graph/DMSTrack

export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/${MY_PROJECT}/AB3DMOT
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/${MY_PROJECT}/AB3DMOT/Xinshuo_PyToolbox
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/${MY_PROJECT}/V2V4Real
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/${MY_PROJECT}/DMSTrack
export PYTHONPATH=${PYTHONPATH}:${MY_HOME}/${MY_PROJECT}/

echo $PYTHONPATH

export JUPYTER_PATH=${PYTHONPATH}
