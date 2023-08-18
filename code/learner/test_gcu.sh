#!/bin/bash

unset TF_XLA_FLAGS
unset XLA_FLAGS
unset ENFLAME_COMPILE_OPTIONS_HLIR
unset ENFLAME_LOG_LEVEL
unset ENFLAME_LOG_DEBUG_MOD
unset TOPS_EXE_CACHE_DISABLE
unset TOPS_EXE_CACHE_PATH

# common setting
VISIBLE_DEVICE=4,5,6,7
#export ENFLAME_DEVICE_MODE=ONEDEVICE_EX
export ENFLAME_ENABLE_TF32=true
export TOPS_VISIBLE_DEVICES=$VISIBLE_DEVICE

export OMP_NUM_THREADS=5
export ECCL_MAX_NCHANNELS=2
export ECCL_RUNTIME_3_0_ENABLE=true


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# dump config
#export COMPILE_OPTIONS_MLIR_DBG="-pass-timing -pass-statistics -mlir-elide-elementsattrs-if-larger=100 --print-ir-before-all=true --print-ir-after-all=true -log-output-path=$DIR/ir_dump/"

#### tensorflow
#export TF_XLA_FLAGS="--tf_xla_auto_jit=-1 --tf_xla_min_cluster_size=4"
#export XLA_FLAGS=" --xla_dump_hlo_as_text --xla_dump_to=hlo_dump --xla_dump_hlo_pass_re='.*'"
# 1C(comment to auto4C)
#export ENFLAME_COMPILE_OPTIONS_HLIR="hlir-training-pipeline{dynamic-shape=false}"
#export ENFLAME_LOG_LEVEL=DEBUG
#export ENFLAME_LOG_DEBUG_MOD=OP

#export ENFLAME_COMPILE_OPTIONS_HLIR="hlir-training-pipeline{op-key=pavo dynamic-shape=false tensor-split=false disable-passes=sink-pass,sink-reshape,broadcast-hoist,hlir-cse,algebraic-simplify-div-mul,hlir-merge-broadcast,postpone-broadcast,op-compose,pred-convert-select-pass,hlir-fusion-elementwise,hlir-union-elementwise-fusion,Canonicalizer,BroadcastFoldingPass,hlir-fusion,pred-convert-select-pass}"

#### torch
<< BLOCK
export ENFLAME_LOG_LEVEL=DEBUG
export ENFLAME_LOG_DEBUG_MOD=OP,LAZYNODE,ECCL # LAZYNODE print too much
export TOPS_EXE_CACHE_DISABLE=false   # default:  true
export TOPS_EXE_CACHE_PATH=$DIR/pt_exec_cache/
# 1C(comment to auto4C)
#export RT_ENABLE_DEFAULT_4C_SETTINGS="false"
BLOCK

cd /aiarena/code/learner/

#topsprof --profile-from-start off --debug --print-app-log --enable-activities "$VISIBLE_DEVICE/general/operator|$VISIBLE_DEVICE/*/sip/*|api trace/*/*" --export-visual-profiler ./vpd python3.8 train.py --single_test

#topsprof --profile-from-start off --debug --print-app-log --enable-activities "*/general/operator|*/*/sip/*|api trace/*/*" --export-visual-profiler ./vpd \
nohup torchrun --nproc_per_node=4 train.py --single_test >> ${LOG_DIR}/train.log 2>&1 &

