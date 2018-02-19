#########################################################################################################################
# Author: Safa Messaoud                                                                                                 #
# E-Mail: messaou2@illinois.edu                                                                                         #
# Instituation: University of Illinois at Urbana-Champaign  															#
# Course: ECE 544_na Fall 2017                                                            								#
# Date: July 2017                                                                                                   	#
#                                                                                                                       #
# Description: 																											#
#	Script to run evaluation 																							#
#																														#
# usage:																												#
#  ./evaluate.sh   																										#
#########################################################################################################################


# Current directory.
CURRENT_DIR=$(pwd)

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=""

# Directory containing the checkpoints.
CHECKPOINT_DIR="${CURRENT_DIR}/model"

# Interval between evaluation runs.
EVAL_INTERVAL_SEC=60

# Minimum global step to run evaluation.
MIN_GlOBAL_STEP=50

# Number of evaluation examples.
NUM_EVAL_EXAMPLES=1000

# Script to be run.
BUILD_SCRIPT="evaluation.py"


# Run the evaluation script. This will run in a loop, periodically loading the
# latest model checkpoint file and computing evaluation metrics.
python3 "${BUILD_SCRIPT}" \
  --eval_interval_sec="${EVAL_INTERVAL_SEC}"\
  --min_global_step="${MIN_GlOBAL_STEP}"\
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  --num_eval_examples="${NUM_EVAL_EXAMPLES}"\



