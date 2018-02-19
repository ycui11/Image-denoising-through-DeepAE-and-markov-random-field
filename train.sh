##########################################################################
# Author: Safa Messaoud                                                  #
# E-Mail: messaou2@illinois.edu                                          #
# Instituation: University of Illinois at Urbana-Champaign  			 #
# Course: ECE 544_na Fall 2017                                           #
# Date: Sptember 2017                                                    #
#                                                                        #
# Description: 															 #
#	Script to train 												     #
#																		 #
# usage:																 #
#  ./train.sh   														 #
##########################################################################


# Current directory.
CURRENT_DIR=$(pwd)

# Directory for saving and loading model checkpoints.
TRAIN_DIR="${CURRENT_DIR}/model"

# Number of training steps.
NB_STEPS=3000

# Create a directory for saving the model checkpoints.
mkdir -p ${TRAIN_DIR}

# script to be run.
BUILD_SCRIPT="train.py"

# Build the model.
python3 "${BUILD_SCRIPT}" \
--number_of_steps "${NB_STEPS}" \
--train_dir "${TRAIN_DIR}" \
