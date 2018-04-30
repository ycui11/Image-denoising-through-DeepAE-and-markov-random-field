
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
