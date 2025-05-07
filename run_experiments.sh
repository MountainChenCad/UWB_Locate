#!/bin.bash
# Shell script to automate UWB localization experiments

# --- Configuration ---
PYTHON_EXECUTABLE="python" # or python3, or full path
BASE_PROJECT_DIR=$(pwd) # Assumes script is run from project root

# Main results directory, timestamped
RESULTS_DIR="$BASE_PROJECT_DIR/paper_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file for the entire script execution
SCRIPT_LOG="$RESULTS_DIR/master_execution.log"
exec > >(tee -a "${SCRIPT_LOG}") 2>&1 # Redirect stdout and stderr to file and console

echo "================================================================================"
echo " UWB Localization Experiment Automation Script "
echo " Start Time: $(date)"
echo " Results will be saved in: $RESULTS_DIR"
echo "================================================================================"
echo

# Dataset directory (relative to project root)
DATASET_DIR="$BASE_PROJECT_DIR/data_set" # Preprocessed data will go here

## --- 1. Data Preprocessing ---
#echo "[STEP 1/5] Running Data Preprocessing..."
#PREPROCESS_LOG="$RESULTS_DIR/preprocess.log"
#if $PYTHON_EXECUTABLE preprocess_data.py > "$PREPROCESS_LOG" 2>&1; then
#    echo "Data preprocessing completed successfully."
#else
#    echo "ERROR: Data preprocessing failed. Check $PREPROCESS_LOG. Exiting."
#    exit 1
#fi
#echo "Preprocessing log: $PREPROCESS_LOG"
#echo "--------------------------------------------------------------------------------"
#echo

# --- 2. Least Squares (LS) Experiments ---
LS_RESULTS_SUBDIR="LS_Results"
LS_OUTPUT_DIR="$RESULTS_DIR/$LS_RESULTS_SUBDIR"
mkdir -p "$LS_OUTPUT_DIR"
echo "[STEP 2/5] Running Least Squares (LS) Experiments..."
LS_EXECUTION_LOG="$LS_OUTPUT_DIR/ls_execution.log"
if $PYTHON_EXECUTABLE LS.py --output_dir "$LS_OUTPUT_DIR" --dataset_base_dir "$DATASET_DIR" > "$LS_EXECUTION_LOG" 2>&1; then
    echo "LS experiments completed successfully."
else
    echo "ERROR: LS experiments failed. Check $LS_EXECUTION_LOG."
fi
echo "LS results are in: $LS_OUTPUT_DIR"
echo "LS execution log: $LS_EXECUTION_LOG"
echo "--------------------------------------------------------------------------------"
echo

# --- 3. Ranging Likelihood (RL) - Default (All 6 Features, Standard Train/Test) ---
RL_DEFAULT_SUBDIR="RL_DefaultAllFeatures_Standard"
RL_DEFAULT_OUTPUT_DIR="$RESULTS_DIR/$RL_DEFAULT_SUBDIR"
mkdir -p "$RL_DEFAULT_OUTPUT_DIR"
echo "[STEP 3/5] Running RL (Default - All 6 Features, Standard Train/Test)..."
RL_DEFAULT_EXECUTION_LOG="$RL_DEFAULT_OUTPUT_DIR/rl_default_execution.log"
ALL_6_FEATURES="energy,max_amp,rise_time,delay_spread,rms_delay,kurtosis"

if $PYTHON_EXECUTABLE RL.py \
    --output_base_dir "$RL_DEFAULT_OUTPUT_DIR" \
    --dataset_base_dir "$DATASET_DIR" \
    --features_to_use "$ALL_6_FEATURES" \
    --experiment_tag "RL_DefaultAllFeatures_Standard" > "$RL_DEFAULT_EXECUTION_LOG" 2>&1; then
    echo "RL (Default) experiments completed successfully."
else
    echo "ERROR: RL (Default) experiments failed. Check $RL_DEFAULT_EXECUTION_LOG."
fi
echo "RL (Default) results are in: $RL_DEFAULT_OUTPUT_DIR"
echo "RL (Default) execution log: $RL_DEFAULT_EXECUTION_LOG"
echo "--------------------------------------------------------------------------------"
echo

# --- 4. RL - Feature Combination Analysis (Standard Train/Test) ---
RL_FEATURE_COMBO_SUBDIR="RL_FeatureCombinations_Standard"
RL_FEATURE_COMBO_OUTPUT_DIR="$RESULTS_DIR/$RL_FEATURE_COMBO_SUBDIR"
mkdir -p "$RL_FEATURE_COMBO_OUTPUT_DIR"
echo "[STEP 4/5] Running RL - Feature Combination Analysis (Standard Train/Test)..."

# Define feature combinations to test (examples)
# You can expand this list significantly
declare -a FEATURE_COMBINATIONS=(
    "energy"
    "max_amp"
    "rise_time"
    "delay_spread"
    "rms_delay"
    "kurtosis"
    "energy,max_amp,rise_time"  # First 3
    "delay_spread,rms_delay,kurtosis" # Last 3
    "energy,delay_spread" # A pair example
    "$ALL_6_FEATURES" # Also run all features under this category for comparison if needed, though step 3 covers it.
)

for features_csv in "${FEATURE_COMBINATIONS[@]}"; do
    # Create a clean tag from the feature list for filenames
    feature_tag=$(echo "$features_csv" | tr ',' '_')
    echo "  Running RL with features: [$features_csv] (Tag: $feature_tag)"

    CURRENT_COMBO_LOG="$RL_FEATURE_COMBO_OUTPUT_DIR/rl_combo_${feature_tag}_execution.log"

    if $PYTHON_EXECUTABLE RL.py \
        --output_base_dir "$RL_FEATURE_COMBO_OUTPUT_DIR" \
        --dataset_base_dir "$DATASET_DIR" \
        --features_to_use "$features_csv" \
        --experiment_tag "RL_Combo_${feature_tag}" > "$CURRENT_COMBO_LOG" 2>&1; then
        echo "  RL with features [$features_csv] completed."
    else
        echo "  ERROR: RL with features [$features_csv] failed. Check $CURRENT_COMBO_LOG."
    fi
done
echo "RL Feature Combination Analysis completed."
echo "RL Feature Combination results are in: $RL_FEATURE_COMBO_OUTPUT_DIR"
echo "--------------------------------------------------------------------------------"
echo

# --- 5. RL - Cross-Environment Validation (Leave-One-Out, All 6 Features) ---
RL_CROSSVALID_SUBDIR="RL_CrossValidation_AllFeatures"
RL_CROSSVALID_OUTPUT_DIR="$RESULTS_DIR/$RL_CROSSVALID_SUBDIR"
mkdir -p "$RL_CROSSVALID_OUTPUT_DIR"
echo "[STEP 5/5] Running RL - Cross-Environment Validation (Leave-One-Out, All 6 Features)..."

ENVIRONMENTS_ARRAY=("environment0" "environment1" "environment2" "environment3")

for test_env_idx in "${!ENVIRONMENTS_ARRAY[@]}"; do
    TEST_ENV_NAME="${ENVIRONMENTS_ARRAY[$test_env_idx]}"
    TRAIN_ENVS_LIST=()
    for train_env_idx in "${!ENVIRONMENTS_ARRAY[@]}"; do
        if [ "$train_env_idx" -ne "$test_env_idx" ]; then
            TRAIN_ENVS_LIST+=("${ENVIRONMENTS_ARRAY[$train_env_idx]}")
        fi
    done
    TRAIN_ENVS_CSV=$(IFS=,; echo "${TRAIN_ENVS_LIST[*]}") # Join with comma

    echo "  Cross-Validation: Training on [$TRAIN_ENVS_CSV], Testing on [$TEST_ENV_NAME]"

    CURRENT_CROSSVALID_LOG="$RL_CROSSVALID_OUTPUT_DIR/rl_crossvalid_train_${TRAIN_ENVS_CSV//,/_}_test_${TEST_ENV_NAME}_execution.log"
    EXPERIMENT_NAME_TAG="RL_CrossValid_TestOn_${TEST_ENV_NAME}"

    if $PYTHON_EXECUTABLE RL.py \
        --output_base_dir "$RL_CROSSVALID_OUTPUT_DIR" \
        --dataset_base_dir "$DATASET_DIR" \
        --features_to_use "$ALL_6_FEATURES" \
        --train_envs "$TRAIN_ENVS_CSV" \
        --test_env "$TEST_ENV_NAME" \
        --experiment_tag "$EXPERIMENT_NAME_TAG" > "$CURRENT_CROSSVALID_LOG" 2>&1; then
        echo "  Cross-validation for Test Env [$TEST_ENV_NAME] completed."
    else
        echo "  ERROR: Cross-validation for Test Env [$TEST_ENV_NAME] failed. Check $CURRENT_CROSSVALID_LOG."
    fi
done
echo "RL Cross-Environment Validation completed."
echo "RL Cross-Validation results are in: $RL_CROSSVALID_OUTPUT_DIR"
echo "--------------------------------------------------------------------------------"
echo

echo "================================================================================"
echo " All Automated Experiments Finished."
echo " End Time: $(date)"
echo " Master log: $SCRIPT_LOG"
echo " Check subdirectories in $RESULTS_DIR for detailed outputs and logs."
echo "================================================================================"
