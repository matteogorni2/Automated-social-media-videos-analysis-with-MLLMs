#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${workdir}/output_logs/qwen3vl_infer_%j.out
#SBATCH --error=${workdir}/error_logs/qwen3vl_infer_%j.err
#SBATCH --ntasks=${ntasks}
#SBATCH --partition=${partition}
#SBATCH --gres=${gres}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem=${mem}
#SBATCH --time=${time}

# ----------------------
# Paths
# ----------------------
WORKDIR=${workdir}
MODEL_PATH=${model_path}
SIF_PATH=${sif_path}
VIDEO_PATH=${video_path}
SCRIPT_PATH=${script_path}
MANIFEST_PATH=${manifest_absolute_path}   # This will be substituted by submit_from_manifest.py

# ----------------------
# Logging info
# ----------------------
echo "======================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Start time:    $(date)"
echo "Manifest path: $MANIFEST_PATH"
echo "======================================"

# ----------------------
# Run inference inside Singularity container
# ----------------------

if [ "$ntasks" -gt 1 ]; then
    ACCELERATE_ARGS="--multi_gpu --num_processes $ntasks"
else
    ACCELERATE_ARGS="--num_processes 1"
fi

singularity exec --nv \
    --bind /nfsd/:/outer_directory \
    "$SIF_PATH" \
    accelerate launch $ACCELERATE_ARGS \
    "$SCRIPT_PATH" \
    --manifest "$MANIFEST_PATH"

# ----------------------
# End logging
# ----------------------
echo "======================================"
echo "End time: $(date)"
echo "======================================"
#
