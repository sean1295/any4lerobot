export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python libero_h5.py \
    --src-paths /scratch/dcs3zc/LIBERO/libero/datasets/libero_90_hf/ \
    --output-path /scratch/dcs3zc/LIBERO/libero/datasets \
    --executor local \
    --tasks-per-job 3 \
    --workers 10