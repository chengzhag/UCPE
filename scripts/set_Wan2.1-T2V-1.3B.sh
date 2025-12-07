for STAGE in FIT VALIDATE TEST PREDICT; do
  export PL_${STAGE}__CKPT_PATH="last"
  export PL_${STAGE}__DATA__DATA_ROOT="data/UCPE"
  export PL_${STAGE}__MODEL__FPS=16
  export PL_${STAGE}__MODEL__HEIGHT=480
  export PL_${STAGE}__MODEL__WIDTH=832
  export PL_${STAGE}__MODEL__NUM_FRAMES=81
  export PL_${STAGE}__MODEL__MODEL_ID="Wan-AI/Wan2.1-T2V-1.3B"
done

export HF_HUB_OFFLINE=1