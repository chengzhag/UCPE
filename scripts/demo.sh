source scripts/set_Wan2.1-T2V-1.3B.sh
export WANDB_MODE=disabled

export WANDB_RUN_ID=6wodf04s
export PL_PREDICT__DATA="DemoDataModule"
export PL_PREDICT__MODEL__CKPT_PATH="logs/6wodf04s/checkpoints/pytorch_model.bin"
# export PL_PREDICT__MODEL__NUM_PREDICT=5  # Number of predictions to generate per input

python -m src.main predict --data.input_file="demo/lens.json"
python -m src.main predict --data.input_file="demo/pose.json"
python -m src.main predict --data.input_file="demo/teaser.json"
