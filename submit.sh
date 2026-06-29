# nohup bash -lc '
#   source /home/i-liyuxin/miniconda3/etc/profile.d/conda.sh
#   conda activate py310
#   cd /home/i-liyuxin/Depression_TTS
#   PYTHONPATH=/home/i-liyuxin/Depression_TTS CUDA_VISIBLE_DEVICES=2 \
#   python -m matcha.train --config-name train_daic_utter \
#   trainer.max_epochs=400 logger=csv \
#   model.use_daic_conditions=true \
#   +model.pretrained_ckpt_path=/home/i-liyuxin/Depression_TTS/ckpts/matcha_vctk.ckpt

# ' > /home/i-liyuxin/Depression_TTS/train_mu_film_from_vctk_$(date +%F_%H-%M-%S).log 2>&1 & echo $!

#  +model.train_new_condition_modules_only=true \
#ablation
# nohup bash -lc '
#   source /home/i-liyuxin/miniconda3/etc/profile.d/conda.sh
#   conda activate py310
#   cd /home/i-liyuxin/Depression_TTS
#   PYTHONPATH=/home/i-liyuxin/Depression_TTS CUDA_VISIBLE_DEVICES=0 \
#   python -m matcha.train --config-name train_daic_utter \
#   trainer.max_epochs=200 logger=csv \
#   model.use_daic_conditions=true \
#   +model.pretrained_ckpt_path=/home/i-liyuxin/Depression_TTS/ckpts/matcha_vctk.ckpt
#   +data.depression_condition_type=phq_scalar +model.phq_scalar_hidden_dim=64

# ' > /home/i-liyuxin/Depression_TTS/train_mu_film_from_vctk_$(date +%F_%H-%M-%S).log 2>&1 & echo $!

#+data.depression_condition_type=phq_scalar +model.phq_scalar_hidden_dim=64
#+data.depression_condition_type=phq_level +model.phq_num_levels=5

# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/i-liyuxin/Depression_TTS \
# python -m matcha.train --config-name train_daic_utter \
#   trainer.precision=32 \
#   model.optimizer.lr=1e-5 \
#   trainer.gradient_clip_val=1.0 trainer.accumulate_grad_batches=2 \
#   model.use_adapter=true model.adapter_dim=16 model.adapter_dropout=0.2 \
#   model.use_decoupled_conditions=false \
#   model.decoder.use_spk_film_fullpath=false

nohup bash -lc '
  source /home/i-liyuxin/miniconda3/etc/profile.d/conda.sh
  conda activate py310
  cd /home/i-liyuxin/Depression_TTS
  PYTHONPATH=/home/i-liyuxin/Depression_TTS CUDA_VISIBLE_DEVICES=1 \
  python synthesis_unfreeze_all_sim_dep_slerp_new_daic_mask.py

' > /home/i-liyuxin/Depression_TTS/synthesis_unfreeze_all_sim4_$(date +%F_%H-%M-%S).log 2>&1 & echo $!

