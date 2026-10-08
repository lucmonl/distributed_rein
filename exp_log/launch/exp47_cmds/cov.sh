# COV arm: both diagnostics (its remap never ran -- the earlier job was cancelled for a bad CMD)
python $CODE/eval_direction.py --run runs/exp46_cov_borrow_l1_20261006-154820_j11180420 --snapshot runs/exp46_cov_borrow_l1_20261006-154820_j11180420/snapshots/round_0100.pt \
  --scorer clogp_residual_deco_strict --split test --posthoc_remap --remap_split dev \
  --remap_grid 11 --remap_prompts 50 --max_prompts 150 --max_new_tokens 96 --suffix remapdiag
python $CODE/scripts/coeff_sweep.py --run runs/exp46_cov_borrow_l1_20261006-154820_j11180420 --snapshot runs/exp46_cov_borrow_l1_20261006-154820_j11180420/snapshots/round_0100.pt --scorer clogp_residual_deco_strict --split test --coeffs=-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5 --max_prompts 60 --max_new_tokens 96
