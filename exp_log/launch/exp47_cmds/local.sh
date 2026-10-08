# LOCAL arm: sweep only; its remap stage already succeeded in job 11195547
python $CODE/scripts/coeff_sweep.py --run runs/exp46_cov_local_20261006-170041_j11180422 --snapshot runs/exp46_cov_local_20261006-170041_j11180422/snapshots/round_0100.pt --scorer clogp_residual_deco_strict --split test --coeffs=-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5 --max_prompts 60 --max_new_tokens 96
