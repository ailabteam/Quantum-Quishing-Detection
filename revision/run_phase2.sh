#!/usr/bin/env bash
# Phase 2: launch once, walk away. Runs the three decision-critical batches:
#   STAGE 1  bottleneck-width controls  -> is the q6 robustness from the VQC or
#            just the wider 512->6 bottleneck? (the decisive quantum test, Cua 3)
#   STAGE 2  bias audit + shortcut-subset eval (Cua 1)
#   STAGE 3  noise-aware defense (Cua 2)
#
# Each CLI tees its own log under experiments_*/logs/ and regenerates the relevant
# REPORT.md. No `set -e`: a failure in one stage does not abort the others, and
# every checkpoint is saved as it finishes, so a mid-run kill loses little.
#
# Run inside tmux:
#   bash revision/run_phase2.sh 2>&1 | tee phase2.log

DATA=data/raw/qrset
COMMON="--batch-size 128 --num-workers 4 --log-every 200"

echo "############ STAGE 1: bottleneck-width controls (decisive quantum test) ############"
for b in 6 8; do
  python -m revision.train --model bottleneck_fc --data $DATA --seed 0 --epochs 3 --n-qubits $b $COMMON --out experiments_vqc_sens
  python -m revision.train --model mlp_head      --data $DATA --seed 0 --epochs 3 --n-qubits $b $COMMON --out experiments_vqc_sens
done
# evaluate VQC configs AND the matched classical controls together (1 seed = screening)
python -m revision.robustness --exp-dir experiments_vqc_sens --data $DATA \
    --pert-seeds 0 --noise-levels 0,0.08,0.10,0.12,0.14,0.16,0.20 --occ-levels 0,100 \
    --batch-size 128 --num-workers 4

echo "############ STAGE 2: bias audit + shortcut-subset eval (Cua 1) ############"
python -m revision.audit_dataset --data $DATA --out experiments_revision/audit --limit-per-class 10000
SUBSET=experiments_revision/audit/length_matched_subset.csv
if [ -f "$SUBSET" ]; then
  python -m revision.eval_subset --exp-dir experiments_revision --data $DATA --subset "$SUBSET"
else
  echo "[phase2] no length_matched_subset.csv produced (decode may have failed); skipping subset eval"
fi

echo "############ STAGE 3: noise-aware defense (Cua 2) ############"
python -m revision.train --model classic_fc --data $DATA --seed 0 --epochs 5 --noise-aware --noise-sigma-max 0.15 $COMMON
python -m revision.train --model mlp_head    --data $DATA --seed 0 --epochs 5 --noise-aware --noise-sigma-max 0.15 $COMMON
# regenerate the MAIN report with clean + noise-aware curves side by side
python -m revision.robustness --exp-dir experiments_revision --data $DATA \
    --pert-seeds 0,1,2 --noise-levels 0,0.06,0.08,0.10,0.12,0.14,0.16,0.20 --occ-levels 0,40,80,100 \
    --batch-size 128 --num-workers 4

echo "############ PHASE 2 DONE ############"
echo "Push results:"
echo "  git add experiments_vqc_sens experiments_revision && git commit -m 'phase2: controls + audit + defense' && git push"
