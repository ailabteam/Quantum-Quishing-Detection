#!/usr/bin/env bash
# Phase 2: launch once, walk away. Three decision-critical batches.
#
#   STAGE 1  multi-seed decisive test (experiments_control): is the q6 VQC
#            robustness a real, seed-STABLE effect, and does it beat the
#            param-matched classical head at the SAME bottleneck width?
#            The sweep was erratic across configs (q6l2 great, q8l2 terrible),
#            which smells of VQC training instability, so single-seed cannot
#            decide. We train q6l2, q8l2 and classical b6 controls over 3 seeds;
#            the per-level acc_std in the report = the seed variance we need to see.
#   STAGE 2  bias audit + shortcut-subset eval (Cua 1)
#   STAGE 3  noise-aware defense (Cua 2)
#
# Each CLI tees its own log and regenerates the relevant REPORT.md. No `set -e`:
# a failed stage does not abort the rest, and every checkpoint is saved as it
# finishes, so a mid-run kill loses little.
#
# Run inside tmux:
#   bash revision/run_phase2.sh 2>&1 | tee phase2.log

DATA=data/raw/qrset
COMMON="--batch-size 128 --num-workers 4 --log-every 200"
CTRL=experiments_control

echo "############ STAGE 1: multi-seed decisive test (q6/q8 VQC vs classical b6) ############"
for s in 0 1 2; do
  python -m revision.train --model qresnet       --data $DATA --seed $s --epochs 3 --n-qubits 6 --n-layers 2 $COMMON --out $CTRL
  python -m revision.train --model qresnet       --data $DATA --seed $s --epochs 3 --n-qubits 8 --n-layers 2 $COMMON --out $CTRL
  python -m revision.train --model bottleneck_fc  --data $DATA --seed $s --epochs 3 --n-qubits 6 $COMMON --out $CTRL
  python -m revision.train --model mlp_head        --data $DATA --seed $s --epochs 3 --n-qubits 6 $COMMON --out $CTRL
done
# 1 perturbation seed is enough; the 3 TRAIN seeds give the stability error bars.
# --out MUST point inside $CTRL or it defaults to experiments_revision and gets clobbered.
python -m revision.robustness --exp-dir $CTRL --data $DATA --out $CTRL/robustness_raw.csv \
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
    --out experiments_revision/robustness_raw.csv \
    --pert-seeds 0,1,2 --noise-levels 0,0.06,0.08,0.10,0.12,0.14,0.16,0.20 --occ-levels 0,40,80,100 \
    --batch-size 128 --num-workers 4

echo "############ PHASE 2 DONE ############"
echo "Push results:"
echo "  git add experiments_control experiments_vqc_sens experiments_revision && git commit -m 'phase2: multi-seed control + audit + defense' && git push"
