#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — holistic executor for the WiFi-CSI activity pipeline
# =============================================================================
#
# One ordered runner for every chunk built so far. Each "stage" below states
# what it is, what (if anything) it runs, and documents its tunable variables
# inline. This is a personal convenience script — not everything is exposed as
# a flag; expose more per-stage variables as each chunk needs them.
#
# USAGE
#   ./run_pipeline.sh                 # run the DEFAULT_STAGES (safe reproduction)
#   ./run_pipeline.sh all             # run every stage in order (incl. heavy ones)
#   ./run_pipeline.sh preprocess evaluate diagnose   # run only these, in this order
#   ./run_pipeline.sh list            # print the stage list and exit
#
# NOTES
#   * Stages run in DEPENDENCY ORDER (see the table in the dispatch section).
#   * Heavy stages (train, sweep, train_ntu) are EXCLUDED from the no-arg
#     default so a plain run doesn't retrain or overwrite a checkpoint.
#     Run them explicitly:  ./run_pipeline.sh train | sweep | train_ntu
#   * window_size / stride are deliberately surfaced but NOT changed here.
#     Per the project brief, ask the owner before moving them — they're easy
#     levers that can mask root causes (see docs/diagnostics.md).
#   * `src/*.py` run via `python -m src.<name>`; `scripts/*.py` run directly.
#
# Extend: add a `stage_<name>() { ... }` function, then add its name to
# ALL_STAGES (and DEFAULT_STAGES if it should run by default).
# =============================================================================

set -euo pipefail

# --- repo root + conda env ---------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-wifisense}" # override with: CONDA_ENV=other ./run_pipeline.sh
for _c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/miniconda3 /opt/conda; do
  if [ -f "$_c/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$_c/etc/profile.d/conda.sh"
    break
  fi
done
conda activate "$CONDA_ENV"

banner() {
  echo ""
  echo "============================================================================="
  echo ">>> $1"
  echo "============================================================================="
}

# =============================================================================
# STAGES
# =============================================================================

# --- setup: environment & data sanity check ---------------------------------
stage_verify() {
  banner "SETUP — verify environment & data (verify_setup.py)"
  # What: checks the conda env, key imports, and that raw datasets are present.
  # Runs: verify_setup.py   (no arguments)
  python verify_setup.py
}

# --- chunk 1: dataset characterization ---------------------------------------
stage_explore() {
  banner "CHUNK 1 — explore datasets (scripts/explore_data.py)"
  # What: prints per-split/-class stats for UT-HAR + NTU-Fi and dumps 5
  #       sample clips per class under data/samples/.
  # Runs: scripts/explore_data.py   (no arguments)
  # NOTE: loads the full NTU-Fi train split (~2.5 GB RAM, slow).
  python scripts/explore_data.py
}

# --- chunk 2: preprocessing --------------------------------------------------
stage_preprocess() {
  banner "CHUNK 2 — preprocess UT-HAR (scripts/preprocess_data.py)"
  # What: amplitude -> hampel -> median -> per-sample z-score over all splits.
  # Runs: scripts/preprocess_data.py   (no arguments)
  # Out:  data/processed/ut_har/ut_har.npz  (consumed by train/visualize/evaluate)
  #       figures/preprocessing/ut_har_before_after.png
  python scripts/preprocess_data.py
}

# --- chunk 3: per-class visual sanity grids ----------------------------------
stage_visualize() {
  banner "CHUNK 3 — per-class CSI grids (scripts/visualize_classes.py)"
  # Variables ---------------------------------------------------------------
  local VIZ_SEED=0  # RNG seed for which 3 samples/class are shown
  local VIZ_FS=1000 # assumed CSI packet rate (Hz); only rescales Doppler axis
  # What: amplitude-heatmap grid + Doppler-spectrogram grid from the
  #       PREPROCESSED train set (needs chunk 2 output).
  # Out:  figures/class_grid.png, figures/doppler_grid.png
  python scripts/visualize_classes.py --seed "$VIZ_SEED" --fs "$VIZ_FS"
}

# --- chunk 4: single training run (HEAVY) ------------------------------------
stage_train() {
  banner "CHUNK 4 — train one config (python -m src.train)"
  # Variables (BiLSTM config; defaults match the frozen model) --------------
  local TRAIN_SEED=42     # RNG seed
  local TRAIN_LR=1e-3     # Adam learning rate
  local TRAIN_HIDDEN=64   # LSTM hidden units per direction
  local TRAIN_LAYERS=2    # stacked LSTM layers
  local TRAIN_DROPOUT=0.3 # dropout (inter-layer + pre-classifier)
  local TRAIN_EPOCHS=80   # max epochs (early stopping may stop sooner)
  local TRAIN_BATCH=64    # batch size
  local TRAIN_PATIENCE=15 # early-stop patience (epochs w/o val-acc gain)
  # What: fits weights on the preprocessed train split, early-stops on val.
  # Out:  runs/<timestamp>/{best.pt, metrics.json, training_curves.png}
  #       (does NOT touch runs/best_bilstm.pt — promote via sweep or by hand)
  python -m src.train \
    --model bilstm --seed "$TRAIN_SEED" --lr "$TRAIN_LR" \
    --hidden "$TRAIN_HIDDEN" --layers "$TRAIN_LAYERS" --dropout "$TRAIN_DROPOUT" \
    --epochs "$TRAIN_EPOCHS" --batch-size "$TRAIN_BATCH" --patience "$TRAIN_PATIENCE"
}

# --- chunk 4: multi-seed robustness sweep (HEAVY) ----------------------------
stage_sweep() {
  banner "CHUNK 4 — multi-seed sweep (scripts/sweep.py)"
  # Variables ---------------------------------------------------------------
  local SWEEP_SEEDS="42 43 44 45 46" # seeds to run the fixed config under
  local SWEEP_LR=1e-3
  local SWEEP_HIDDEN=64
  local SWEEP_LAYERS=2
  local SWEEP_DROPOUT=0.3
  local SWEEP_EPOCHS=80
  local SWEEP_PROMOTE=1 # 1 = copy best-by-val checkpoint to runs/best_bilstm.pt
  # What: runs the same config across seeds, reports val acc mean±std, and
  #       (if SWEEP_PROMOTE=1) freezes the winner as the stable checkpoint.
  # Out:  runs/sweep_<timestamp>/seed_<s>/..., summary.json
  local promote_flag=""
  [ "$SWEEP_PROMOTE" = "1" ] && promote_flag="--promote"
  # shellcheck disable=SC2086
  python scripts/sweep.py --seeds $SWEEP_SEEDS \
    --lr "$SWEEP_LR" --hidden "$SWEEP_HIDDEN" --layers "$SWEEP_LAYERS" \
    --dropout "$SWEEP_DROPOUT" --epochs "$SWEEP_EPOCHS" $promote_flag
}

# --- chunk 5: freeze & evaluate once on test ---------------------------------
stage_evaluate() {
  banner "CHUNK 5 — evaluate on test (python -m src.evaluate)"
  # Variables ---------------------------------------------------------------
  local EVAL_CKPT="runs/best_bilstm.pt" # frozen checkpoint to score
  local EVAL_SPLIT="test"               # train | val | test (test = the one peek)
  # What: overall acc, macro F1, per-class report on the frozen checkpoint.
  # Out:  figures/confusion_matrix.png, predictions_test.csv, eval_metrics_test.json
  python -m src.evaluate --checkpoint "$EVAL_CKPT" --split "$EVAL_SPLIT"
}

# --- chunk 6: stitch a continuous capture ------------------------------------
stage_capture() {
  banner "CHUNK 6 — build continuous capture (scripts/build_continuous_capture.py)"
  # Variables ---------------------------------------------------------------
  local CAPTURE_N=8          # number of clips stitched end-to-end
  local CAPTURE_SEED=0       # selection RNG seed (round-robin across classes)
  local CAPTURE_SPLIT="test" # which split to draw clips from
  # What: concatenates N raw test clips into one stream + ground-truth timeline.
  # Out:  data/continuous/synthetic_capture.npz
  python scripts/build_continuous_capture.py \
    --n "$CAPTURE_N" --seed "$CAPTURE_SEED" --split "$CAPTURE_SPLIT"
}

# --- chunk 6: milestone 3-panel visualization --------------------------------
stage_finalviz() {
  banner "CHUNK 6 — final visualization (scripts/final_visualization.py)"
  # Variables ---------------------------------------------------------------
  local FV_WINDOW=250  # sliding-window length  (== segment_len; ASK before changing)
  local FV_STRIDE=25   # sliding-window hop      (ASK before changing)
  local FV_DPI=200     # output figure DPI
  local FV_DEVICE=auto # auto | cpu | cuda
  # What: sliding-window inference over the capture -> 3-panel figure
  #       (CSI heatmap / class-prob bands / ground truth), prints per-window acc.
  # Out:  figures/final_visualization.png
  python scripts/final_visualization.py \
    --window-size "$FV_WINDOW" --stride "$FV_STRIDE" --dpi "$FV_DPI" --device "$FV_DEVICE"
}

# --- chunk 7: accuracy diagnosis ---------------------------------------------
stage_diagnose() {
  banner "CHUNK 7 — diagnose accuracy (scripts/diagnose_accuracy.py)"
  # Variables ---------------------------------------------------------------
  local DIAG_WINDOW=250  # must match the capture's stride/window story (ASK before changing)
  local DIAG_STRIDE=25   # (ASK before changing)
  local DIAG_DEVICE=auto # auto | cpu | cuda
  # What: separates genuine model error / boundary effects / preprocessing edge
  #       effects; clean-test vs in-segment vs continuous accuracy + verdict.
  # Out:  figures/{per_class_confusion_continuous,window_position_accuracy,
  #       lie_down_failure_diagnosis}.png, figures/diagnostics_summary.json
  python scripts/diagnose_accuracy.py \
    --window-size "$DIAG_WINDOW" --stride "$DIAG_STRIDE" --device "$DIAG_DEVICE"
}

# --- chunk 8: post-processing comparison (stage 10) --------------------------
stage_postprocess() {
  banner "CHUNK 8 — compare post-processing (scripts/compare_postprocessing.py)"
  # Variables ---------------------------------------------------------------
  local PP_WINDOW=250  # sliding-window length  (== segment_len; ASK before changing)
  local PP_STRIDE=25   # sliding-window hop      (ASK before changing)
  local PP_K=""        # smoothing window in windows; empty = ~half a segment (ASK before tuning)
  local PP_LAPLACE=1.0 # Laplace alpha for the HMM transition matrix
  local PP_DPI=200     # smoothed-figure DPI
  local PP_DEVICE=auto # auto | cpu | cuda
  # What: runs moving_average / majority_vote / hmm_decode over the capture,
  #       reports per-window + in-segment accuracy and transition rate, and
  #       re-renders the milestone figure with a raw-vs-best probability panel.
  # Out:  notes/postprocessing.md, figures/final_visualization_smoothed.png
  # Like finalviz/diagnose this is light and overwrites its own outputs each
  # run (no skip-if-exists), so it stays in the default reproduction set.
  local k_flag=""
  [ -n "$PP_K" ] && k_flag="--k $PP_K"
  # shellcheck disable=SC2086
  python scripts/compare_postprocessing.py \
    --window-size "$PP_WINDOW" --stride "$PP_STRIDE" $k_flag \
    --laplace "$PP_LAPLACE" --dpi "$PP_DPI" --device "$PP_DEVICE"
}

# --- chunk 9: NTU-Fi → common (250,90) representation ------------------------
stage_preprocess_ntu() {
  banner "CHUNK 9 — preprocess NTU-Fi to UT-HAR format (scripts/preprocess_ntu_fi.py)"
  # What: bilinearly resize every NTU-Fi (342,2000) sample to UT-HAR's
  #       (250,90), run the IDENTICAL UT-HAR pipeline, and carve a stratified
  #       10% val split out of train (NTU-Fi ships no val).
  # Out:  data/processed/ntu_fi/ntu_fi.npz (carries class_names)
  #       figures/preprocessing/ntu_fi_resize.png
  # Loads .mat files lazily, so peak RAM stays small (~15 s).
  python scripts/preprocess_ntu_fi.py
}

# --- chunk 9: train a fresh BiLSTM on NTU-Fi (HEAVY) -------------------------
stage_train_ntu() {
  banner "CHUNK 9 — train NTU-Fi BiLSTM (scripts/sweep.py, identical chunk-5 recipe)"
  # Variables (identical to the UT-HAR sweep; only --data/--promote-to differ) -
  local SWEEP_SEEDS="42 43 44 45 46"
  local SWEEP_LR=1e-3
  local SWEEP_HIDDEN=64
  local SWEEP_LAYERS=2
  local SWEEP_DROPOUT=0.3
  local SWEEP_EPOCHS=80
  local SWEEP_PROMOTE=1 # 1 = copy best-by-val checkpoint to runs/best_bilstm_ntu.pt
  # What: same recipe as chunk 5 on the NTU-Fi npz; promotes the best-by-val
  #       seed to the stable NTU checkpoint.
  # Out:  runs/sweep_<ts>_ntu/..., runs/best_bilstm_ntu.pt
  local promote_flag=""
  [ "$SWEEP_PROMOTE" = "1" ] && promote_flag="--promote --promote-to runs/best_bilstm_ntu.pt"
  # shellcheck disable=SC2086
  python scripts/sweep.py --seeds $SWEEP_SEEDS \
    --lr "$SWEEP_LR" --hidden "$SWEEP_HIDDEN" --layers "$SWEEP_LAYERS" \
    --dropout "$SWEEP_DROPOUT" --epochs "$SWEEP_EPOCHS" \
    --data data/processed/ntu_fi/ntu_fi.npz --tag ntu $promote_flag
}

# --- chunk 9: characterize the UT-HAR ↔ NTU-Fi domain gap --------------------
stage_domainshift() {
  banner "CHUNK 9 — domain-shift matrix (cross_dataset_eval.py + domain_shift_matrix.py)"
  # What: NTU-Fi in-domain test eval + both zero-shot cross-domain directions
  #       + the assembled 2×2 figure. Uses the frozen UT-HAR and NTU-Fi
  #       checkpoints (no training), so it stays in the default reproduction.
  # Out:  figures/ntu/eval_metrics_test.json,
  #       figures/cross_{uthar_on_ntu,ntu_on_uthar}_*.{png,json},
  #       figures/domain_shift_matrix.png
  python -m src.evaluate --checkpoint runs/best_bilstm_ntu.pt \
    --data data/processed/ntu_fi/ntu_fi.npz --split test --out-dir figures/ntu
  python scripts/cross_dataset_eval.py --checkpoint runs/best_bilstm.pt \
    --data data/processed/ntu_fi/ntu_fi.npz --tag uthar_on_ntu
  python scripts/cross_dataset_eval.py --checkpoint runs/best_bilstm_ntu.pt \
    --data data/processed/ut_har/ut_har.npz --tag ntu_on_uthar
  python scripts/domain_shift_matrix.py
}

# --- chunk 10: Widar3.0 BVP exploration --------------------------------------
stage_widar() {
  banner "CHUNK 10 — Widar3.0 BVP exploration (explore_widar.py + visualize_widar_bvp.py)"
  # What: characterize the downloaded Widar3.0 BVP set (counts by gesture /
  #       user / position / orientation, T and value stats, label conventions)
  #       and render the 6-gesture × 4-timestep BVP example grid. No modeling —
  #       this is the load+explore chunk (see notes/widar_data.md,
  #       docs/chunk10_widar_bvp.md). Requires data/raw/widar3/bvp/ (download
  #       BVP.zip per README.md; gitignored, ~400 MB / 43.7k files).
  # Out:  stdout report + figures/widar_bvp_examples.png
  local WIDAR_STATS_SAMPLE=3000 # files opened for T/value stats (0 = all; slow)
  python scripts/explore_widar.py --sample "$WIDAR_STATS_SAMPLE"
  python scripts/visualize_widar_bvp.py
}

# --- chunk 11: Widar3.0 BVP preprocessing pipeline ---------------------------
stage_bvp() {
  banner "CHUNK 11 — Widar3.0 BVP preprocessing pipeline (bvp_pipeline_demo.py)"
  # What: the model-ready pipeline over Widar3.0 BVP — composable transforms
  #       (src/data/bvp_preprocess.py: normalize_bvp / pad_or_truncate /
  #       augment_bvp) plus the PyTorch Dataset and the four canonical
  #       cross-domain split helpers (src/data/widar_dataset.py: cross_user /
  #       cross_position / cross_orientation / in_domain). Here we just render
  #       the pipeline demo; chunk 12 trains/visualizes on these splits.
  #       Requires data/raw/widar3/bvp/ (see chunk 10).
  # Out:  figures/bvp_pipeline_demo.png (raw -> normalized -> padded, with the
  #       per-frame energy-centroid motion trajectory overlaid).
  # 13. python scripts/bvp_pipeline_demo.py
  python scripts/bvp_pipeline_demo.py
}

# --- chunk 12: train the BVP CNN-RNN, one model per split (HEAVY) -------------
stage_train_widar() {
  banner "CHUNK 12 — train BVP CNN-RNN over the 4 cross-domain splits (src/train_widar.py)"
  # Variables ---------------------------------------------------------------
  local TW_EPOCHS=40  # max epochs per split (early stopping may stop sooner)
  local TW_BATCH=64   # batch size
  local TW_PATIENCE=8 # early-stop patience (epochs w/o val-acc gain)
  local TW_HIDDEN=128 # GRU hidden units per direction
  local TW_SEED=42    # RNG seed
  # What: fits ONE bvp_cnn_rnn per split (in_domain / cross_user /
  #       cross_position / cross_orientation), each on its own train partition,
  #       early-stopping on a same-domain val carve. CPU-heavy over the full
  #       ~43.7k corpus — scope it with --room / --max-per-gesture for a quick
  #       pass (see docs/chunk12_spatial_visualization.md).
  # Out:  runs/best_bvp_<split>.pt (x4) + runs/<ts>_<split>/{best.pt,metrics.json,
  #       training_curves.png}.  Excluded from the no-arg default (heavy).
  # Skip-if-exists: if all four stable checkpoints already exist, do nothing.
  local missing=0
  for s in in_domain cross_user cross_position cross_orientation; do
    [ -f "runs/best_bvp_${s}.pt" ] || missing=1
  done
  if [ "$missing" = "0" ]; then
    echo "All four runs/best_bvp_*.pt exist — skipping (delete one to retrain)."
    return 0
  fi
  python src/train_widar.py \
    --epochs "$TW_EPOCHS" --batch-size "$TW_BATCH" --patience "$TW_PATIENCE" \
    --hidden "$TW_HIDDEN" --seed "$TW_SEED"
}

# --- chunk 12: evaluate every split + the 2x2 domain-results figure -----------
stage_evaluate_widar() {
  banner "CHUNK 12 — evaluate BVP across all four splits (src/evaluate_widar.py)"
  # What: loads each runs/best_bvp_<split>.pt, rebuilds its held-out test set,
  #       and scores it; assembles the 2x2 confusion-matrix figure. The BVP
  #       environment-invariance test (compare vs chunk 9's domain-shift matrix).
  #       Light if checkpoints exist; harmlessly no-ops if none do.
  # Out:  figures/widar_domain_results.png, figures/widar/<split>_{metrics.json,
  #       predictions.csv}.
  # Skip-if-exists: skip when the results figure is already present.
  if [ -f "figures/widar_domain_results.png" ]; then
    echo "figures/widar_domain_results.png exists — skipping (delete to re-eval)."
    return 0
  fi
  python src/evaluate_widar.py
}

# --- chunk 12: spatial-motion milestone figure -------------------------------
stage_spatial() {
  banner "CHUNK 12 — spatial-motion visualization (scripts/spatial_viz.py)"
  # Variables ---------------------------------------------------------------
  local SP_DPI=160 # output figure DPI
  local SP_SEED=0  # which sample per gesture is drawn (deterministic)
  # What: the tier-2 analog of chunk 6's 3-panel figure — for six gestures, a
  #       BVP-frames strip over the velocity plane plus the integrated motion
  #       trajectory, titled with predicted vs ground-truth (misses flagged).
  #       Needs one runs/best_bvp_*.pt; skips cleanly if none exist yet.
  # Out:  figures/spatial_motion.png
  # Skip-if-exists: skip when the figure already exists.
  if [ -f "figures/spatial_motion.png" ]; then
    echo "figures/spatial_motion.png exists — skipping (delete to re-render)."
    return 0
  fi
  if ! ls runs/best_bvp_*.pt >/dev/null 2>&1; then
    echo "No runs/best_bvp_*.pt yet — run train_widar first; skipping."
    return 0
  fi
  python scripts/spatial_viz.py --dpi "$SP_DPI" --seed "$SP_SEED"
}

# --- chunk 13: MM-Fi pose dataset — census + ground-truth skeletons ----------
stage_explore_mmfi() {
  banner "CHUNK 13 — explore MM-Fi pose dataset (scripts/explore_mmfi.py)"
  # What: Phase 3 (POSE ESTIMATION, a REGRESSION task — not classification).
  #       Prints sample counts by env/subject/action and CSI + 3D-keypoint
  #       shapes/ranges, then renders 8 ground-truth skeletons — the first time
  #       the project shows real human poses.
  # Out:  figures/mmfi_gt_skeletons.png
  # Data: MM-Fi is NOT downloaded by this pipeline (large, Google Drive). See
  #       docs/chunk13_mmfi_setup.md. When run DIRECTLY the script fails loudly
  #       if data is missing; here we self-skip so the default reproduction run
  #       doesn't break for users who haven't downloaded it.
  if [ ! -d "data/raw/mmfi" ]; then
    echo "data/raw/mmfi/ absent — MM-Fi not downloaded; skipping."
    echo "  To enable: see docs/chunk13_mmfi_setup.md (start with E01)."
    return 0
  fi
  if [ -f "figures/mmfi_gt_skeletons.png" ]; then
    echo "figures/mmfi_gt_skeletons.png exists — skipping (delete to re-render)."
    return 0
  fi
  python scripts/explore_mmfi.py
}

# --- chunk 14: MM-Fi pose pairs — preprocessing + alignment verification ------
stage_verify_pose_pairs() {
  banner "CHUNK 14 — verify MM-Fi pose pairs (scripts/verify_pose_pairs.py)"
  # What: builds (CSI window, 3D pose) training pairs (src/data/pose_preprocess.py
  #       + src/data/mmfi_pose_dataset.py) and PROVES they are temporally aligned
  #       before any training — an index check (window centered on the labeled
  #       frame) plus an independent cross-loader check (frame-mode vs
  #       sequence-mode agree). Misalignment is the #1 cause of a pose model that
  #       "trains" but predicts garbage; catch it here.
  # Out:  figures/pose_pair_check.png (CSI amplitude window beside its GT skeleton)
  # Window: --window-size defaults to 1 (MM-Fi benchmark protocol). ASK before
  #         changing — it breaks comparability with published numbers (see
  #         docs/chunk14_pose_pipeline.md).
  # Data: MM-Fi is NOT downloaded by this pipeline (large, Google Drive). See
  #       docs/chunk13_mmfi_setup.md. Self-skips here if absent / already rendered
  #       so the default reproduction run doesn't break.
  if [ ! -d "data/raw/mmfi" ]; then
    echo "data/raw/mmfi/ absent — MM-Fi not downloaded; skipping."
    echo "  To enable: see docs/chunk13_mmfi_setup.md (start with E01)."
    return 0
  fi
  if [ -f "figures/pose_pair_check.png" ]; then
    echo "figures/pose_pair_check.png exists — skipping (delete to re-verify)."
    return 0
  fi
  python scripts/verify_pose_pairs.py
}

# --- chunk 15: train the CSI->3D-pose regressor (HEAVY) ----------------------
stage_train_pose() {
  banner "CHUNK 15 — train CSI->3D-pose regressor (src/train_pose.py)"
  # Variables ---------------------------------------------------------------
  local TP_SPLIT=cross_subject # cross_subject | cross_environment | random_split
  local TP_EPOCHS=60           # max epochs (early stopping may stop sooner)
  local TP_BATCH=64            # batch size
  local TP_PATIENCE=12         # early-stop patience (epochs w/o val-MPJPE gain)
  local TP_LR=1e-3             # Adam learning rate
  local TP_BONE=0.1            # bone-length regularizer weight (0 disables)
  local TP_SEED=42             # RNG seed
  local TP_PROGRESS=5          # save a GT-vs-pred skeleton figure every N epochs
  # What: fits CSIPoseNet (src/models/csi_pose_net.py) — a small CNN encoder +
  #       MLP head regressing a (17,3) 3D pose from a CSI window. This is the
  #       project's first REGRESSION model: target is continuous joint
  #       coordinates, metric is MPJPE (mm), NOT accuracy. Early-stops on val
  #       MPJPE and saves skeleton-convergence snapshots under <run>/progress/.
  #       window_size stays at the MM-Fi benchmark default 1 (ASK before
  #       changing — breaks comparability with the paper). See
  #       docs/chunk15_pose_model.md.
  # Out:  runs/best_pose.pt + runs/<ts>/{best.pt,metrics.json,training_curves.png,
  #       progress/epoch_NNN.png}.  Excluded from the no-arg default (heavy).
  # Data: MM-Fi is NOT downloaded by this pipeline (large, Google Drive). See
  #       docs/chunk13_mmfi_setup.md. Self-skips here if absent / already trained
  #       so the default reproduction run doesn't break.
  if [ ! -d "data/raw/mmfi" ]; then
    echo "data/raw/mmfi/ absent — MM-Fi not downloaded; skipping."
    echo "  To enable: see docs/chunk13_mmfi_setup.md (start with E01)."
    return 0
  fi
  if [ -f "runs/best_pose.pt" ]; then
    echo "runs/best_pose.pt exists — skipping (delete to retrain)."
    return 0
  fi
  python src/train_pose.py \
    --split "$TP_SPLIT" --epochs "$TP_EPOCHS" --batch-size "$TP_BATCH" \
    --patience "$TP_PATIENCE" --lr "$TP_LR" --bone-weight "$TP_BONE" \
    --seed "$TP_SEED" --progress-every "$TP_PROGRESS"
}

# --- chunk 16: evaluate the CSI->3D-pose regressor ---------------------------
stage_evaluate_pose() {
  banner "CHUNK 16 — evaluate CSI->3D-pose (src/evaluate_pose.py)"
  # What: scores runs/best_pose.pt with MPJPE + PA-MPJPE (Procrustes-aligned),
  #       overall AND per-joint, on the held-out partition of each split it can
  #       build from what's on disk (cross_subject is the faithful test;
  #       cross_environment needs an env beyond E01; random_split is the leaky
  #       i.i.d. ceiling — each flagged in the output). Renders the per-joint
  #       error bar chart (wrists/ankles worst) and a benchmark-comparison
  #       table. Light + non-destructive. See docs/chunk16_pose_deliverable.md.
  # Out:  figures/pose_per_joint_error.png, figures/pose_eval_metrics.json
  # Data + checkpoint gated: self-skips if MM-Fi or the checkpoint is absent so
  # the default reproduction run doesn't break.
  if [ ! -d "data/raw/mmfi" ]; then
    echo "data/raw/mmfi/ absent — MM-Fi not downloaded; skipping."
    echo "  To enable: see docs/chunk13_mmfi_setup.md (start with E01)."
    return 0
  fi
  if [ ! -f "runs/best_pose.pt" ]; then
    echo "runs/best_pose.pt absent — run train_pose first; skipping."
    return 0
  fi
  python src/evaluate_pose.py
}

# --- chunk 16: the capstone — animated WiFi-predicted 3D skeleton -------------
stage_pose_viz() {
  banner "CHUNK 16 — pose deliverable GIF (scripts/pose_visualization.py)"
  # What: the project's Phase-3 capstone — runs the regressor frame-by-frame
  #       over a held-out (cross-subject) motion clip and animates the predicted
  #       3D skeleton (red) beside ground truth (green), plus a prediction-only
  #       "in the wild" panel, annotated with the running MPJPE. RAW model
  #       output — no temporal smoothing (ask before adding; honest by design).
  #       Auto-picks the most-motion held-out clip; override with --subject /
  #       --action. See docs/chunk16_pose_deliverable.md.
  # Out:  figures/pose_prediction.gif (+ figures/pose_prediction_strip.png)
  # Data + checkpoint gated; self-skips if the GIF already exists (delete to
  # re-render) so the default reproduction run stays light.
  if [ ! -d "data/raw/mmfi" ]; then
    echo "data/raw/mmfi/ absent — MM-Fi not downloaded; skipping."
    echo "  To enable: see docs/chunk13_mmfi_setup.md (start with E01)."
    return 0
  fi
  if [ ! -f "runs/best_pose.pt" ]; then
    echo "runs/best_pose.pt absent — run train_pose first; skipping."
    return 0
  fi
  if [ -f "figures/pose_prediction.gif" ]; then
    echo "figures/pose_prediction.gif exists — skipping (delete to re-render)."
    return 0
  fi
  python scripts/pose_visualization.py
}

# =============================================================================
# DISPATCH
# =============================================================================
# Full ordered list (dependency order). Heavy/optional ones noted.
ALL_STAGES=(verify explore preprocess visualize train sweep evaluate capture finalviz diagnose postprocess preprocess_ntu train_ntu domainshift widar bvp train_widar evaluate_widar spatial explore_mmfi verify_pose_pairs train_pose evaluate_pose pose_viz)
# What a no-arg run does: the safe reproduction using the EXISTING frozen
# checkpoints — skips only the heavy, overwrite-y trainers (train/sweep/train_ntu/
# train_widar).  evaluate_widar and spatial are default-safe: they self-skip when
# their outputs exist and no-op cleanly when no BVP checkpoint has been trained.
DEFAULT_STAGES=(verify explore preprocess visualize evaluate capture finalviz diagnose postprocess preprocess_ntu domainshift widar bvp evaluate_widar spatial explore_mmfi verify_pose_pairs evaluate_pose pose_viz)

case "${1:-}" in
list)
  echo "All stages (dependency order):"
  printf '  %s\n' "${ALL_STAGES[@]}"
  echo ""
  echo "Default (no-arg) stages:"
  printf '  %s\n' "${DEFAULT_STAGES[@]}"
  exit 0
  ;;
all)
  STAGES=("${ALL_STAGES[@]}")
  ;;
"")
  STAGES=("${DEFAULT_STAGES[@]}")
  ;;
*)
  STAGES=("$@")
  ;;
esac

echo "Env: $CONDA_ENV   Root: $ROOT"
echo "Running stages: ${STAGES[*]}"

for s in "${STAGES[@]}"; do
  fn="stage_$s"
  if ! declare -F "$fn" >/dev/null; then
    echo "ERROR: unknown stage '$s' (try: ./run_pipeline.sh list)" >&2
    exit 2
  fi
  "$fn"
done

banner "DONE — ${STAGES[*]}"
