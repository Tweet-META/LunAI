# Scarlet Meister: 300k-frame transfer experiment

Start from each formal 2M checkpoint. Fine-tune only on Hard Scarlet Meister for
300,000 additional environment frames (cumulative stop: 2,300,000). Five variants
times seeds 0, 1, 2 = 15 runs, 4,500,000 added frames. Models run sequentially;
each uses the existing eight-environment PPO trainer.

## Preserved settings

Each variant keeps its observation scales, PCCM observation mode, and PCC reward
weight (0 or 0.1). No-PCCM uses full scales, occupancy_only, reward weight 0.
Wall reward attenuation and all game collision rules remain unchanged.
Learning rate 4e-5 and entropy coefficient .0015 stay constant; gamma .99,
frame_stack 4, action_repeat 1, 1800-frame episode cap, rollout 1024,
minibatch 256, four epochs. Existing model **and optimizer** states are loaded;
episode/update/frame counters continue, and fresh Scarlet Meister episodes start.
Subtract 2,000,000 from logged total_frame_steps to plot transfer learning.

These are transfer/adaptation experiments, not from-scratch learning or a claim
of convergence. Comparing pre/post performance should use the retained original
2M model and the new 2.3M model on the same Scarlet Meister evaluation seeds.
Pre-transfer evaluation may be performed later since the original model is kept.
Compare Full vs NoYellow at matched PCC reward weights; Full vs NoPCCM both use
PCC reward weight zero. Reward itself is not a cross-reward-variant performance metric.

## Run from the project root

```powershell
python -u tools/finetune_scarlet_meister.py
```

The script checks **all** selected inputs before starting. It requires exactly
2,000,000 saved frames and the expected observation configuration. A missing
source or existing destination stops the batch before training. A runtime error
stops subsequent runs. At each successful end it checks for 2,300,000 frames.
PCC reward weights are assigned from the formal variant names; the original PPO
checkpoint configuration does not store the reward weight.

Original four seed-0 models may live directly under checkpoints/formal_1800_2m;
the script also supports their seed_0 subfolder, but refuses ambiguous duplicates.
NoPCCM seed 0 and all seed 1/2 models use their seed subfolders.

Outputs retain model names under:

```text
checkpoints/scarlet_meister_300k/seed_N/lunai_v9.1*.pt
training_logs/scarlet_meister_300k/seed_N/lunai_v9.1*.csv
training_logs/scarlet_meister_300k/seed_N/lunai_v9.1*.transfer.json
```

Transfer metadata records source SHA256, argument list and status. Other than the
new runner and this documentation, no existing training/environment code is edited.

Optional controls:

```powershell
python tools/finetune_scarlet_meister.py --dry-run
python tools/finetune_scarlet_meister.py --check-only
python -u tools/finetune_scarlet_meister.py --seeds 1 2
python -u tools/finetune_scarlet_meister.py --seeds 0 --variants nopccm
```

The batch launcher does not overwrite or auto-resume interrupted outputs. If a
run is interrupted, preserve its model/log and inspect its saved cumulative
frame count before constructing a continuation command. Completed or remaining
variants can be selected explicitly; do not relaunch the whole default batch
over existing outputs.
