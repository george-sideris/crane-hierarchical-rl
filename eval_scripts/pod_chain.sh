#!/bin/bash
# Self-contained pod pipeline: train one arm in the foreground, then argmax-wave its
# own checkpoints, then mark done. Run detached via setsid nohup.
#   pod_chain.sh <arm> <task> <iters> <freeze:yes|no> <seed>
set -u
ARM=$1 TASK=$2 ITERS=$3 FREEZE=$4 SEED=$5
cd /data/crane_testbed || exit 1
export PYTHONPATH=/data/crane_testbed/source/crane_testbed:/data/crane_testbed/scripts/envs
CKPT=/data/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt
BCFLAGS="--bc_checkpoint $CKPT --sigma_init 0.3 --critic_warmup_iters 0 \
  --anneal_sigma_iters 100 --anneal_sigma_to 0.01"
[ "$FREEZE" = yes ] && BCFLAGS="$BCFLAGS --freeze_encoder"
STEPFLAG="agent.num_steps_per_env=8"
case "$TASK" in *Scratch*|*CosSin*) BCFLAGS=""; STEPFLAG="" ;; esac

/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task "$TASK" $BCFLAGS \
  --seed "$SEED" --num_envs 40 --max_iterations "$ITERS" --headless \
  --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
  --gripper_effort 2000 --num_logs 200 $STEPFLAG agent.save_interval=10 \
  > logs/p3v2_train.log 2>&1
echo "train exit $? $(date -u +%H:%M)" > /data/train_finished

RUN=$(ls -dt logs/rsl_rl/*/20* | head -1)
mkdir -p logs/battery_ckpts/$ARM
cp "$RUN"/model_*.pt logs/battery_ckpts/$ARM/ 2>/dev/null
rm -f logs/battery_ckpts/$ARM/model_0.pt
STRIDE=20 bash eval_scripts/thesis_battery.sh waves "$ARM" > /data/battery_waves.log 2>&1
# stage 3: the citable row - deep 100-episode eval of this arm's best wave checkpoint
bash eval_scripts/thesis_battery.sh deep "$ARM" > /data/battery_deep.log 2>&1
touch /data/CHAIN_DONE
