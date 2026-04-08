#! /bin/bash

export HYDRA_FULL_ERROR=1
export TORCH_LOGS="-dynamo,-inductor,-aot"
# export WANDB_MODE="offline"
# export HF_HUB_OFFLINE=1

MASTER_PORT=$((25000+$SLURM_JOB_ID%5000))
if [ "$SLURM_GPUS_ON_NODE" -eq 1 ] && [ "$SLURM_NNODES" -eq 1 ]; then
    echo "Non-distributed training"
    export LAUNCH="python train.py"
else
    HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    MASTER_ADDR=$(echo $HOSTNAMES | cut -d ' ' -f1)
    MASTER_IP=$(getent hosts $MASTER_ADDR | awk '{print $1}')
    echo "Distributed Setup"
    echo "Hostnames: $HOSTNAMES"
    echo "Master endpoint: $MASTER_IP:$MASTER_PORT ($MASTER_ADDR)"
    echo "GPUs per node: $SLURM_GPUS_ON_NODE"
    export LAUNCH="torchrun --master-port $MASTER_PORT --nnodes $SLURM_NNODES --nproc-per-node "'$SLURM_GPUS_ON_NODE'" --rdzv-id $SLURM_JOB_ID --rdzv-backend c10d --rdzv-endpoint $MASTER_IP:$MASTER_PORT "'--rdzv-conf="is_host=$( [ $SLURM_PROCID -eq 0 ] && echo true || echo false )"'" train.py"
fi

export EXTRA_ARGS="$@"
launch() {
    echo "Arguments: $*"
    echo "Overrides: $EXTRA_ARGS"
    export ARGS="$* $EXTRA_ARGS"
    read -r -d '' COMMON_SCRIPT << 'EOF'
# Trap SIGUSR1 in this shell (PID 1 inside container) and forward it to train.py workers
SIGNAL_HANDLED=false
forward_sigusr1() {
    if [ "$SIGNAL_HANDLED" = "true" ]; then
        return
    fi
    echo "[node rank $SLURM_PROCID]: got SIGUSR1, forwarding to local ranks"
    SIGNAL_HANDLED=true
    # We invoke bash, which invokes torchrun, which invokes the train script local_world_size times
    # We cant just filter pgrep for python train.py because that will also catch dataloader workers etc
    # The hierarchy lets us get all the train script PIDs though
    TORCHRUN_PID=$(pgrep -P $CHILD_PID | head -n 1)
    pids=$(pgrep -P $TORCHRUN_PID)
    echo "[node rank $SLURM_PROCID]: found trainer pids: $(echo $pids)"
    if [ -n "$pids" ]; then
        kill -USR1 $pids
    fi
}
trap forward_sigusr1 USR1

eval "$LAUNCH $ARGS" &
CHILD_PID=$!

# Wait in endless loop until our child process has actually exited
# We cant just call wait once because the trap will interrupt it, so we loop until the process is gone
# and then exit with its exit code
while kill -0 "$CHILD_PID" 2>/dev/null; do
    wait "$CHILD_PID"
    EXIT_CODE=$?

    if ! kill -0 "$CHILD_PID" 2>/dev/null; then
        exit $EXIT_CODE
    fi
done
EOF

    if [[ -n "$SLURM_STEP_ID" && ("$SLURM_STEP_ID" != "0" || -z "$SLURM_BATCH_JOB") ]]; then
        # If we're already inside an srun step, we don't want to try to launch another one
        bash -lc "$COMMON_SCRIPT"
    else
        srun --ntasks-per-node=1 bash -lc "$COMMON_SCRIPT"
    fi
}
export -f launch