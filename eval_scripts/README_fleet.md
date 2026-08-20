# Fleet orchestration

    eval_scripts/campaign_status.sh     pods, queue, evidence collected
    eval_scripts/fleet_dispatch.sh      launch queued jobs onto idle pods (flock, one instance)
    eval_scripts/fleet_watch.sh         alert on failure, stall, or idle pod (loops)
    eval_scripts/fleet_tb_sync.sh       pull TB events, params, eval jsons home

State lives in `logs/fleet/` (gitignored, survives session restarts):
`pods.txt` (port host name), `seed_queue.txt` (one job per line), the upload tars.

Job lines:

    train  <arm> <task> <iters> <freeze:yes|no> <seed>   -> pod_chain.sh: train, wave grid, deep row
    bceval <tag> <checkpoint-path> <scoring|bc>          -> pod_bc_eval.sh: deep row only

## Registering a new pod

A pod is usable only when the API reports a `type:"tcp"` port AND the GPU has
free VRAM (community hosts hand out cards another tenant is already filling).
After creating one, wait for `runtime.ports` to contain a tcp entry, then:

    ssh -p <port> root@<ip> 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'
    # >= 35000 MiB, then append "port ip name" to logs/fleet/pods.txt

CHECK BACK on pods whose port was not mapped yet at first look: a pod that gets
its port late is invisible to the dispatcher and bills while idle until someone
registers it (this cost about an hour on seed-c, 2026-08-20).
