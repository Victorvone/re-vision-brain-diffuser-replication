# re:vision Brain-Diffuser replication

Replicating Ozcelik & VanRullen (2023) "Natural scene reconstruction from fMRI signals using
generative latent diffusion" (Brain-Diffuser) for the re:vision initiative, using the
LAION-fMRI dataset instead of the original NSD. Team: Victor von Eisenhart-Rothe, Lucas Nunn,
Adrien Doerig (Freie Universität Berlin).

**Status and deadlines change over time - `docs/replication-task-overview.md` is the current
source of truth, not this file.** Proposal due 2026-09-15, report due 2027-02-15, final revised
report due 2027-04-30 (verify against the task overview before relying on these).

## Repo rules

- `original_repo/` is a pristine copy of the upstream code and **must never be edited**. All
  work happens in `working_repo/`.
- Python environment: `/home/ubuntu/ml-env/bin/python3` (Python 3.12, PyTorch 2.12+ROCm) - not
  the `environment.yml` pinned in the repo, which targets Python 3.8/CUDA 11.3 and doesn't apply
  to this machine.
- Large pretrained checkpoints (`working_repo/vdvae/model/`, `working_repo/versatile_diffusion/pretrained/`)
  are gitignored - too big for GitHub, and re-downloadable. Don't try to commit them.
- No em-dashes in written output (docs, commit messages, proposal text).
- The proposal (`docs/proposal_draft.md`) and any other reviewer-facing document stay high-level:
  no code-level detail, no references to internal-only docs like this file or the technical
  requirements doc.

## Compute environment - read before running anything heavy

This machine ("victor1") is an **LXC container**, hard-capped to exactly **16GiB RAM and 8 CPU
threads** by a cgroup limit at the container root - confirmed by walking the full cgroup
hierarchy (`cat /sys/fs/cgroup/memory.max`). The underlying host chip (AMD Ryzen AI Max+ 395,
Radeon 8060S, unified memory) is far more capable - `rocminfo` reports ~96GB + ~32GB
GPU-accessible memory pools - but none of that is usable from inside this container as things
stand. Since it's a unified-memory architecture, the 16GB cap is also the effective GPU-memory
ceiling, regardless of what `rocminfo`/`rocm-smi` report.

A request to raise this allocation was handed to the VM's administrator on 2026-09-07 but was
not confirmed resolved as of that date. **Check `docs/replication-technical-requirements.md` §0
update for the current state before assuming more than 16GB is available.**

There is **no swap**, so an OOM kill is immediate and silent, not a gradual slowdown you can
react to. Two consequences:
- Any heavy job (checkpoint loading, DDIM sampling, large regression fits) should run inside a
  memory-capped, actively monitored scope - e.g.
  `systemd-run --user --scope -p MemoryMax=... -p CPUQuota=... -- timeout <seconds> <cmd>` -
  rather than backgrounded unattended. An earlier unmonitored run left the VM pegged for ~24h
  before it crashed.
- When something OOMs, check `journalctl --user -u <scope>.scope` for the exact "killed by the
  OOM killer" confirmation rather than guessing at the cause.

## Known fix: Versatile Diffusion OOM on load

`get_model()(cfgm)` instantiates the 3.3B-param Versatile Diffusion model in **fp32 by default**
(~13.3GB) before any checkpoint is loaded - on this machine that alone nearly exhausts the
16GB budget. Fix: cast to fp16 immediately after instantiation, before `torch.load`/
`load_state_dict`, and load the checkpoint with `mmap=True`. Working example in
`working_repo/scripts/smoke_test_versatile_diffusion.py`. This fix has **not yet** been ported
into the real `versatilediffusion_reconstruct_images.py` or `roi_versatilediffusion_reconstruct.py`
- do that before running either for real.

## Where things live

- `docs/replication-technical-requirements.md` - the living technical spec: what changes per
  pipeline stage to adapt Brain-Diffuser for LAION-fMRI, and environment/compute findings. Keep
  this updated as decisions are made; don't let it drift from what's actually been done.
- `docs/replication-task-overview.md` - the full checklist from here to final submission. Status
  source of truth.
- `docs/proposal_draft.md` - the proposal itself, filled in section by section, word-limited
  (~1,300 words).
- `docs/re-vision-initiative-context.md` - crawled reference for re:vision's site rules,
  templates, and the `laion_fmri` dataset API. Static snapshot; re-check the live site if
  something here seems off.
- `working_repo/scripts/smoke_test_*.py` - environment/checkpoint verification scripts (synthetic
  inputs, real weights). Not the same as the full real-data smoke test, which is still pending.
