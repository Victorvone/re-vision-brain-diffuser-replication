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
GPU-accessible memory pools.

**Correction (2026-09-07, verified by direct test): the 16GiB cgroup cap does NOT gate GPU
device memory.** A direct `torch.cuda`/hipMalloc allocation of 20GB left `/sys/fs/cgroup/memory.current`
unchanged while `torch.cuda.memory_allocated()` showed the full 20GB - the ~96GB GPU pool is
device memory managed by the amdgpu driver, charged separately from the cgroup. The cap only
bites host-RAM-resident (CPU-side) allocations. This means the admin's earlier assessment was
correct and no cluster-side change was needed for the Versatile Diffusion OOM (see "Known fix"
below) - the practical implication is: **build and load models directly onto the GPU device**
(`with torch.device('cuda:0'):` around construction, `torch.load(..., map_location='cuda:0')`
for checkpoints) rather than materializing them in CPU RAM first, since CPU RAM is still hard-capped
at 16GiB with no swap.

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

`get_model()(cfgm)` instantiates the 3.3B-param Versatile Diffusion model in **fp32 on CPU by
default** (~17GB peak cgroup RAM measured directly, worse than the ~13.3GB originally estimated)
- on this machine that alone exceeds the 16GB cgroup budget before any checkpoint is even
loaded, and no CPU-side workaround (fp16 cast, mmap) gets a turn early enough to prevent it. The
earlier "known fix" (cast to fp16 + mmap after CPU instantiation) reduced but didn't eliminate
the risk - it still peaked at ~16.3GB in testing, uncomfortably close to the cap.

**Real fix (2026-09-07): never materialize the model in CPU RAM at all.** Build it directly on
the GPU, since GPU device memory isn't charged to the cgroup (see "Compute environment" above):
```python
with torch.device('cuda:0'):
    net = get_model()(cfgm)
net.half()
sd = torch.load(pth, map_location='cuda:0')
net.load_state_dict(sd, strict=False)
```
This requires two things beyond the snippet above:
1. `accelerate` installed in `ml-env` (`pip install accelerate`) - transformers'
   `CLIPModel.from_pretrained`, called internally during construction, otherwise refuses to run
   under an ambient `torch.device` context.
2. A one-line patch already applied in `working_repo/versatile_diffusion/lib/model_zoo/diffusion_utils.py`'s
   `make_beta_schedule` - it forces its tiny (~1000-element) internal computation onto CPU
   regardless of the ambient device context, since it needs a real `.numpy()` call that a CUDA
   tensor can't service directly. Without this patch, construction under `torch.device('cuda:0')`
   crashes here first.

Peak cgroup RAM with this approach: ~13.6GB, flat throughout construction, `.half()`, and
checkpoint load (measured directly) - versus ~17GB (over budget) for plain fp32 construction and
~16.3GB for the old fp16+mmap-on-CPU approach. This is now the pattern used in
`working_repo/scripts/smoke_test_versatile_diffusion.py`, `versatilediffusion_reconstruct_images.py`,
and `roi_versatilediffusion_reconstruct.py` - all three ported and the smoke test verified
end-to-end (real checkpoint load + full DDIM sampling pass).

## Where things live

- `docs/replication-technical-requirements.md` - the living technical spec: what changes per
  pipeline stage to adapt Brain-Diffuser for LAION-fMRI, and environment/compute findings. Keep
  this updated as decisions are made; don't let it drift from what's actually been done.
- `docs/replication-task-overview.md` - the full checklist from here to final submission. Status
  source of truth.
- `docs/laion-pretraining-confound.md` - analysis of the Versatile Diffusion / LAION pretraining
  overlap: the problem, impact estimate per finding, and controls in cost order (including the
  alternative-diffusion-model option). Internal; the proposal carries only a short disclosure.
- `docs/proposal_draft.md` - the proposal itself, filled in section by section, word-limited
  (~1,300 words).
- `docs/re-vision-initiative-context.md` - crawled reference for re:vision's site rules,
  templates, and the `laion_fmri` dataset API. Static snapshot; re-check the live site if
  something here seems off.
- `working_repo/scripts/smoke_test_*.py` - environment/checkpoint verification scripts (synthetic
  inputs, real weights). Not the same as the full real-data smoke test, which is still pending.
