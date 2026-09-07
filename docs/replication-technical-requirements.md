# Technical Requirements - Adapting Brain-Diffuser for LAION-fMRI

This is the standalone technical reference for replicating Ozcelik & VanRullen (2023) "Brain-Diffuser" using the re:vision initiative's LAION-fMRI dataset instead of the original NSD dataset. It documents, per pipeline stage: what the original code (`original_repo/`) does, what's NSD-specific vs. reusable as-is, exactly what must change for LAION-fMRI, and the results of our environment/dependency verification pass. The proposal draft (`proposal_draft.md`) references this document rather than repeating it.

All work happens in `working_repo/` (a full, unmodified copy of `original_repo/` made 2026-09-03). `original_repo/` itself is never edited.

## 0. Environment status (verified 2026-09-03)

This machine has **no NVIDIA/CUDA GPU** - it has an AMD APU (Radeon 8060S, ROCm) - and only **16GB RAM with no swap**. `original_repo/environment.yml` pins Python 3.8.13 + PyTorch 1.12.1 + CUDA 11.3, which is moot here regardless of dataset (no CUDA hardware, no conda installed). Instead we're using the existing `/home/ubuntu/ml-env` (Python 3.12, PyTorch 2.12.1+rocm7.2, already confirmed working against the AMD GPU).

**Packages added to ml-env** to cover the repo's dependencies not already present:
`nibabel`, `omegaconf`, `easydict`, `kornia`, `h5py`, `scikit-image`, `einops`, `clip` (OpenAI CLIP, installed from `git+https://github.com/openai/CLIP.git`), and `opencv-python-headless` (swapped in for `opencv-python` - the pinned package fails to import here because it needs a system `libGL.so.1` that's absent; headless doesn't need it and the repo never calls any `cv2` GUI/display function, so this is a safe substitution).

**Import verification pass:** every `.py` file in `working_repo/` was syntax-checked (`ast.parse`), and all 15 argparse-based entry-point scripts under `scripts/` and `data/prepare_nsddata.py` were run with `--help` to trigger their real top-level imports (torch, clip, kornia, the vendored `versatile_diffusion/lib/*`, etc.) without executing any compute/download logic. The two scripts with no CLI args (`scripts/save_test_images.py`, `data/download_nsddata.py`) were checked by inspecting their import statements directly (both trivial: `numpy`/`os`/`PIL` and `os` only).

- **Initial run surfaced one real gap**: `einops` (pinned in `environment.yml` but not yet installed) - missing on the first pass, causing `ModuleNotFoundError` in the 4 scripts that import the vendored diffusion code (`cliptext_extract_features.py`, `clipvision_extract_features.py`, `roi_versatilediffusion_reconstruct.py`, `versatilediffusion_reconstruct_images.py`). Installed; all 4 now import cleanly.
- **False positive**: `ast.parse` flagged a UTF-8 BOM (`U+FEFF`) at the start of `versatile_diffusion/lib/experiments/{sd_default,vd_default}.py` as a syntax error. Confirmed via a direct runtime `import` that Python's actual import machinery (unlike bare `ast.parse`) handles the BOM transparently - this is harmless. (Two more BOM-flagged copies exist under `versatile_diffusion/log/sd_nodataset/.../code/`, a leftover duplicated code snapshot from a prior eval run, not imported by anything in `scripts/` - safe to ignore.)
- **Result: all 17 entry-point scripts now import cleanly** end-to-end under Python 3.12 + PyTorch 2.12/ROCm + `transformers 5.13.0`/`huggingface_hub 1.23.0`/`tokenizers 0.22.2` - despite this being a large version jump from the original pins (`transformers==4.19.2`, `huggingface-hub==0.11.0`). No import-time API breakage was found.
- **Caveat / what this does NOT prove**: this check never loaded real model weights (none are downloaded yet) or executed the actual regression/reconstruction/diffusion code paths - only the module-level imports. Deeper API incompatibilities in the vendored `versatile_diffusion/lib/model_zoo/*` code (e.g. old-style `transformers.CLIPTokenizer`/config-loading calls that may have changed signatures in newer `transformers`) can still surface once we run a real smoke test with actual weights. That full smoke test is intentionally deferred until after the proposal is finalized (per the current plan).
- `aws` CLI is **not installed** on this machine - irrelevant for our purposes since we won't use `data/download_nsddata.py` at all (see §1 below), but noting it since that script assumes it's present.

### Update 2026-09-07: runtime smoke test, and a compute-environment correction

The "full smoke test... deferred until after the proposal" note above is now partially superseded: `working_repo/scripts/smoke_test_vdvae.py` and `working_repo/scripts/smoke_test_versatile_diffusion.py` were run against the **real downloaded checkpoints** (synthetic 2-sample inputs, not real LAION-fMRI data - that part is still deferred, see the task overview). Both pass end-to-end: checkpoint loading, encoder/decoder forward passes, and DDIM sampling all work under the newer PyTorch/transformers stack. This resolves the "deeper API incompatibilities... can still surface" caveat above for the VDVAE and Versatile Diffusion *model code paths* specifically. Still unverified: the sklearn regression scripts, the evaluation scripts, and the ROI scripts (still import-checked only).

**Real finding, not anticipated**: the Versatile Diffusion smoke test OOM-killed the VM on first attempt (crashed 2026-09-03, ~24h of runaway CPU before the machine went down) because `get_model()(cfgm)` instantiates the 3.3B-param model in **fp32** (~13.3GB) by default, before the checkpoint is even loaded. Fix, applied in the smoke test script: cast to fp16 immediately after instantiation (matching the checkpoint's native dtype) and `mmap=True` on `torch.load`, before `load_state_dict`. This dropped peak RAM from OOM to ~12.5GB. **This fix has only been applied to the smoke test script - it still needs to be ported to `versatilediffusion_reconstruct_images.py` and `roi_versatilediffusion_reconstruct.py` themselves** before those are run for real (see §3's update below).

**Correction to this section's environment description**: "16GB RAM with no swap" is accurate for what's *usable*, but the reason is more specific than "this machine has 16GB RAM" - this machine (`victor1`) is an LXC container on a host with a substantially more capable chip (AMD Ryzen AI Max+ 395 APU, unified memory; `rocminfo` reports ~96GB + ~32GB GPU-accessible memory pools). The container itself is hard-capped to exactly 16GiB RAM and 8 CPU threads via a cgroup limit at the container root (confirmed by walking the full cgroup hierarchy - host has 32 threads per `lscpu`). Since this is unified memory, the container's RAM cap is also the GPU-memory cap. **As of this writing, a request to raise this allocation has been handed to the VM's administrator but not yet confirmed resolved** - treat 16GB as the working assumption for all planning until confirmed otherwise, and re-check before assuming more headroom is available for the real end-to-end run.

## 1. Data acquisition & prep - fully NSD-specific, needs full reimplementation

`data/download_nsddata.py` and `data/prepare_nsddata.py` are entirely tied to NSD's data layout and are **not reusable** for LAION-fMRI:
- `download_nsddata.py`: hardcodes `sub in [1,2,5,7]`, 37 sessions/subject, shells out to `aws s3 cp` against `s3://natural-scenes-dataset/...`. LAION-fMRI is accessed via the `laion_fmri` Python package instead (see `re-vision-initiative-context.md` §7) - this script is simply not used.
- `prepare_nsddata.py -sub x`: parses NSD's `nsd_expdesign.mat` design file (27,750-trial structure, `subjectim`/`masterordering` fields), NSD's 5-subjects-share-1000-images test split convention, `nsd_stimuli.hdf5` (425×425×3 images), and `COCO_73k_annots_curated.npy` (5 captions/image). Output format it produces - `nsd_{train,test}_{fmriavg_nsdgeneral,stim,cap}_sub{x}.npy` - is what every downstream script consumes, so **we must write a LAION-fMRI equivalent that produces the same shapes/format** (per-subject train/test fMRI-beta arrays, stimulus-image arrays, caption arrays), using `laion_fmri.subject.load_subject().get_betas()` / `.get_trial_info()` and `laion_fmri.load_stimuli()` (see context doc for the API) plus one of the provided splits (`random_0` as the direct NSD-train/test analog, since NSD's split isn't leakage-controlled either - `tau`/`cluster_k5`/`ood` are for the optional generalization stage per re:vision's rules, not the base replication).
- Captions: LAION-fMRI provides `stim.captions.human(...)` - need to confirm during implementation whether it provides ~5 captions/image like NSD/COCO, or a different count; `cliptext_extract_features.py` averages embeddings over however many captions are passed in per image, so a different count is not a blocker, just something to confirm.

## 2. VDVAE stage - mostly reusable, two constants to fix

`vdvae_extract_features.py` → `vdvae_regression.py` → `vdvae_reconstruct_images.py`. This stage only depends on having correctly-shaped fMRI and stimulus-image arrays (i.e., whatever our LAION-fMRI data-prep step above produces) - the VDVAE model itself (pretrained ImageNet64 checkpoint) is dataset-independent.

**Must change:**
- `assert sub in [1,2,5,7]` in all three scripts - trivial one-line relaxation to accept LAION-fMRI's subject IDs.
- The `/300` fMRI-beta scaling constant, present in `vdvae_regression.py` (and both CLIP regression scripts) - this normalizes NSD's specific GLM beta units before ridge regression. LAION-fMRI's GLMsingle betas may be on a different scale; this constant needs re-deriving (e.g. by checking the actual beta value range/std for LAION-fMRI) or replacing with a principled standardization (z-scoring) instead of a hardcoded magic number.

**Not blocking, but note:** `vdvae_extract_features.py` resizes stimulus images to 64×64 and has no CPU fallback (`.cuda()` unconditional) - fine on this machine since ROCm PyTorch reports `cuda` availability, but worth confirming batch size (`-bs`, default 30) fits in the 16GB RAM/unified-memory budget once real data is used.

## 3. Versatile Diffusion stage - reusable logic, single-GPU consolidation required

`cliptext_extract_features.py`, `clipvision_extract_features.py`, `cliptext_regression.py`, `clipvision_regression.py`, `versatilediffusion_reconstruct_images.py`. Same subject-ID and `/300`-scaling fixes as §2 apply here too (regression scripts use `alpha=100000` for CLIP-text, `alpha=60000` for CLIP-vision - these ridge-regularization strengths were tuned for NSD's data scale and may need re-tuning once the beta-scaling issue above is resolved).

**Must change - hard requirement, not optional:** `versatilediffusion_reconstruct_images.py` explicitly hardcodes a **2-GPU split** - `net.clip`/`net.autokl` on `cuda(0)`, the diffusion UNet and conditioning tensors forced onto `cuda(1)` - because the original authors ran this on "two 12GB GPUs" (per README). This machine has one (unified-memory) AMD GPU, so every `.cuda(1)` call needs to become `.cuda(0)` (i.e., everything on the single available device). The same fix applies to `roi_versatilediffusion_reconstruct.py` (§5). Given 16GB total system RAM shared between OS and GPU, we should also expect to need reduced batch sizes and/or fp16 precision to avoid OOM (there's no swap to cushion an overrun).

**Confirmed, not just anticipated (2026-09-07):** the fp16 need above isn't hypothetical - it's what caused the VM crash documented in §0's update. The fix is now proven in `scripts/smoke_test_versatile_diffusion.py`: call `net.half()` immediately after `get_model()(cfgm)`, *before* `torch.load`/`load_state_dict` (the model defaults to fp32 - ~13.3GB - which alone nearly exhausts the 16GB budget before the checkpoint is even loaded), and load the checkpoint with `mmap=True`. Still needs porting into `versatilediffusion_reconstruct_images.py` and `roi_versatilediffusion_reconstruct.py` when those are adapted - not done yet.

## 4. Evaluation - one NSD-specific constant to re-derive

`eval_extract_features.py` (extracts Inception-v3/CLIP/AlexNet×2/EfficientNet-b1/SwAV-ResNet50 features from ground-truth and reconstructed images) and `evaluate_reconstruction.py` (computes PixCorr, SSIM, 2-way identification accuracy).

**Must change:**
- `self.num_test = 982` is hardcoded (NSD's shared-1000-images test split, minus known overlaps) - must be replaced with whatever our actual LAION-fMRI test-split size ends up being (from §1's data-prep reimplementation).
- `eval_extract_features.py` hardcodes `device = 1` (assumes ≥2 GPUs) - same single-device fix as §3.
- `torch.hub.load('facebookresearch/swav:main', ...)` - pulls SwAV weights from GitHub at runtime; network access to GitHub already confirmed working, no action needed beyond noting it happens automatically.
- 425×425 image resizing for PixCorr/SSIM is NSD-stimulus-specific - should be checked against LAION-fMRI's native 1000×1000px stimuli (probably fine to keep resizing to whatever fixed size, just confirm it's applied consistently to both ground-truth and reconstructed images).

## 5. ROI analysis - needs new ROI source; known upstream bug

`roi_extract.py`, `roi_generate_features.py`, `roi_vdvae_reconstruct.py`, `roi_versatilediffusion_reconstruct.py`.

- `roi_extract.py` reads NSD's specific ROI atlas files (`nsddata/ppdata/subj{XX}/func1pt8mm/roi/*.nii.gz`, specific label codes for V1–V4/floc-faces/floc-words/etc.) - **not transferable**; LAION-fMRI ships its own ROI masks accessible via `get_rois()` / `sub.get_available_rois()` (see context doc §7), so this script needs rewriting against that API rather than NSD's file-based atlas.
- `roi_generate_features.py` carries the same single-GPU fix as §3 where applicable, plus fixed "magic" normalization constants (`*50`, `*9`, `*15`) that the original authors themselves flag (via the README's ROI-analysis caveat) as producing only approximate results due to an unfixed bug. **Decision needed before we invest effort here**: fix the bug ourselves, use it as-is with the caveat disclosed, or exclude ROI analysis from the replication's scope (re:vision's rules only require replicating findings mentioned in the title/abstract - worth checking whether ROI-level results are actually claimed there before committing time to this stage).
- `roi_vdvae_reconstruct.py` needed no GPU-count fix (single-GPU already); confirmed import-clean.

## 6. Statistics - not a code-adaptation question, but a re:vision compliance one

None of the original scripts perform significance testing (`evaluate_reconstruction.py` just prints point-estimate metrics), so there's no original-code statistical logic to adapt. What's required instead, per re:vision's rules (`re-vision-initiative-context.md` §3–4): since LAION-fMRI has only 5 subjects, any comparison we want to report with a p-value must be done as a **subject-level, non-parametric permutation test** (5 subjects means we can't use NSD's typical group-level stats even if the original paper did). This needs to be designed fresh for whichever specific comparisons we choose to report (e.g., "is reconstruction quality above chance?" per subject, analogous to the worked example in the context doc's permutation-testing guide) - it is new analysis code we write, not an adaptation of anything in `original_repo/`.

## Summary table

| Stage | Reusable as-is? | Must change | Effort |
|---|---|---|---|
| Data acquisition/prep | No | Full reimplementation against `laion_fmri` API | High |
| VDVAE | Mostly | Subject-ID assert, `/300` scaling constant | Low |
| Versatile Diffusion | Mostly | Subject-ID assert, `/300` scaling, **2-GPU→1-GPU** | Medium |
| Evaluation | Mostly | `num_test=982` constant, `device=1`→`0` | Low |
| ROI analysis | Partially | New ROI source (LAION-fMRI API), decide on known bug | Medium, scope-dependent |
| Statistics | N/A (new code) | Design permutation tests per re:vision rules | Medium |
