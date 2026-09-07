# Replication Task Overview

A single checkable list of everything left to do for our re:vision replication of Ozcelik & VanRullen (2023) "Brain-Diffuser", from here through final submission. Cross-references `re-vision-initiative-context.md` (rules/deadlines) and `replication-technical-requirements.md` (per-stage technical detail).

**Deadlines** (from `re-vision-initiative-context.md` §6): proposal due **Sep 15, 2026**; report due **Feb 15, 2027**; final revised report due **Apr 30, 2027**.

## Done
- [x] Crawled re:vision site rules/templates/dataset docs → `re-vision-initiative-context.md`
- [x] Extracted target paper full text → `ozcelik-vanrullen-2023-brain-diffuser-paper-fulltext.txt`
- [x] Extracted pre-filled proposal template → `re-vision-proposal-template-brain-diffuser-prefilled.txt`
- [x] Confirmed target paper is on re:vision's suggested-studies list (Decoding/Reconstruction, NSD)
- [x] Investigated `original_repo/` in full: what's present (code only, no data/weights), what's NSD-specific, all hardcoded assumptions per pipeline stage
- [x] Audited this machine's hardware/environment (AMD ROCm GPU, no CUDA, 16GB RAM/no swap, 1.5TB disk free, network OK)
- [x] Created `working_repo/` as an untouched copy of `original_repo/` - all future edits happen there only
- [x] Extended `ml-env` with the packages Brain-Diffuser needs beyond what was already installed (`nibabel`, `omegaconf`, `easydict`, `kornia`, `h5py`, `scikit-image`, `einops`, `clip`, `opencv-python-headless`)
- [x] Verified all 17 entry-point scripts in `working_repo/` import cleanly under this environment (lightweight check - no real weights/data yet); findings in `replication-technical-requirements.md` §0
- [x] Wrote up per-stage technical adaptation requirements → `replication-technical-requirements.md`
- [x] Ran runtime smoke tests (not just import checks) for VDVAE and Versatile Diffusion against real downloaded checkpoints, synthetic inputs (2026-09-07) - both pass end-to-end; fixed a real fp32-instantiation OOM bug found along the way (`replication-technical-requirements.md` §0 update, §3 update)
- [x] Pushed the repo to GitHub as a private backup (`https://github.com/Victorvone/re-vision-brain-diffuser-replication`, 2026-09-07) - `original_repo/` and `working_repo/` folded in as plain subdirectories, large checkpoints gitignored. Note: this is a private working backup, not the same as the **public** repo the report requirement further down calls for - that's still a separate, later action.

## In progress
- [x] **Fill out the proposal** (`proposal_draft.md`) - all 10 sections now have drafted answers, verified section by section together:
  1. Replicator author info
  2. Original study authors
  3. Scientific subfield tags
  4. Dataset/data used in the original study
  5. Context of the study
  6. Main findings to replicate
  7. How each finding was generated
  8. How we'll replicate using LAION-fMRI
  9. Statistics plan
  10. Generalization plan (optional - included in the current draft)
  - **Remaining before this can be considered done:** word count is 1,277 / 1,300 words including the optional §10 - technically under the limit but with almost no margin (checked 2026-09-07); needs a final read-through/trim pass, not just a word-count check. If §10 (generalization, ~179 words) ends up dropped, there's much more room.
  - **Time pressure:** proposal is due **Sep 15, 2026** - as of this writing that's about a week out, and three unstarted items below (site sign-up, author-contact email, submission) still need to happen after the draft is finalized.

## Not started - before/around proposal submission
- [ ] Decide whether to attempt the optional generalization analysis (affects §10 of the proposal and eligibility for the Generalization Award)
- [x] Decide ROI-analysis scope: **include Finding 3, and attempt to fix the known upstream bug in `roi_generate_features.py` ourselves** (decided while drafting proposal §8, revised from an earlier "disclose as caveat" decision)
- [ ] Sign up on re:vision's site for this study (reserves it for 4 weeks; max 2 teams per study)
- [ ] Send proposal + author-contact email to Ozcelik & VanRullen (template in context doc §9), alongside proposal submission to re:vision
- [ ] Submit proposal by **Sep 15, 2026**, then wait for one round of board feedback

## Not started - after proposal feedback, before the report
- [ ] **Confirm the VM memory/CPU allocation has been raised** (or plan around 16GB if not) - this machine is currently capped to 16GB RAM / 8 CPU threads by its host container, despite the underlying GPU supporting ~96GB; a request was handed to the VM admin 2026-09-07 but resolution wasn't confirmed as of this writing. See `replication-technical-requirements.md` §0 update. Blocks realistic planning for regression-matrix memory sizing and batch sizes below.
- [ ] Install `laion_fmri` Python package and download the data (per context doc §7) - actual data volume unknown until we check dataset size for our subjects/sessions of interest
- [ ] Write LAION-fMRI data-prep script (replaces `download_nsddata.py`/`prepare_nsddata.py`) producing the same array shapes/format the downstream scripts expect (`replication-technical-requirements.md` §1)
- [ ] Apply code patches in `working_repo/` (never `original_repo/`):
  - [ ] Relax `assert sub in [1,2,5,7]` across all 15 scripts to accept LAION-fMRI subject IDs
  - [ ] Re-derive/replace the `/300` fMRI-beta scaling constant for LAION-fMRI's beta scale (§2–3)
  - [ ] Consolidate the hardcoded `cuda(0)`/`cuda(1)` dual-GPU split to single-device in `versatilediffusion_reconstruct_images.py` and `roi_versatilediffusion_reconstruct.py` (§3, §5)
  - [ ] Replace hardcoded `num_test=982` with LAION-fMRI's actual test-split size (§4)
  - [ ] Rewrite `roi_extract.py` against LAION-fMRI's ROI API (§5)
  - [ ] Diagnose and fix the known bug in `roi_generate_features.py` (magic normalization constants flagged in §5 of `replication-technical-requirements.md`) before running Finding 3
- [ ] Download pretrained weights (VDVAE ImageNet64 checkpoints from OpenAI; Versatile Diffusion checkpoints from HuggingFace `shi-labs/versatile-diffusion`) into `working_repo/vdvae/model/` and `working_repo/versatile_diffusion/pretrained/`
- [ ] **Full smoke test**: run the adapted pipeline end-to-end on real LAION-fMRI data + real weights for at least one subject, watching for the deeper runtime API-compatibility issues that our lightweight import-only check couldn't catch (real `transformers`/`huggingface_hub` model-loading calls inside `versatile_diffusion/lib/model_zoo/*`), and for OOM issues given the 16GB/no-swap constraint (batch size, precision tuning). Note the VDVAE/VD *model-loading* half of this is already done with synthetic data (see "Done" above) - what's left here is real data + the regression stage. Specifically check the sklearn Ridge regression memory footprint before assuming it fits: weight matrices scale as `n_voxels × latent_dim`, and CLIP-Vision's latent is ~197k-dim - using NSD's ~15-20k voxel count as a rough proxy, that matrix alone could be ~10-15GB in float32, a plausible second OOM risk distinct from the one already fixed.
- [ ] Run the full replication across all subjects/all pipeline stages
- [ ] Design and run the subject-level permutation tests for whichever findings we're comparing statistically (per context doc §4)
- [ ] If attempting generalization: run Method 1 (tau split) and Method 2 (cluster_k5), optionally Method 3 (OOD images), per context doc §5
- [ ] Generate comparison plots styled like the original paper's figures (report template requirement)
- [ ] Set up a public GitHub repo for our code (report requirement)
- [ ] Write the report using the report template structure (context doc §8b), including the AI-usage disclosure section
- [ ] Submit report by **Feb 15, 2027**; address any board feedback; submit final revised report by **Apr 30, 2027**
