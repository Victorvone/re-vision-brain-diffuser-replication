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

## In progress
- [ ] **Fill out the proposal** (`proposal_draft.md`), section by section, together - 10 sections from the template, verifying each answer before moving on:
  1. Replicator author info (needs input from you: names, institutions, emails, experience self-ratings)
  2. Original study authors (trivial - from the paper)
  3. Scientific subfield tags
  4. Dataset/data used in the original study (NSD specifics - from the paper + repo)
  5. Context of the study (brief background, in our own words)
  6. Main findings to replicate (from the paper's abstract/title)
  7. How each finding was generated (method/outcome measure/stats, from the paper)
  8. How we'll replicate using LAION-fMRI (grounded in `replication-technical-requirements.md`)
  9. Statistics plan - permutation-testing design per re:vision's subject-level requirement
  10. Generalization plan (optional; only if we choose to attempt it - Method 1 + 2 required, Method 3 optional per the rules)

## Not started - before/around proposal submission
- [ ] Decide whether to attempt the optional generalization analysis (affects §10 of the proposal and eligibility for the Generalization Award)
- [x] Decide ROI-analysis scope: **include Finding 3, and attempt to fix the known upstream bug in `roi_generate_features.py` ourselves** (decided while drafting proposal §8, revised from an earlier "disclose as caveat" decision)
- [ ] Sign up on re:vision's site for this study (reserves it for 4 weeks; max 2 teams per study)
- [ ] Send proposal + author-contact email to Ozcelik & VanRullen (template in context doc §9), alongside proposal submission to re:vision
- [ ] Submit proposal by **Sep 15, 2026**, then wait for one round of board feedback

## Not started - after proposal feedback, before the report
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
- [ ] **Full smoke test**: run the adapted pipeline end-to-end on real LAION-fMRI data + real weights for at least one subject, watching for the deeper runtime API-compatibility issues that our lightweight import-only check couldn't catch (real `transformers`/`huggingface_hub` model-loading calls inside `versatile_diffusion/lib/model_zoo/*`), and for OOM issues given the 16GB/no-swap constraint (batch size, precision tuning)
- [ ] Run the full replication across all subjects/all pipeline stages
- [ ] Design and run the subject-level permutation tests for whichever findings we're comparing statistically (per context doc §4)
- [ ] If attempting generalization: run Method 1 (tau split) and Method 2 (cluster_k5), optionally Method 3 (OOD images), per context doc §5
- [ ] Generate comparison plots styled like the original paper's figures (report template requirement)
- [ ] Set up a public GitHub repo for our code (report requirement)
- [ ] Write the report using the report template structure (context doc §8b), including the AI-usage disclosure section
- [ ] Submit report by **Feb 15, 2027**; address any board feedback; submit final revised report by **Apr 30, 2027**
