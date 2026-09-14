# The LAION Pretraining Confound

Working analysis of a confound raised on 2026-09-11: Versatile Diffusion, the frozen generative
model at the heart of Brain-Diffuser's stage 2, was pretrained on the same image corpus that most
LAION-fMRI stimuli were drawn from. This document states the problem, estimates its impact, and
sets out the available controls in cost order. Internal working doc, not reviewer-facing. The
proposal (`proposal_draft.md` section 8) carries a short public disclosure derived from this.

## 1. The problem

Brain-Diffuser retrains nothing except two ridge regressions. Every generative and feature-extraction
component is frozen and pretrained. Verified inventory, with sources in `working_repo/`:

| Component | Role | Trained on | Source |
|---|---|---|---|
| VDVAE | stage-1 generative prior | ImageNet 64x64 | `scripts/vdvae_extract_features.py:42` |
| CLIP ViT-L/14 | CLIP-Text and CLIP-Vision conditioning | OpenAI WIT-400M | `versatile_diffusion/lib/model_zoo/clip.py:27` |
| Versatile Diffusion UNet + AutoKL | stage-2 generative prior | **Laion2B-en** | `versatile_diffusion/README.md:53` |
| Inception V3, AlexNet, EfficientNet-B1 | evaluation metrics | ImageNet | `scripts/eval_extract_features.py:95,102,122` |
| SwAV ResNet50 | evaluation metric | ImageNet (self-supervised) | `scripts/eval_extract_features.py:126` |
| CLIP ViT-L/14 | evaluation metric | OpenAI WIT-400M | `scripts/eval_extract_features.py:111` |

LAION-fMRI's 25,052 stimuli break down as 21,871 LAION-natural, 2,570 THINGS, 240 NSD/MSCOCO and 371
out-of-distribution. The figure that matters for us is the **per-subject analysis pool**, not the
full set: the `random_0` split gives 5,833 images per subject, 4,666 train and 1,167 test, and the
test set is 84.7% LAION-derived (989 of 1,167; the rest is 130 THINGS and 48 NSD). Verified by
reading the split files shipped in `laion_fmri/splits/data/sub-*/random_0.json`; the composition is
identical across all five subjects.

**Provenance confirmed (2026-09-14).** LAION-fMRI's natural images were drawn from "a curated
120-million-image subset of LAION-2B filtered to natural photographs (Roth & Hebart, 2025)", stated
in the dataset repository's own docs (`docs/source/stimulus_selection.rst:8`, and repeated in
`stimulus_data.rst:8`, `faq.rst:17`, `experimental_design.rst:10`), which links the pool as
https://huggingface.co/datasets/andropar/relaion2b-natural. That is Re-LAION-2B, the 2024 safety
re-release, which differs from LAION-2B by a negligible number of removed links. Versatile Diffusion
trained on Laion2B-en. Stimuli and training corpus therefore come from the same pool: the overlap is
definitional, not a naming coincidence.

Two residual uncertainties cannot be closed from public information. VD's README says Laion2B-en was
used "with customized data filters" without specifying them, and notes that training runs to less
than one epoch, so not every image in the corpus was necessarily seen. Corpus membership is
established; the per-image hit rate is not.

### How the original study compares

**No component of the original pipeline was trained on MS-COCO.** NSD stimuli come from MS-COCO;
the model corpora are ImageNet, WIT-400M and Laion2B-en. Since COCO images are Flickr-sourced and
widely reposted, some fraction almost certainly appears in the web-scraped corpora, but that fraction
is unquantified and nobody has cleanly measured COCO's presence in LAION-2B, despite its relevance
to the common practice of computing FID on COCO for LAION-trained text-to-image models.

The confound is therefore **new in degree, not in kind**. For Ozcelik and VanRullen, overlap between
the stage-2 prior and the stimuli is incidental and partial. For us it is by construction. That
distinction should be stated plainly rather than glossed as "the original had this too."

## 2. Why this is not classical test-set leakage

The decisive structural point: **the raw stimulus never enters the model.** Everything Versatile
Diffusion receives at reconstruction time is fMRI-derived, namely the predicted CLIP-Vision
embedding, the predicted CLIP-Text embedding, and the VDVAE initial guess. There is no path by
which "this image was in my training set" helps unless the brain-to-latent regression has already
localized the image well enough to retrieve it. If the fMRI carried no information, a perfectly
memorizing model would still produce nothing resembling the stimulus.

The information bottleneck is the regression, and the regression is fit only on the training split.
This is why the core finding, that fMRI supports joint low-level and high-level reconstruction, is
not threatened by the overlap.

## 3. Estimated impact

### 3.1 Memorization proper: low risk

Versatile Diffusion's own README notes that "typical training is less than one epoch" on Laion2B, so
a given image is seen roughly once. The diffusion-memorization literature (Carlini et al. 2023 on
extracting training data, Somepalli et al. 2023 on replication) finds verbatim memorization to be
rare and concentrated on images duplicated many times in the training set, and to require targeted
prompting plus many sampling attempts to elicit. A singly-seen image reached through a noisy
fMRI-predicted embedding is close to a worst case for triggering it. Treat this as a low-probability
contributor. Magnitudes here are recalled from the literature rather than independently verified.

### 3.2 Prior matching: the real concern

The substantive issue is distributional rather than mnemonic. The generative prior is tuned to
exactly the distribution the stimuli were drawn from, so even a weak conditioning signal lands in a
region of image space where the prior's typical sample resembles the ground truth more closely than
it would for a stimulus set the prior had never been shaped by. This inflates absolute metric values
relative to NSD.

**The bias runs in one direction only.** Prior matching can inflate our metrics; it cannot depress
them. That makes the two outcomes asymmetric:

- Values **at or below** the original's: the true value is also at or below, since removing the
  confound only pushes it down. This licenses the **sign** of the conclusion, that the original's
  level was not reached, but not its **magnitude**: the observed shortfall is a lower bound on the
  true shortfall, not an estimate of it.
- Values **comparable or higher**: the ambiguous case, and where the controls below earn their keep.

Both readings hold only with respect to this confound. NSD and LAION-fMRI also differ in field
strength, trial counts, voxel counts and image distribution, and those differences can push in
either direction.

### 3.3 Exposure by metric and by finding

Low-level metrics (PixCorr, SSIM) are driven largely by the VDVAE initial guess, whose ImageNet prior
is equally unmatched to NSD and LAION-fMRI, so they are the least exposed. The six high-level
identification metrics run through Versatile Diffusion and are the most exposed.

| Finding | Exposure | Reasoning |
|---|---|---|
| 1, joint reconstruction | Low | Qualitative; the core claim survives via the section 2 argument |
| 2, quantitative quality | **High** | The replication criterion is a direct metric comparison against NSD values |
| 3, ablations | Low | Internal comparison, all VD-using variants share the prior; inflation reinforces rather than breaks the expected ordering, and Only-VDVAE bypasses VD entirely |
| 4, regression weights by ROI | **None** | Versatile Diffusion is not in the path at all; this analysis reads the regression weights |
| 5, ROI-optimal stimuli | Moderate | Uses VD, but the outcome is qualitative consistency with known selectivity, not a metric value |

The measuring instruments are clean. Every evaluation backbone is ImageNet-trained or WIT-trained,
so no scoring component has seen LAION. The confound sits in the generator, not the ruler.

## 4. Controls, in cost order

### C1. Only-VDVAE contrast, run as a two-by-two with C2 (free, already in the proposal)

The Only-VDVAE ablation from Finding 3 uses only the ImageNet64 VDVAE, with no Versatile Diffusion in
the path, so its prior is indifferent to whether an image came from LAION. That makes it the natural
baseline for isolating what the VD prior contributes.

**Preferred design:** cross C1 with C2 inside our own data. Evaluate {full model, Only-VDVAE} on
{LAION-derived test images, THINGS and MS-COCO test images}. The interaction term is the
prior-matching estimate. Only-VDVAE absorbs exactly the weakness C2 has on its own, namely that the
non-LAION subsets differ in content and may simply be easier or harder to reconstruct, since its
prior does not care which subset an image came from. Everything stays within subject, within dataset
and on the same regressions, so the NSD-versus-LAION-fMRI differences drop out entirely.

**Rejected alternative:** comparing our Only-VDVAE against the original's Table 2 and our full model
against Table 1, then reading the gap between those gaps. This looks equivalent but is not. It
assumes the dataset differences affect stage 1 and stage 2 comparably, and they plausibly do not:
stage 1 regresses onto VDVAE latents and stage 2 onto CLIP embeddings, so if LAION-fMRI supports CLIP
prediction better relative to VDVAE prediction than NSD does, that mimics prior matching exactly. It
also rests on a single-subject reference, since the original's Table 2 is Sub1 only. Use it as a
sanity check, not as the estimate.

**Size of the non-LAION cell (measured, not estimated):** 178 test images per subject (130 THINGS,
48 NSD), the same count for all five subjects, of which 95 are shared by every subject and 510 are
distinct across subjects. Workable for a descriptive metric contrast, thin for anything stronger.

### C2. Image-source contrast within LAION-fMRI (cheap)

LAION-fMRI contains THINGS and NSD/MSCOCO images that were not drawn from LAION. In the `random_0`
test split this is 178 images per subject (130 THINGS, 48 NSD) against 989 LAION-derived ones.
Compare reconstruction metrics on the LAION-derived test images against the non-LAION ones. If prior matching drives results,
LAION-natural should score higher. Confounded on its own by content differences, since THINGS is
object-centric on plain backgrounds and COCO is multi-object scenes, which is why it should be run
as the two-by-two described in C1 rather than alone. Needs no new model and no retrained regressions.

### C3. Prior-only floor (cheap)

Feed Versatile Diffusion the CLIP features predicted for a different image and measure the same
metrics. This quantifies how much of the score comes from "generate a plausible LAION-like image"
versus stimulus-specific information. Note this is a descriptive baseline, deliberately not a
permutation significance test, since the proposal commits to introducing no inferential tests the
original did not run.

### C4. Overlap provenance (resolved 2026-09-14, no longer an action)

Corpus membership is confirmed in section 1: the stimuli's source pool is a curated 120M subset of
LAION-2B, the corpus VD was trained on. Nothing further to run, and no further precision available.

**Per-image matching was never possible, and is now moot.** The stimulus metadata carries only
`image_name`, `dataset`, `participant`, `unique_or_shared` and `n_reps`
(`laion_fmri/stimuli.py:229`), with no LAION sample IDs, URLs or hashes to intersect against
LAION-2B's published metadata. Image names do label provenance at corpus level (`LAION`,
`LAION_cluster`, `THINGS`, `NSD`, `OOD`), which is what makes C2 possible, but they do not identify
the underlying LAION record. So the exact fraction of stimuli VD actually saw, after its undisclosed
filters and sub-epoch training, is not recoverable by any route available to us.

### C5. Alternative diffusion model (expensive, strongest)

Swap the stage-2 generative model for one not trained on LAION. Detailed in section 5.

### C6. Disclose and proceed

`proposal_draft.md` section 8 carries a short disclosure, reproduced verbatim in section 8 below.
This is the floor, not a substitute for C1 through C4.

## 5. The alternative-model option in detail

### 5.1 What the substitute must do

From the original paper's stage-2 description, a replacement needs: a latent diffusion model with an
encoder and decoder, so the stage-1 VDVAE guess can be encoded and partially renoised; an
image-to-image path with partial forward diffusion, 37 of 50 steps in the original; CLIP-Vision
conditioning at patch level, 257x768 from ViT-L/14; CLIP-Text conditioning, 77x768; and joint
conditioning on both, mixed in cross-attention at relative strengths of 0.6 vision and 0.4 text.

### 5.2 There is no drop-in replacement

Versatile Diffusion's dual CLIP-Vision plus CLIP-Text cross-attention pathway is architecturally
unusual, and the models that have it are LAION-trained. A control has to be assembled from two pieces.

**Careful statement of the claim.** "Every CLIP-conditioned large-scale diffusion model is
LAION-trained" is too strong and has a ready counterexample in CommonCanvas, which is
CLIP-text-conditioned and LAION-free. The defensible claim is narrower and turns on the *dual*
pathway: no openly available model combines joint CLIP-Vision and CLIP-Text conditioning with
non-LAION pretraining. CommonCanvas is text-conditioned only, and supplying the image pathway means
IP-Adapter, which is LAION-2B-trained and runs on LAION-trained OpenCLIP encoders (section 5.5). So
substitution cannot cleanly remove LAION, which is why C1 and C2 carry the load instead.

### 5.3 Closest viable stack

**Base: CommonCanvas-XL-C.** SDXL-architecture latent diffusion, trained on CommonCatalog, roughly
70M Creative-Commons Flickr/YFCC images with BLIP-2 synthetic captions, and no LAION data. Supplies
the latent space, VAE encode and decode, and the image-to-image path. Strength 0.75 over 50 steps
reproduces the paper's 37-step partial forward diffusion exactly.

**Image conditioning: IP-Adapter-Plus-SDXL.** Supplies the missing CLIP-Vision pathway using full
patch embeddings rather than a pooled vector, the structural analogue of VD's 257x768 conditioning.
Its `scale` parameter balances image against text conditioning, standing in for VD's 0.6/0.4
cross-attention interpolation. A 22M-parameter adapter that leaves the base model frozen.

### 5.4 What it costs

Both regressions must be retrained against different targets, since the embedding spaces change.
The conditioning mix is an approximation: IP-Adapter adds a scaled parallel cross-attention branch,
whereas Versatile Diffusion linearly interpolates two attention matrices. Both are disclosable
deviations rather than blockers.

### 5.5 Residual caveat, and it matters

Every IP-Adapter variant, SD1.5 and SDXL alike, uses OpenCLIP ViT-H-14 or ViT-bigG-14, and OpenCLIP
is LAION's own reproduction of CLIP, trained on LAION-2B. There is no OpenAI ViT-L/14 option. So this
stack removes LAION from the generator and reintroduces it at the image encoder, an axis on which the
original was clean, since OpenAI's CLIP predates LAION entirely.

This is a much weaker channel. CLIP is discriminative and cannot emit pixels, so having seen an image
during contrastive training cannot let anything reconstruct it; at most it makes that image's
embedding slightly more canonical. Whatever advantage exists applies to training and test splits
alike, shifting the overall quality level rather than the train-to-test relationship the
reconstruction claim rests on. It is a distribution-match effect, not a memorization one.

Partial mitigation: SDXL carries OpenAI's CLIP ViT-L/14 as one of its two text encoders, alongside
OpenCLIP bigG, so the original's exact 77x768 CLIP-T regression target can be kept for that stream.
The vision side has no equivalent escape.

Honest framing for the report: this control isolates the generative prior, which is where the
confound actually operates, while leaving a LAION-trained encoder in the conditioning path. Stating
the limit is better than implying it is airtight.

## 6. Recommended sequence

1. **C4 is done.** Provenance is confirmed (section 1), so the confound is real and the controls
   below are warranted. Start at step 2.
2. **C1 crossed with C2 as standard reporting.** Run as the single two-by-two described in C1, not
   as two separate contrasts. Free or nearly so, and belongs in the report regardless.
3. **C3 if time allows.** A single shuffled-conditioning run gives a useful floor.
4. **C5 only if C1 through C3 leave the picture ambiguous**, and only after the core replication is
   complete. It is a substantial compute and engineering commitment on a 16GiB-RAM machine.

## 7. Open questions

- **Both resolved 2026-09-14.** The stimulus metadata carries no LAION identifiers, so per-image
  matching is impossible (C4); and the Roth & Hebart (2025) 120M corpus is confirmed to be a curated
  subset of LAION-2B (section 1), so the confound is real. What stays unknowable is the per-image
  hit rate, given VD's undisclosed filters and sub-epoch training.
- Is Versatile Diffusion's released checkpoint trained on the full Laion2B-en or a filtered subset?
  The README mentions "customized data filters" without specifying them.
- Worth raising with Ozcelik and VanRullen in the author-contact email, since they may have already
  considered LAION-COCO overlap when selecting Versatile Diffusion.

## 8. Proposal disclosure text (canonical wording)

Reproduced verbatim from section 8 of `re-vision-proposal-brain-diffuser.docx`, which is now the
source of truth for the proposal (authored in Word). Three copy-editing slips are still present in
that document and should be fixed there: "; We aim" (capital after a semicolon), "the LAION-fMRI's
THINGS" (spurious article), and "Furthermore, The original" (capital mid-sentence).

> **Potential confound:** VD is used frozen and was pretrained on LAION-2B, and LAION-fMRI's natural
> images were drawn from a curated 120M subset of that same corpus. This is not classical test-set
> leakage: the raw stimuli never enter the model, as everything VD receives is fMRI-derived (the
> predicted CLIP-Vision and CLIP-Text embeddings and the VDVAE initial guess), so the overlap cannot
> by itself make a reconstruction resemble the particular image a subject viewed. It can, however,
> better match the generative prior to the stimulus distribution, which may inflate absolute values
> relative to NSD. We know of no openly available diffusion model combining VD's joint CLIP-Vision
> and CLIP-Text conditioning with non-LAION pretraining, so substituting the generative model is not
> an option; We aim to assess the effect via two internal controls instead: the Only-VDVAE ablation,
> whose ImageNet prior is equally unmatched to both datasets, and evaluations based on the LAION-
> fMRI's THINGS and MS-COCO stimuli, which are not LAION-derived. Furthermore, The original study
> likely faces a weaker version of the same confound, since MS-COCO images are widely reposted and
> some are likely present in LAION-2B, though we know of no published estimate of how many.

Two points deliberately left out of the proposal for length, but worth making in the report: the
Only-VDVAE control estimates the effect whereas the non-LAION-subset control is confounded by
content differences between the image sets (THINGS is object-centric on plain backgrounds, MS-COCO
is multi-object scenes), and the bias from prior matching runs in one direction only, so a result at
or below the original's supports the qualitative conclusion that its level was not reached, while
the size of the shortfall stays a lower bound rather than an estimate.

## Sources

- CommonCanvas paper: https://arxiv.org/abs/2310.16825
- CommonCanvas-XL-C weights: https://huggingface.co/common-canvas/CommonCanvas-XL-C
- IP-Adapter weights and model card: https://huggingface.co/h94/IP-Adapter
- diffusers IP-Adapter documentation: https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter
