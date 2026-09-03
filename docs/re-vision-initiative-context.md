# re:vision Initiative - Context Reference

Source: https://re-vision-initiative.org (crawled 2026-09-03: home, `/replication`, `/participate`, `/dataset`, `/statistics`, `/generalization`, `/studies`, plus 4 linked Google Doc templates/examples). This file is a working reference distilled from that site - re-check the live site before final submission in case content changes.

**Contact:** re-vision-initiative@uni-giessen.de
**Also known as:** "the revision initiative" / "the revision challenge" (colon in "re:vision" is stylistic; same project)

## 1. Overview

re:vision - Replication Initiative for Visual Neuroscience - tests the robustness of published findings from **image-related fMRI neuroscience** using a new, densely-sampled dataset, **LAION-fMRI (LfMRI)**. Organized by researchers at the Max Planck Institute for Human Cognitive and Brain Sciences (Leipzig) and Justus-Liebig-University (Giessen). Funded in part by LOEWE (Hessen) and ERC.

Core question: **"Do our findings replicate?"**

Two goals:
1. **Replication** - do published findings hold up on a broader/independent fMRI dataset (LAION-fMRI)?
2. **Generalization** (optional) - do those findings generalize across different image distributions within LAION-fMRI?

**Organizers:** Luca Kämmer (coordination), Josefine Zerbe (dataset creation/coordination), Johannes Roth (platform/dataset engineering), Alessandro Gifford (coordination), Peer Herholz (data infrastructure/docs), Martin Hebart (scientific advisor).

**Review board:** Apurva Ratan Murty (Georgia Tech), Iris Groen (U Amsterdam), Margaret Henderson (CMU), Martin Hebart (JLU/MPI-CBS), Michael Bonner (Johns Hopkins), Radoslaw Cichy (FU Berlin).

## 2. LAION-fMRI dataset facts

- **25,052** unique images total, of which **1,492** are shared (repeated) images
  - LAION-natural: 21,871 images (Roth & Hebart, 2025)
  - NSD/MSCOCO (natural scenes): 240 images
  - THINGS (object images): 2,570 images
  - Out-of-distribution (OOD): 371 images
  - Claimed to have ~2.5× the effective dimensionality of NSD or THINGS in CLIP embedding space
- **5 subjects**, >30 image-viewing sessions each (~60 hrs/subject), 4–12 image repeats
- **7T multi-echo fMRI**, 1.8mm isotropic
- Also includes: extensive retinotopy (phase-encoded, for early visual area delineation), resting-state, diffusion MRI (DTI), high-res T1w anatomical, category-selective functional localizers (faces/scenes/bodies/objects)
- **Format:** BIDS; raw volumes in subject dirs, processed outputs in `derivatives/`. Primary analysis starting point = **GLMsingle single-trial beta estimates** `(n_trials × n_voxels)` per session.
- **Noise ceiling maps**: per-session and cross-session max-explainable-variance estimates (0–100 scale), usable to filter to reliably-driven voxels.
- **ROI masks**: volumetric, surface, and FreeSurfer-label formats.
- **Stimulus derivatives**: metadata, CLIP/DINOv2/PEcore/SigLIP2 embeddings, human + AI captions, object segmentations. Stimulus images: 1000×1000px JPEG, packed in a dataset-wide HDF5 file.
- **License:** fMRI data + public derivatives = CC0 1.0. Raw stimulus images require a Data Use Agreement (requested automatically by the dataloader on stimulus download).
- Interactive 3D brain explorer: https://laion-fmri.hebartlab.com/brain/
- Full technical docs: https://laion-fmri.hebartlab.com/

## 3. Replication rules (from `/replication`)

- Research question: *"Can we replicate findings from image-related fMRI neuroscience?"*
- Focus on **condition-rich fMRI datasets**. Studies can use large public datasets or smaller/private ones, as long as they are in principle replicable with LAION-fMRI.
- Only need to replicate the **main findings mentioned in the title and abstract** of the target paper.
- Only replicate findings that relate to **fMRI data**. Findings based exclusively on behavior, modeling, or other data types can be skipped.
- **Stay as close as possible to the original methods**, including using the original code if public. Contact the original authors with questions about running code or generating necessary metadata.
- **Exclude OOD images** from LAION-fMRI during the replication itself - those are for generalization only.
- **Statistics:** stay as close as possible to the original study's statistics, **but** because LAION-fMRI has only 5 subjects, **all analyses must use subject-level (within-subject) statistical tests**. Studies using between-subject/group-level statistics must be converted to non-parametric permutation testing (see §4).

## 4. Permutation testing guide (from `/statistics`)

**Why:** LAION-fMRI has only 5 subjects, so significance testing must be done at the single-subject level. If the original paper already used single-subject stats, keep that method. If it used group-level stats (t-test/ANOVA across subjects), switch to permutation testing and report results per subject individually. Note: the data are already-estimated single-trial betas, not raw time series - inference happens when fitting/scoring models on the betas.

**5-step method:**
1. **Compute your test statistic** - the one that directly reflects your hypothesis (e.g. model performance, mean activation), once per subject on real data.
2. **Decide what to permute** - permute along the dimension that carries the effect of interest while keeping the rest of the data structure intact. This is the most critical/error-prone step: permuting the wrong dimension (or the data as a whole) invalidates the null distribution and p-values.
3. **Build the null distribution** - repeat step 1 with a different random permutation from step 2, **1,000–10,000 times** (more if feasible).
4. **Compute the p-value**:
   - One-sided: `p = (1 + #(T_perm ≥ T_obs)) / (1 + N_perm)`
   - Two-sided: `p = (1 + #(|T_perm| ≥ |T_obs|)) / (1 + N_perm)`
5. **Correct for multiple comparisons** the same way the original paper did, applied **separately per subject**. Only needed if your replication yields multiple p-values per subject.

**Worked examples** (both use PPA/FFA voxels, per-voxel Pearson r):
- *Ex 1 - Is encoding performance above chance?* Statistic = mean Pearson r (predicted vs actual betas) across PPA+FFA voxels, per subject. Permute: shuffle image indices (breaks image↔response correspondence). Null: no consistent image-specific relationship. p = one-sided, proportion of permuted means ≥ observed.
- *Ex 2 - Does CLIP outperform AlexNet?* Statistic = mean r(CLIP) − mean r(AlexNet). Permute: per image, randomly swap which model's prediction is labeled CLIP vs AlexNet, recompute the difference. Null: model label is arbitrary per image. p = two-sided, proportion of |permuted diffs| ≥ |observed diff|.

Code tutorial referenced but not yet linked with a stable URL on the crawled pages - check the live `/statistics` page's "Tutorial" link if code examples are needed.

## 5. Generalization guide (from `/generalization`) - optional, but required for Generalization Award

Research question: *"Do these results generalize to different image distributions?"* Not required to participate, but only teams that test it are eligible for the **Generalization Award**. If a study only used a narrow image subset (e.g. only food or faces), these methods likely don't apply - stick to replication only in that case.

If you do generalize, **use Method 1 and Method 2**; Method 3 (OOD images) is additional/optional. All splits/images accessed via the dataloader (`laion_fmri` package, see §7).

**Method 1 - Independent within-distribution ("tau" split).** An 80/20 train-test split that maximally covers the image space while minimizing train/test overlap (unlike naive random sampling, it explicitly controls for leakage from overly-similar images).
- *With train-test split:* (1) retrain on the 80% train split, (2) evaluate on the 20% held-out test split, (3) report alongside replication - consistent performance = evidence of generalization.
- *Without train-test split (e.g. RSA):* (1) run the full analysis on the train split, (2) repeat unchanged on the test split (no parameter changes), (3) consistent findings = evidence of generalization.

**Method 2 - Out-of-distribution clusters (`cluster_k5`).** Image set partitioned into 5 distinct embedding-space clusters.
- *With train-test split:* retrain on 4 clusters, evaluate on the held-out 5th; repeat for all 5 folds; average and report the mean.
- *Without train-test split:* run the analysis on 4 clusters excluding the 5th; repeat for all 5 folds; average and report.

**Method 3 - Out-of-distribution images (`ood`).** Curated set of images outside the main distribution, spanning several unconventional categories (dataloader examples reference types like `"shape"`, `"unusual"`, `"cropped"` - check the live `/generalization` page figure or `laion_fmri` docs for the complete/current list of OOD categories, since the site presents them as an image figure rather than plain text).
- *With train-test split:* pick applicable OOD type(s), train on standard images, evaluate on selected OOD images, report alongside replication.
- *Without train-test split:* pick applicable OOD type(s), run the full analysis on those OOD images, compare to replication results.

## 6. Participation process, deadlines & awards (from `/participate` + home)

**Step-by-step:**
1. **Choose a finding** - from the suggested-studies list or propose your own (must be in principle replicable with LAION-fMRI).
2. **Sign up** - via the sign-up button. Deadline: **Sep 15, 2026** or when sign-up capacity is reached. Up to **2 replication attempts per study** are allowed; signing up reserves the study for you for **4 weeks**.
3. **Write a short proposal** (~1,300 words) - using the provided template (§8), within 4 weeks of signup, **latest Sep 15, 2026**. Email it to re:vision **and** to the original authors (first + last author) using the provided email template (§9).
4. **Receive feedback** - the re:vision board reviews and gives brief feedback once, to help refine the approach before analysis starts.
5. **Start replicating** - after feedback, download LAION-fMRI via the `laion_fmri` Python package (§7) and run the pipeline. Contact original authors for help with code/methods as needed.
6. **Write a replication report** - using the provided template (§8). Include methods, results, and interpretation of whether the finding replicated/generalized. Share code in a **public GitHub repo**. Due **Feb 15, 2027**.
7. **Receive one round of review** - board gives feedback only if there are methodological problems or missing info; no extended back-and-forth.
8. **Submit final report** - revised per feedback, due **Apr 30, 2027**. Valid submissions get a chance at consortium co-authorship and a prize. Stay reachable afterward for follow-up questions.

**Timeline summary:**
| Date | Milestone |
|---|---|
| May 18, 2026 | re:vision launch at VSS (kickoff + dataset release, 3.5h hands-on session) |
| **Sep 15, 2026** | Sign-up + proposal submission deadline |
| **Feb 15, 2027** | Report submission deadline |
| Mar 31, 2027 | Board feedback on reports (1st and only round) |
| **Apr 30, 2027** | Final revised report deadline |
| 2027 | Consortium summary paper writing |

**Eligibility / team rules:** Teams of **1–3 researchers**. At least one team member should have fMRI analysis experience. Open to anyone, including non-academic/industry participants. Cannot replicate your own findings (but re:vision can connect you with someone who will).

**Authorship & review:** All teams submitting a **valid** report get the chance at consortium co-authorship on the summary paper (up to 3 people/team), pending sign-off on paper content. Reports are reviewed for methodological clarity/rigor; **negative results are equally valuable** and are included regardless of outcome.

**Awards** (chosen by the board after final submission, based on rigor and presentation quality - independent of whether replication succeeded; split among team members if a team wins):
- **Replication Award - $2,500**: most rigorous, well-executed replication.
- **Generalization Award - $2,500**: most compelling evidence for/against generalization (requires having done the optional generalization analysis).
- **Facilitator Award - $1,000**: for original authors who were especially supportive of a replication team (code, metadata, methodological help). Replicators nominate the original authors for this in their final report.

**Open science:** Code sharing (e.g. GitHub) is strongly encouraged, doesn't need to be polished, just functional. The dataset itself is already public via the package.

## 7. Dataset access - `laion_fmri` Python package API cheatsheet

```bash
python -m pip install "git+https://github.com/ViCCo-Group/LAION-fMRI.git@main"
```
S3 bucket is public, no AWS credentials needed. Raw stimulus images require accepting a Data Use Agreement (prompted automatically).

```python
from laion_fmri.config import dataset_initialize
from laion_fmri.discovery import describe, get_rois, get_subjects
from laion_fmri.download import download, download_stimuli, download_embeddings, download_segmentations, download_captions
from laion_fmri.subject import load_subject
from laion_fmri.splits import get_split_masks
import laion_fmri

dataset_initialize("/path/to/data")     # one-time setup
get_subjects()                          # ['sub-01', 'sub-03', ...]
get_rois("sub-03", category="face")
describe()                              # human-readable bucket summary

download(subject="sub-03", ses="ses-01", n_jobs=4)         # one session, BIDS-aware, idempotent
download(subject="sub-03", ses="averages")                 # subject-level aggregate maps only
download(subject="all", n_jobs=4)                           # everything, all subjects
download(subject="sub-03", ses="ses-01", include_stimuli=True, n_jobs=4)

download_stimuli(); download_embeddings(); download_segmentations(); download_captions()

sub = load_subject("sub-03")
betas = sub.get_betas(session="ses-01")                     # (n_trials, n_voxels), float32
betas_ffa = sub.get_betas(session="ses-01", roi="FFA1")
betas_nc = sub.get_betas(session="ses-01", nc_threshold=0.2)  # well-driven voxels only
trials = sub.get_trial_info(session="ses-01")                # DataFrame: image IDs, conditions...
nc = sub.get_noise_ceiling(session="ses-01")                 # (n_voxels,), 0-100
nc_12rep = sub.get_noise_ceiling(desc="Noiseceiling12rep")
sub.get_sessions(); sub.get_available_rois(); sub.get_available_categories(); sub.get_n_voxels()

stim = laion_fmri.load_stimuli()
stim.metadata.head()                                          # 25,052 image rows
stim.embeddings.get("CLIP", "<image_name>.jpg")
stim.captions.human("<image_name>.jpg")
stim.segmentations.nouns("<image_name>.jpg")

# Subject-level namespaces, aligned to trial index (rows line up with betas/trial tables)
trials = sub.metadata
X = sub.embeddings.all("CLIP", session="ses-01")
img = sub.images.get(42)                                       # PIL image for trial 42

# Train/test splits - pool="shared" or per-subject
train_mask, test_mask = get_split_masks(trials, "random_0", pool="shared")   # baseline (random_0..random_4)
train_mask, test_mask = get_split_masks(trials, "tau", pool="shared")        # Method 1
for k in range(5):
    train_mask, test_mask = get_split_masks(trials, f"cluster_k5_{k}", pool="shared")  # Method 2
train_mask, test_mask = get_split_masks(trials, "ood", pool="shared", ood_types=["shape","unusual","cropped"])  # Method 3
X_train, X_test = betas[train_mask], betas[test_mask]
```

CLI alternative:
```bash
laion-fmri config --data-dir ./laion_fmri_data
laion-fmri info
laion-fmri download --subject sub-03 --ses ses-01 --n-jobs 4
laion-fmri download-embeddings
laion-fmri download-segmentations
laion-fmri download-captions
laion-fmri download-stimuli
```

## 8. Templates (field lists)

Full Google Doc templates/examples are linked from `/participate`:
- Proposal template: https://docs.google.com/document/d/1kYw-1LrrUR67Dqw5LU2QpVwi2b7ZH2LyAl1faD4YoB0/edit
- Proposal example (real, filled): https://docs.google.com/document/d/1IEl-HiRtzUxGwH7kH3KUkBYhSXl8NAFFHcRPnRSPZSY/edit
- Report template: https://docs.google.com/document/d/13Y4ogeLLi1Ih_VXSDonKNESsUk8d8hvYKGE_-_mysCw/edit
- Report example: not yet published ("Coming soon" as of crawl date)
- Author-contact email template: https://docs.google.com/document/d/137MOSQoznXBV3T4XD2x2rk5N2L44mj26ZAJUrkQtIY8/edit

A **pre-filled proposal template for our exact target paper** (Ozcelik & VanRullen 2023) already exists in this repo at `docs/Natural_Scene_Reconstruction_docx_content.txt` (converted from the `.docx` - it's the blank template with the paper's title/DOI slot filled in, not actual proposal content yet).

### 8a. Proposal template (10 sections, word limit ~1,300)
1. **Authors of the replication** (1–3 people): names, institutions, e-mails; self-rated fMRI experience (years), big-data experience (yes/no, e.g. NSD), research-topic familiarity, methodological familiarity (both: "very familiar" / "familiar" / "somewhat familiar" / "not familiar").
2. **Authors of the original study** (copy from paper).
3. **Scientific subfield** - choose from: Brain–model alignment, Cortical organization, Object representations, Scene representations, Feature selectivity, Representational geometry, Encoding models, Decoding/Reconstruction, Stimulus optimization, Cross-subject, or other (multi-select allowed).
4. **Dataset/data used in original study**: which dataset (NSD/THINGS/BOLD5000/private/etc.), preprocessing version, analysis space (native vs standardized), # participants, image subset used, ROIs used, metadata used.
5. **Context of the study** - brief background/significance in your own words, to orient reviewers.
6. **Main findings of the original study** - list each central result you intend to replicate (focus on what's in the abstract or implied by it); note which figure panels relate to each finding.
7. **How each finding was generated** - method, outcome measure (e.g. Pearson r), statistical test, and the original result (significance/effect size, or qualitative assessment).
8. **How you'll replicate these results** - plan per finding using LAION-fMRI; note any new metadata needed (e.g. DNN embeddings) or approximable metadata (e.g. via DNN for behavioral ratings); flag any results you can't replicate with LAION-fMRI.
9. **Statistics** - explain original tests; note that all stats must be within-subject (5 subjects only); if original used within-subject stats, reuse that approach; if group-level, convert to permutation testing (§4) and state what you'll permute across.
10. **Generalization plan** (only required if attempting generalization) - how you'd apply Method 1 & 2 (and optionally 3) per finding; can be limited to the main finding if full coverage is too costly; state which OOD image subset if using Method 3.

### 8b. Report template (8 sections + facilitator nomination)
1. **Authors of the replication** - names, affiliations, conflicts of interest.
2. **Main findings of the original study** - reuse from proposal + add any findings not previously mentioned.
3. **Methods** - summarize how the replication was conducted; how methods were translated to LAION-fMRI; any meaningful deviations.
3\*. **Results of the replication** *(numbered 3 again in the source template)* - per finding: same outcome measures as original (qualitative result, effect size, p-value); p-values reported per subject; direct comparison to original results; plots styled similarly to the original (structure/proportions/colors) for easy comparison; interpretation rating per finding: "fully" / "mostly" / "partially" / "mostly not" / "not at all" / "unclear"; share code as public GitHub repo if possible.
4. **Results of the generalization** (if applicable) - same reporting structure as replication results; explicitly compare generalization vs replication (does the effect pattern change? does performance change even if the pattern holds?); note what's the same vs different vs the original study; interpretation rating per finding using the same 6-point scale.
5. **Conclusion** - state whether the original study's overall interpretation still holds (same 6-point scale), explain how/why the interpretation changes, and give any intuition for why findings did/didn't replicate.
6. **Limitations** - limitations of your own replication attempt; any divergence from original methods and why.
7. **Supplementary material** - any additional results/plots/thoughts (no length limit).
8. **AI usage** - disclose how/to what extent AI was used across: understanding the paper, digesting original code, conducting the replication, writing the report. Must validate all AI output yourself.
- **Facilitator award nomination** (optional): whether to nominate the original authors for the $1,000 award, with justification and contact details.

### 8c. Real filled proposal example - pattern reference
Roth & Kämmer's proposal to replicate Conwell et al. 2024 ("A large-scale examination of inductive biases shaping high-level visual representation in brains and machines", https://doi.org/10.1038/s41467-024-53147-y) is a useful structural pattern: NSD dataset, OTC ROI, veRSA/cRSA methods, 3 findings (CNN vs Transformer equivalence; task-objective equivalence; training-diet dominance) each with method/outcome measure/stat test/result spelled out, a within-subject permutation plan per finding (permute predicted-vs-true betas between conditions at the image level), and a generalization plan using all 3 methods against the primary finding only (for compute reasons).

## 9. Author-contact email template (send alongside proposal)

> Dear [Original Authors],
> I am writing as part of re:vision, a replication initiative for image-related neuroscience. The goal of the initiative is to test whether findings from fMRI studies that visually present images can be replicated using the LAION-fMRI dataset, a new broadly sampled 7T fMRI dataset.
> Our team is planning to replicate findings from your paper, "[paper title]." As part of the initiative, we are preparing a short proposal outlining our planned replication approach, which is attached here for your information and feedback.
> We would be grateful for any comments you may have on our proposal. We would also be grateful if we could potentially contact you in the future for help regarding methodological details, code, metadata, or other aspects that would help us stay as close as possible to the original analysis. We would like to faithfully replicate the effort you had in your original study.
> re:vision also includes a $1,000 Facilitator Award for original authors of replicated papers who support replication teams, for example by helping with code, metadata, or methodological clarification. So, beyond helping improve science, for your help you could also get a cash reward.
> Thank you for considering this. We would be very happy to receive any feedback or pointers you are willing to share.
> Best regards,
> [replication team names]
> [Institution]

## 10. Our target study on the suggested-studies list

**"Natural scene reconstruction from fMRI signals using generative latent diffusion"** - Furkan Ozcelik, Rufin VanRullen (2023). https://doi.org/10.1038/s41598-023-42891-8

- Listed category tags: **Decoding/Reconstruction**; dataset: **NSD**.
- re:vision's one-line summary of it: *"Trained a model to reconstruct both low-level layout and more semantic/high-level aspects of complex natural scenes."*
- Official code ("Brain-Diffuser") is already cloned locally at `original_repo/` (from https://github.com/ozcelikfu/brain-diffuser) - per the replication rules (§3), we should reuse this original code where possible rather than reimplementing from scratch. It covers: NSD data prep, VDVAE first-stage reconstruction, Versatile Diffusion second-stage reconstruction (conditioned on predicted CLIP-Text + CLIP-Vision features), quantitative evaluation, and ROI analysis.
- Full paper text has been extracted to `docs/Ozcelik_VanRullen_2023_NaturalSceneReconstruction.txt` (15 pages).
- The `docs/Natural scene reconstruction....docx` file in this repo turned out to be the **blank re:vision proposal template** with this paper's title pre-filled (not the paper itself) - its content is in `docs/Natural_Scene_Reconstruction_docx_content.txt` and gives us the exact section headers/prompts we need to fill in for our own proposal (§8a duplicates this in condensed form).

## 11. Open items / things to verify closer to submission

- The full list of OOD image *type names* (Method 3) is only shown as a figure on `/generalization`, not as plain text; the `laion_fmri` dataloader docs/code examples reference `"shape"`, `"unusual"`, `"cropped"` - confirm the complete set once the package is installed (`laion-fmri info` or docs site).
- The permutation-testing **code tutorial** link on `/statistics` should be followed once we're implementing statistics, in case it has copy-pasteable code beyond the two worked examples above.
- Confirm current sign-up capacity / whether the study is still available to claim (max 2 teams per study) before finalizing our proposal.
