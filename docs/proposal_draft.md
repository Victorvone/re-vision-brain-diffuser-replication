# re:vision Proposal: Natural scene reconstruction from fMRI signals using generative latent diffusion

DOI: https://doi.org/10.1038/s41598-023-42891-8
Original authors: Furkan Ozcelik, Rufin VanRullen (2023), *Scientific Reports* 13:15666
Word limit ~1,300 words

Status: **Reviewed version (2026-09-09)**, as approved by Victor and his professor. Answer text below is
verbatim from that review; only the markdown scaffolding (section headers, blockquoted template prompts,
list markers) is repo formatting. Do not alter answer wording without an explicit instruction.
Each section keeps the original template prompt verbatim (from
`re-vision-proposal-template-brain-diffuser-prefilled.txt`) followed by the answer.

---

## 1. Information about the authors of the replication (1-3 people)

> Names of the replicators.
> Institutions of the replicators.
> E-mail addresses of the replicators.
> fMRI experience: Years of experience with fMRI analysis.
> Big data experience: Have you worked with large public datasets (e.g. NSD) before?
> Research experience: How familiar are you with the topic/area of research of this study? "Very familiar", "familiar", "somewhat familiar", "not familiar"
> Methodological experience: How familiar are you with the analysis methods used? "Very familiar", "familiar", "somewhat familiar", "not familiar"

**Answer:**

- **Victor von Eisenhart-Rothe**
  - Institution: Freie Universität Berlin
  - E-mail: victorvoneisenhart@gmail.com
  - fMRI experience: 0 years
  - Big data experience: Yes
  - Research experience: Familiar
  - Methodological experience: Familiar
- **Lucas Nunn**
  - Institution: Freie Universität Berlin
  - E-mail: luc.nunn@gmail.com
  - fMRI experience: 1 year
  - Big data experience: Yes
  - Research experience: Somewhat familiar
  - Methodological experience: Somewhat familiar
- **Adrien Doerig**
  - Institution: Freie Universität Berlin
  - E-mail: adrien.doerig@fu-berlin.de
  - fMRI experience: 10+ years
  - Big data experience: Yes
  - Research experience: Very familiar
  - Methodological experience: Very familiar

## 2. Authors of the original study

> Authors of the original study (just copy from the paper): Names of the original authors of the study.

**Answer:**

Furkan Ozcelik, Rufin VanRullen

## 3. Scientific subfield

> How would you categorize the scientific subfield of this study: "Brain–model alignment", "Cortical organization", "Object representations", "Scene representations", "Feature selectivity", "Representational geometry", "Encoding models", "Decoding/Reconstruction", "Stimulus optimization", "Cross-subject", or something else (you can choose multiple).

**Answer:**

Decoding/Reconstruction

## 4. Dataset / data used in the original study

> Dataset / data used in the original study (specify if only a subset was used):
> Dataset: Which dataset set was used in the study (e.g. NSD, THINGS, BOLD5000, private)?
> Preprocessing: Which preprocessed version of the dataset was used?
> Analysis space: Was the analysis performed in native subject space or a standardized space?
> Participants: How many participants were used from the dataset?
> Images: Which subset of images was used (e.g. only shared images, only images of food)?
> ROIs: Which ROIs were used (e.g. occipitotemporal cortex (OTC) or whole brain)?
> Metadata: Which metadata was used (e.g. image captions, DNN embeddings, similarity ratings)?

**Answer:**

- **Dataset:** Natural Scenes Dataset (NSD); participants viewed natural images drawn from MS-COCO.
- **Preprocessing:** NSD's provided single-trial GLM beta weights, computed with fitted HRF plus GLMdenoise and ridge-regression denoising.
- **Analysis space:** Native subject space (functional, 1.8mm resolution).
- **Participants:** 4 of NSD's 8 subjects (sub1, sub2, sub5, sub7), who completed all scanning sessions.
- **Images:** The training set contained 8,859 images and 24,980 fMRI trials (up to 3 repetitions for each image), and the test set contained 982 images and 2,770 fMRI trials. fMRI trials were averaged for images with multiple repetitions. Test images are common for all subjects, while training images are different.
- **ROIs:** The NSDGeneral mask (1.8mm), a broad visual-cortex ROI spanning early through higher visual areas, not restricted to a single region. A secondary analysis, "ROI-optimal stimuli," additionally probes visual-field ROIs V1-V4, functional localizer ROIs for faces/words/places/bodies, and eccentricity bands (see Finding 5 in 6.).
- **Metadata:** COCO captions per image, used for CLIP-Text conditioning (5 per image, embeddings averaged).

## 5. Context of the study

> Give a brief context / background of the study and its significance in your own words.
> This is meant for the reviewers to have an easier time placing the study and the findings you will try to replicate.

**Answer:**

Neural decoding research aims to reconstruct what a person perceived from their brain activity alone. Earlier fMRI-based image reconstruction methods succeeded at capturing either low-level visual properties (shape, layout, texture) or high-level semantic content (object categories, scene descriptions), but rarely both together for complex natural scenes. This study introduced "Brain-Diffuser", a two-stage framework combining a hierarchical VAE (VDVAE) for low-level structure with a pretrained latent diffusion model (Versatile Diffusion), conditioned on fMRI-predicted CLIP-Text and CLIP-Vision features, to reconstruct both aspects jointly. On the NSD benchmark, the authors report outperforming previous models both qualitatively and quantitatively, pushing existing brain-decoding benchmarks forward, with potential impact on both applied (e.g. brain-computer interfaces) and fundamental neuroscience.

## 6. Main findings of the original study

> List each central result of the study that you intend to replicate. Focus on the results mentioned in the abstract of the paper (or implied by the abstract's content).
> Please indicate which figure panels are related to each finding.

**Answer:**

1. **Joint low-level and high-level reconstruction of complex natural scenes.** Brain-Diffuser reconstructs both low-level properties (layout, shape) and high-level semantic content together for complex scenes. (Fig. 3: example reconstructions; Figs. 1-2: the two-stage model producing them.)
2. **Strong quantitative reconstruction quality on the NSD benchmark.** Brain-Diffuser achieves strong scores on both low-level and high-level image-quality metrics (PixCorr, SSIM, and 2-way identification accuracy across six feature spaces: AlexNet(2), AlexNet(5), Inception, CLIP, EffNet-B, SwAV) on the shared NSD test set. (Table 1: quantitative metrics.)
   Note: the original paper frames this comparatively, as outperforming three prior reconstruction methods (Lin et al., Takagi et al., Gu et al.; Figs. 5-6); we cannot access those models' outputs for LAION-fMRI, so we replicate Brain-Diffuser's own absolute metrics instead (see 8.).
3. **Component contributions established by ablation.** Each component of the two-stage model contributes distinctly, and the full model is the best joint compromise: Only-VDVAE (stage 1 alone) is best on all low-level measures but worst by a large margin on all high-level ones; Brain-Diffuser without VDVAE (stage 2 alone) shows the inverse pattern; removing CLIP-Text or CLIP-Vision degrades performance relative to the full model. (Table 2: quantitative comparison; Fig. 7: qualitative examples.)
4. **Division of labour across brain regions.** An ROI analysis of the regression weights shows that early visual regions (V1-V4) are more informative about the VDVAE features, while category-selective higher regions (Face, Word, Place, Body) carry more information about the CLIP features; the CLIP-Vision versus VDVAE difference runs in the same direction as the CLIP-Text versus VDVAE difference, but is much weaker. (Fig. 8.)
5. **ROI-optimal stimuli consistent with known functional selectivity.** When the trained model is applied to synthetic fMRI patterns that activate specific regions-of-interest (visual-field ROIs V1-V4, functional localizer ROIs for faces/words/places/bodies, and eccentricity bands), the resulting reconstructions show scene content consistent with each ROI's known neuroscientific selectivity. (Fig. 9: individual ROIs; Fig. 10: ROI combinations; Fig. 11: eccentricity bands.)

## 7. How were each of these findings generated (method and outcome measure)

> For each of the results listed above, briefly explain which method, outcome measure (e.g. pearson correlation), and statistical test were used to generate that finding.
> Please indicate the result of each finding in terms of the significance value and effect size (if reported in the paper), or qualitative assessment in the case of qualitative results.

**Answer:**

**Finding 1 (joint low-level+high-level reconstruction):**
- Method: a two-stage model. Stage 1 uses ridge regression from fMRI activity to VDVAE latent features, decoded into an initial low-level reconstruction. Stage 2 uses ridge regression from fMRI activity to CLIP-Vision and CLIP-Text features, which condition a diffusion model (Versatile Diffusion) that refines the stage-1 reconstruction into the final image.
- Outcome measure: qualitative visual comparison of reconstructions to ground-truth test images (Fig. 3); no statistical test.
- Result: qualitative. The authors report reconstructions "preserve most of the layout and semantic information" though not pixel-perfect, with failure cases documented separately (Fig. 4).

**Finding 2 (strong quantitative reconstruction quality):**
- Method: same regression+reconstruction model as Finding 1, evaluated by comparing each test-set reconstruction to its ground-truth image.
- Outcome measure: 8 point-estimate image-quality metrics (no significance testing), covering low-level similarity (pixel/structural correlation) and high-level similarity (identification accuracy across several pretrained vision/language networks).
- Result (Table 1): PixCorr=0.254, SSIM=0.356, AlexNet(2)=94.2%, AlexNet(5)=96.2%, Inception=87.2%, CLIP=91.5%, EffNet-B=0.775, SwAV=0.423, the best (or tied-best) value among all compared models on every metric.

**Finding 3 (component contributions established by ablation):**
- Method: four ablated variants of the full model (Only-VDVAE, w/o VDVAE, w/o CLIP-Text, w/o CLIP-Vision), evaluated on Sub1's test set with the same procedure as Finding 2.
- Outcome measure: the same 8 point-estimate metrics (Table 2), plus qualitative inspection of reconstructions (Fig. 7); no significance testing.
- Result (Table 2): Only-VDVAE is best on low-level measures (PixCorr=0.358, SSIM=0.437) but worst on high-level ones (Inception=77.0%, CLIP=71.1%, EffNet-B=0.906, SwAV=0.581); w/o VDVAE is worst on low-level (PixCorr=0.143, SSIM=0.302) while among the best on high-level (Inception=87.3%, CLIP=92.6%); the full model (PixCorr=0.305, SSIM=0.367, Inception=87.8%, CLIP=92.5%, EffNet-B=0.768, SwAV=0.415) is the optimal compromise across both. Qualitatively, Only-VDVAE produces vague silhouettes and w/o VDVAE loses object layout.

**Finding 4 (division of labour across brain regions):**
- Method: for each voxel in 8 ROIs (V1-V4 from population receptive field mapping; Face, Word, Place and Body ROIs from functional localizers), the strength (L1 norm) of the ridge regression weights was computed for the CLIP and the VDVAE features and expressed as a percentile; results are reported as the CLIP-minus-VDVAE difference, to control for differences in ROI size, overall activity and noise level.
- Outcome measure: difference in regression-weight percentile, averaged over voxels per ROI, with error bars showing the standard error of the mean across the 4 subjects (Fig. 8); no statistical test.
- Result: descriptive. The difference is negative (VDVAE-weighted) for V1-V4 and positive (CLIP-weighted) for the category-selective ROIs, with the CLIP-Text contrast substantially larger than the CLIP-Vision contrast.

**Finding 5 (ROI-optimal stimuli):**
- Method: synthetic activation patterns for each ROI are constructed (without real fMRI data) and passed through the trained regression and reconstruction model to generate images.
- Outcome measure: qualitative visual inspection of the resulting synthetic reconstructions, judged against each ROI's known functional selectivity; no statistical test.
- Result: qualitative. Reconstructions are reported as "consistent with neuroscientific knowledge" (e.g. face-like content for the Face-ROI, scene/place content for the Place-ROI).

## 8. How to replicate these results

> Outline how you intend to replicate each result using LAION-fMRI.
> Is there new metadata that has to be generated for the replication (e.g. DNN embeddings, luminosity ratings etc.)?
> Is there any necessary metadata that can be approximated using DNNs (e.g. behavioural ratings).
> Are there main results in the study that you cannot replicate using LAION-fMRI?

**Answer:**

**Finding 1 (joint low-level+high-level reconstruction):** We will prepare LAION-fMRI data the same way NSD data was prepared (train/test split, excluding out-of-distribution images), run the same two-stage model on LAION-fMRI's data, and compare reconstructions to ground truth qualitatively, as in the original study.

**Finding 2 (strong quantitative reconstruction quality):** We will evaluate Finding 1's reconstructions using the same quantitative metrics, reported per subject (LAION-fMRI has 5 subjects), and compare them against the values the original study reports for NSD, asking whether Brain-Diffuser trained on LAION-fMRI reaches comparable metric values (see 9.).

**Finding 3 (component contributions established by ablation):** We will run the same four ablation variants on LAION-fMRI, per subject, and compute the same eight metrics for each, testing whether the ordinal pattern across variants is preserved. These variants are internal to Brain-Diffuser and require no external models, so this comparison transfers in full.

**Finding 4 (division of labour across brain regions):** We will recompute the regression-weight percentile analysis from our LAION-fMRI-trained regressions, using the dataset's own retinotopically defined early visual ROIs and its category-selective localizer ROIs, and test whether the same early-versus-higher division holds, following the original analysis exactly.

**Finding 5 (ROI-optimal stimuli):** We will include this finding using LAION-fMRI's own ROI definitions, qualitatively assessing whether reconstructions match known functional selectivity, as in the original study. The original code has a disclosed bug limiting precision for most ROIs; we will attempt to fix it before running this finding.

**New metadata needed:** None. LAION-fMRI already provides everything the model requires: images, captions, ROI definitions, retinotopy and functional localizers.

**Metadata approximable via DNNs:** Not applicable.

**Results we cannot replicate using LAION-fMRI:** The comparison against three other reconstruction models (Finding 2) will not be replicated, as explained in 6.; we instead compare against the original study's own reported performance.

## 9. Statistics

> Explain the relevant statistical tests used in the original study.
> Since LAION-fMRI only has 5 subjects all statistics have to be done within-subject, not at the group level!
> If the study already uses within-subjects statistics, use their statistical approach.
> If the study uses group-level statistics, please change it to permutation-based single-subject statistics. More information, including a tutorial, can be found on our website.
> Please indicate for each finding if the statistics of the original study had to be changed in meaningful ways. If you run permutation, please indicate across which parameters you intend to run the permutation. Please consult the tutorial for details.
> Please reach out to us if you are unsure or think that a permutation approach is not applicable to the finding(s) you are replicating.

**Answer:**

None of the five findings involved formal statistical testing in the original study (see 7.): every result is either qualitative or a point estimate reported without variance estimates or significance tests. Following the instruction to stay as close as possible to the original study's approach, we do not introduce inferential tests that the original study did not perform. No finding therefore requires a change of statistical method, and no permutation scheme applies.

Findings 2 and 3: we report the same eight metrics descriptively, computed per subject on LAION-fMRI, and compare them directly against the values the original study reports on NSD in Tables 1 and 2. The replication criterion for Finding 2 is whether Brain-Diffuser trained on LAION-fMRI attains metric values comparable to those reported on NSD; for Finding 3, whether the ordinal pattern across the ablation variants is preserved (Only-VDVAE highest on low-level and lowest on high-level metrics, w/o VDVAE the inverse, full model best jointly). The four 2-way identification metrics have a defined chance level of 50%, which provides an interpretable floor. Absolute magnitudes are not strictly comparable across datasets, which differ in subjects, image distribution, field strength and trial counts.

Finding 4: replicated exactly as in the original, with the regression-weight differences averaged over voxels per ROI and reported across subjects with standard-error bars as in Fig. 8. The comparison is the sign and the rank ordering of the CLIP-minus-VDVAE difference across ROIs.

Findings 1 and 5 are qualitative in the original study and remain so here.

## 10. How to test generalization of these findings (required only for generalization)

> If you want to also conduct the generalization, please use method 1 and method 2 that are explained in our generalization guide. Additionally you can use method 3, which involves using a subset of the OOD images included in LAION-fMRI. If you are not doing generalization, you do not need to fill this in and can leave it empty.
> If testing the generalization of all findings listed above is too time-consuming or computationally intense, you can limit it to the main finding. If that is the case, please state so here and explain.
> Please indicate briefly how you would use methods 1 & 2 (and method 3) to test the generalizability of the finding(s).
> If you use method 3 (the OOD images), please indicate which subset of OOD images you are using.
> These three methods may not apply to findings that only use a subset of the images (i.e. only images of food).

**Answer:**

We plan to attempt generalization for Finding 2, using re:vision's Method 1 (a broad-coverage train/test split for unseen images) and Method 2 (out-of-distribution image clusters). We will retrain on each split's training portion, evaluate on the held-out portion, and compare the resulting metrics to our main replication. We will additionally test Method 3, using LAION-fMRI's "shape," "unusual," and "cropped" out-of-distribution image categories; we include all three rather than a single category, since Finding 2 is a general reconstruction-quality metric not tied to any one visual property.

Finding 1 will be generalized using the same procedure, and assessed qualitatively (cf. 7.). Findings 3 and 4 will not be generalized, to keep the computational cost manageable, as the template permits limiting generalization to the main finding. Finding 5 does not use a held-out image split, so generalisation is not applicable.
