# re:vision Proposal: Natural scene reconstruction from fMRI signals using generative latent diffusion

Source of truth: `re-vision-proposal-brain-diffuser.docx` (authored in Word, includes the figure/table
appendix). This file is a plain-text mirror of that document for diffing and review; edit the .docx,
then regenerate this mirror.

DOI: https://doi.org/10.1038/s41598-023-42891-8

Word limit ~1,300 words

## 1. Information about the authors of the replication (1-3 people)

Victor von Eisenhart-Rothe, Freie Universität Berlin, victorvoneisenhart@gmail.com. Experience - fMRI: 0 years; big data: yes; research and methodological: familiar.

Lucas Nunn, Freie Universität Berlin, luc.nunn@gmail.com.  Experience - fMRI: 1 year; big data: yes; research and methodological: somewhat familiar.

Adrien Doerig, Freie Universität Berlin, adrien.doerig@fu-berlin.de.  Experience - fMRI: 10+ years; big data: yes; research and methodological: very familiar.

## 2. Authors of the original study (just copy from the paper)

Furkan Ozcelik, Rufin VanRullen

## 3. How would you categorize the scientific subfield of this study

Decoding/Reconstruction

## 4. Dataset / data used in the original study (specify if only a subset was used)

Dataset: Natural Scenes Dataset (NSD); participants viewed natural images drawn from MS-COCO.

Preprocessing: NSD's single-trial GLM beta weights (fitted HRF, GLMdenoise, ridge-regression denoising).

Analysis space: Native subject space (functional, 1.8mm resolution).

Participants: 4 of NSD's 8 subjects (sub1, sub2, sub5, sub7), who completed all scanning sessions.

Images: 8,859 training images and 24,980 fMRI trials (up to 3 repetitions per image, averaged); 982 test images and 2,770 trials. Test images are common to all subjects, training images are not.

ROIs: The NSDGeneral mask (1.8mm), a broad visual-cortex ROI spanning early through higher visual areas, not restricted to a single region. The secondary "ROI-optimal stimuli" analysis additionally probes visual-field ROIs V1-V4, functional localizer ROIs for faces/words/places/bodies, and eccentricity bands (Finding 5).

Metadata: COCO captions per image, used for CLIP-Text conditioning (5 per image, embeddings averaged).

## 5. Context of the study

Neural decoding research aims to reconstruct what a person perceived from their brain activity alone. Earlier fMRI-based reconstruction methods captured either low-level visual properties (shape, layout, texture) or high-level semantic content (object categories, scene descriptions), but rarely both together for complex natural scenes. This study introduced "Brain-Diffuser", a two-stage framework (described in 7.) that reconstructs both jointly and reportedly outperforms previous models on the NSD benchmark, both qualitatively and quantitatively, pushing brain-decoding benchmarks forward, with potential impact on applied (e.g. brain-computer interfaces) and fundamental neuroscience.

## 6. Main findings of the original study

Joint low-level and high-level reconstruction of complex natural scenes. Layout and shape are recovered together with semantic content. (Fig. 3: example reconstructions; Figs. 1-2: the two-stage model producing them.)

Strong quantitative reconstruction quality on the NSD benchmark. The eight metrics are PixCorr and SSIM (pixel and structural correlation) plus 2-way identification accuracy across six pretrained vision/language feature spaces (AlexNet(2), AlexNet(5), Inception, CLIP, EffNet-B, SwAV), on the shared NSD test set (Table 1). Note: the original paper frames this comparatively, as outperforming three prior reconstruction methods (Lin et al., Takagi et al., Gu et al.; Figs. 5-6); we cannot access those models' outputs for LAION-fMRI, so we replicate Brain-Diffuser's own absolute metrics instead (see 8.).

Component contributions established by ablation. Each component contributes distinctly and the full model is the best joint compromise; removing CLIP-Text or CLIP-Vision degrades performance relative to it. (Table 2; Fig. 7; pattern quantified in 7.)

Functional relationship between brain regions and model components. An ROI analysis of the regression weights shows that the early visual regions are more informative about the VDVAE features, while the category-selective higher regions carry more information about the CLIP features (ROIs listed in 7.); the CLIP-Vision versus VDVAE difference runs in the same direction as the CLIP-Text versus VDVAE difference, but is much weaker. (Fig. 8.)

ROI-optimal stimuli consistent with known functional selectivity. Applied to synthetic fMRI patterns activating specific ROIs (listed in 4.), the trained model produces scene content matching each ROI's known selectivity (Figs. 9-11: individual ROIs, ROI combinations, eccentricity bands).

## 7. How were each of these findings generated (method and outcome measure)

**Finding 1:**
Method: a two-stage model. Stage 1 uses ridge regression from fMRI activity to the latent features of a hierarchical VAE (VDVAE), decoded into an initial low-level reconstruction. Stage 2 uses ridge regression from fMRI activity to CLIP-Vision and CLIP-Text features, which condition a pretrained Versatile Diffusion model (VD) that refines the stage-1 reconstruction into the final image.

Outcome measure: qualitative visual comparison of reconstructions to ground-truth test images (Fig. 3). No finding in the original study was tested statistically (see 9.).

Result: the authors report reconstructions "preserve most of the layout and semantic information" though not pixel-perfect, with failure cases documented separately (Fig. 4).

**Finding 2:**
Method: the same regression+reconstruction model as Finding 1, comparing each test-set reconstruction to its ground-truth image.

Outcome measure: the eight point-estimate metrics listed in 6.

Result (Table 1, in the appendix): the best (or tied-best) value among all compared models on every metric.

**Finding 3:**
Method: four ablated variants of the full model (Only-VDVAE, w/o VDVAE, w/o CLIP-Text, w/o CLIP-Vision), evaluated on Sub1's test set as in Finding 2.

Outcome measure: the same eight metrics (Table 2), plus qualitative inspection of reconstructions (Fig. 7).

Result (Table 2, Fig. 7, in the appendix): Only-VDVAE is best on the low-level measures and worst on the high-level ones; w/o VDVAE is worst on low-level while among the best on high-level; the full model is the optimal compromise across both. Qualitatively, Only-VDVAE produces vague silhouettes and w/o VDVAE loses object layout.

**Finding 4:**
Method: for each voxel in 8 ROIs (V1-V4 from population receptive field mapping; Face, Word, Place and Body ROIs from functional localizers), the strength (L1 norm) of the ridge regression weights for the CLIP and the VDVAE features was expressed as a percentile and reported as the CLIP-minus-VDVAE difference, controlling for differences in ROI size, overall activity and noise level.

Outcome measure: that difference, averaged over voxels per ROI, with error bars showing the standard error of the mean across the 4 subjects (Fig. 8).

Result: descriptive, as stated in 6.

**Finding 5:**
Method: synthetic activation patterns for each ROI are constructed (without real fMRI data) and passed through the trained regression and reconstruction model.

Outcome measure: qualitative inspection of the resulting images, as described in 6.

Result: reconstructions are reported as "consistent with neuroscientific knowledge" (e.g. face-like content for the Face-ROI, scene/place content for the Place-ROI).

## 8. How to replicate these results

Finding 1: LAION-fMRI data will be prepared the same way NSD data was (train/test split, excluding out-of-distribution images) and the same two-stage model run on it, with reconstructions compared to ground truth qualitatively, as in the original study. The findings below use these same runs, per subject (LAION-fMRI has 5 subjects).

Finding 2: the same eight metrics on those reconstructions, compared against the values the original study reports for NSD (criterion in 9.).

Finding 3: the same four ablation variants, with the same eight metrics for each. These variants are internal to Brain-Diffuser and require no external models, so this comparison transfers in full.

Finding 4: the regression-weight percentile analysis recomputed from our LAION-fMRI-trained regressions, using the dataset's own retinotopically defined early visual ROIs and its category-selective localizer ROIs, testing whether the same early-versus-higher division of labour holds.

Finding 5: as in the original, using LAION-fMRI's own ROI definitions, again assessed qualitatively. Note: the original code has a disclosed bug limiting precision for most ROIs; we will attempt to fix it before running this finding.

New metadata needed: none. LAION-fMRI already provides everything the model requires: images, captions, ROI definitions, retinotopy and functional localizers.

Results we cannot replicate using LAION-fMRI: the comparison against three other reconstruction models (Finding 2), for the reason given in the note in 6.; we instead compare against the original study's own reported performance.

Potential confound: VD is used frozen and was pretrained on LAION-2B, and LAION-fMRI's natural images were drawn from a curated 120M subset of that same corpus. This is not classical test-set leakage: the raw stimuli never enter the model, as everything VD receives is fMRI-derived (the predicted CLIP-Vision and CLIP-Text embeddings and the VDVAE initial guess), so the overlap cannot by itself make a reconstruction resemble the particular image a subject viewed. It can, however, better match the generative prior to the stimulus distribution, which may inflate absolute values relative to NSD. We know of no openly available diffusion model combining VD's joint CLIP-Vision and CLIP-Text conditioning with non-LAION pretraining, so substituting the generative model is not an option; We aim to assess the effect via two internal controls instead: the Only-VDVAE ablation, whose ImageNet prior is equally unmatched to both datasets, and evaluations based on the LAION-fMRI's THINGS and MS-COCO stimuli, which are not LAION-derived. Furthermore, The original study likely faces a weaker version of the same confound, since MS-COCO images are widely reposted and some are likely present in LAION-2B, though we know of no published estimate of how many.

## 9. Statistics

As noted in 7., none of the five findings was tested statistically: every result is either qualitative or a point estimate reported without variance estimates. Staying as close as possible to the original approach, we introduce no inferential tests.

Findings 2 and 3: the eight metrics are compared descriptively against the values reported on NSD in Tables 1 and 2. The criterion for Finding 2 is whether comparable metric values are attained; for Finding 3, whether the ordinal pattern across the ablation variants (7.) is preserved. Absolute magnitudes are not strictly comparable across datasets, which differ in subjects, image distribution, field strength and trial counts.

Finding 4: replicated exactly as in the original (7.); the comparison is the sign and the rank ordering of the CLIP-minus-VDVAE difference across ROIs.

Findings 1 and 5 are qualitative in the original study and remain so here.

## 10. How to test generalization of these findings (required only for generalization)

We plan to attempt generalization for Finding 2, using re:vision's Method 1 (a broad-coverage train/test split for unseen images), Method 2 (out-of-distribution image clusters) and Method 3 (LAION-fMRI's "shape", "unusual" and "cropped" out-of-distribution categories, all three rather than a single one, since Finding 2 is a general reconstruction-quality metric not tied to any one visual property): we retrain on each split's training portion, evaluate on the held-out portion, and compare the resulting metrics to our main replication.

Finding 1 follows the same procedure, assessed qualitatively (7.). Findings 3 and 4 will not be generalized, to keep the computational cost manageable, as the template permits limiting generalization to the main finding. Finding 5 does not use a held-out image split, so generalisation is not applicable.

## 11. Appendix (Screenshots of Figures and Tables)
