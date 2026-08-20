# Skin Lesion Classification: A Fairness Audit Across Skin Tones

## Problem Statement

Dermatology imaging datasets used to train AI diagnostic tools often underrepresent darker skin tones. Models trained on such datasets have been documented to show lower performance on darker skin tones — a persistent equity challenge in medical AI with potential clinical consequences. Melanoma in patients with darker skin is also associated with later-stage diagnosis and poorer outcomes, raising concerns that an AI screening tool could reproduce or amplify existing disparities rather than safely reducing diagnostic burden.

This project builds a 3-class skin lesion classifier (non-neoplastic / benign / malignant) and asks a harder question than “what's the accuracy?”: does this specific model show a measurable performance disparity across Fitzpatrick skin-tone bands, how robust is that disparity statistically, can it be reduced with standard mitigation techniques, and if not, what factors might explain it?

The goal was not to produce a clean “I fixed bias” result, but a rigorous, honestly reported audit — including negative results, reproducibility failures, and an investigation into why mitigation attempts did or did not work, rather than treating the model as a black box.
---

## Part 1: Sourcing the Dataset

### Attempt 1 — Fitzpatrick17k via GitHub CSV + original image URLs
The canonical way to get Fitzpatrick17k is the [mattgroh/fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k)
repo: a CSV of 16,577 rows with Fitzpatrick scale labels (I–VI) and diagnosis metadata, with images
hosted at their original source URLs (dermaamin.com and atlasdermatologico.com.br — two independent
dermatology atlas sites, not a stable CDN).

This failed at the environment level before it even got to the dataset: the working Colab/Kaggle
runtime returned `Failed to resolve` DNS errors for every domain tested, including `google.com` —
a session-level networking issue, not a dead-link problem. Diagnosed via a standalone DNS resolution
test script comparing a known-good domain against the target domains.

### Attempt 2 — Kaggle mirror (`nazmusresan/fitzpatrick17k`) — background-removed images
Switched to a pre-downloaded Kaggle mirror to sidestep the link-rot problem entirely. This worked
structurally (16,574/16,577 images present, 0 corrupted, filenames matched CSV `md5hash` values
after correcting an initial sampling bug — see below), but a visual sanity-check grid (5 sample
images per Fitzpatrick band) revealed a serious problem: the images had gone through a "background
removal" preprocessing step that produced blocky, inconsistent segmentation artifacts — in several
cases eating into lesion-relevant skin area, not just background.

Quantified via near-black pixel fraction per image, sampled across bands:
- Dark skin: 40.4% near-black
- Light skin: 41.8% near-black
- Medium skin: 26.9% near-black

The inconsistency across bands (medium notably lower than light/dark) meant this preprocessing
artifact could confound any fairness result computed downstream — if performance later differed
by skin tone, it would be impossible to know whether that reflected the model or the corrupted
input data. This dataset was discarded for training use.

**Filename-matching false alarm:** an early verification script compared only the first 1,000
filenames on disk against the first 1,000 CSV rows and found just 56 matches, which looked like
a serious labeling problem. This turned out to be a sampling bug — file listing order (alphabetical)
and CSV row order are unrelated, so comparing two arbitrary un-aligned slices produces a misleadingly
low overlap. Re-run against the *full* sets: 16,574/16,574 matched, 0 unmatched. Lesson: always
verify joins against complete sets, not partial slices, before concluding data is broken.

### Attempt 3 — SFU corrected/raw release
Found via a research group ([Kumar Abhishek et al., SFU](https://www.sfu.ca/~kabhishe/SkinDatasetsCritique/))
that had specifically studied and documented quality issues in Fitzpatrick17k, including duplicate
detection and erroneous-image flagging. Their release provides raw, non-background-removed images,
organized into 114 diagnosis-abbreviation folders, with filenames encoding diagnosis abbreviation,
Fitzpatrick scale, and a truncated 8-character MD5 hash (e.g. `po-mi_f2_20_1699ebca.jpg`).

Download was hosted on SharePoint, which does not serve direct file downloads via `wget`/`curl` —
the URL redirects to an interactive `onedrive.aspx` viewer, returning an HTML page instead of the
zip (caught immediately by checking `Content-Type` and file size — 369KB of HTML, not a multi-GB
archive). Resolved via manual browser download, then re-upload to Kaggle as a private dataset.

Verification: all 16,577 filenames parsed cleanly via regex; the truncated 8-char hash suffix
matched the first 8 characters of the original CSV's full `md5hash` column with 0 collisions and
100% match rate, allowing full diagnosis/three-partition-label recovery via join. Sanity-check grid
on these raw images confirmed near-black pixel fraction dropped to ~2% and became consistent across
all three Fitzpatrick bands (0.021/0.022/0.023) — the preprocessing confound was resolved.

### Attempt 4 — DDI (Diverse Dermatology Images), for supplementary data
Sourced later (Part 3 below) via the Stanford AIMI portal, chosen specifically because it is
dermatologist-curated and explicitly balanced across Fitzpatrick I–VI — unlike Fitzpatrick17k,
which is a scraped, imbalanced convenience sample.

- First attempted via Redivis (Stanford's institutional data review platform) — abandoned due to
  slow approval queue, incompatible with project timeline.
- Second attempt via the direct Stanford AIMI Azure portal (`stanfordaimi.azurewebsites.net`) —
  self-service registration, faster.
- **License constraint:** DDI's research use agreement explicitly prohibits redistribution and
  link-sharing. Unlike Fitzpatrick17k, this dataset was kept in a strictly private, non-shared
  Kaggle dataset with no collaborators, and is not included in this repository — only the
  processing code is.
- 656 images, one row per image, metadata columns: `DDI_ID`, `DDI_file`, `skin_tone` (coded as
  paired FST groups: 12=light, 34=medium, 56=dark), `malignant` (bool), `disease` (fine-grained
  diagnosis, including site-specific subtype names like `melanoma-acral-lentiginous`).

---

## Part 2: Data Preparation

- Final usable Fitzpatrick17k set: 16,012 rows (16,577 minus 3 unmatched images minus 565 rows
  with unlabeled Fitzpatrick scale).
- Fitzpatrick scale (1–6) collapsed into three bands for stratification: light (1–2), medium (3–4),
  dark (5–6) — raw distribution: light 7,755 / medium 6,089 / dark 2,168 (~48% / 38% / 13.5%).
- Stratified 70/15/15 train/val/test split, stratified jointly on a combined key
  (`three_partition_label` × `fst_band`) so every split preserves the same diagnosis-type and
  skin-tone proportions — verified post-split: all three splits matched to within 0.1 percentage
  points on both dimensions.
- Smallest stratification cells (`benign_dark`: 203 total, `malignant_dark`: 208 total) flagged
  early as a likely source of noisy subgroup metrics later, given they reduce to roughly 30 test
  cases each after the 70/15/15 split.

---

## Part 3: Model Training — Four Stages

### Stage 1: Baseline
- **Architecture:** EfficientNet-B0, ImageNet-pretrained, early feature blocks (0–5) frozen,
  fine-tuned blocks 6–8 + a new classifier head (Dropout 0.4 → Linear(3)).
- **Loss:** class-weighted cross-entropy (weights inverse to class frequency: non-neoplastic
  0.46, benign 2.47, malignant 2.47) + label smoothing (0.1).
- **Optimizer:** Adam, lr=1e-4, weight decay 1e-4, `ReduceLROnPlateau` scheduler, early stopping
  (patience=4 on validation loss).
- **Overfitting iteration:** an initial unregularized run (full backbone fine-tuning, no dropout
  tuning, no weight decay) showed val loss bottoming out at epoch 3 while train accuracy climbed
  past 93% — classic overfitting. Fixed incrementally: (1) freezing early backbone layers cut
  trainable parameters from ~4M to ~3.16M and slowed the train/val divergence; (2) raising dropout
  to 0.4 and adding weight decay reduced it further; (3) label smoothing specifically addressed a
  secondary pattern where val loss kept rising even as val accuracy still improved — a sign of
  overconfident wrong predictions, which label smoothing directly penalizes.
- **Result (seeded, reproducible run):** 82.4% overall test accuracy, malignant recall 72%,
  malignant precision 69%.

### Fairness audit methodology (applied at every stage)
- Malignant recall stratified by Fitzpatrick band, with bootstrap 95% confidence intervals
  (1,000 resamples) to account for small subgroup sample sizes (dark-skin malignant test set:
  n=31).
- Permutation significance testing (5,000 permutations) on pairwise recall gaps, restricted to
  true-malignant cases only (not overall accuracy, which is confounded by majority-class
  performance — an early version of this analysis mistakenly used overall accuracy and found a
  "significant" gap that was actually being driven by non-neoplastic recall differences, not the
  clinically important malignant recall; corrected to isolate the malignant-only comparison).
- Baseline finding: malignant recall — light 70.0%, medium 77.2%, dark 61.3% (seeded run).
  Permutation tests on earlier unseeded runs returned p=0.08–0.50 (not significant at p<0.05),
  reflecting the statistical power limits of a 31-case subgroup, not necessarily the absence of
  a real effect.

### Stage 2: Reweighting (class-balanced sampler)
- **Approach:** `WeightedRandomSampler`, per-sample weight = 1 / (size of its diagnosis × skin-tone
  group), upsampling `malignant_dark` (146 unique train images) and `benign_dark` (142 unique train
  images) far more heavily per epoch than their raw frequency would allow, while keeping the same
  class-weighted loss as Stage 1.
- **Result:** malignant recall — light 75.6%, medium 76.3%, dark 64.5%. A modest +3.2pp
  improvement for dark skin over baseline, but light/medium also moved, keeping the gap roughly
  similar (11.8pp medium-dark gap vs. 15.9pp at baseline — narrower, but not by a large margin).
- **Root cause of limited impact:** the sampler does not create new data — it re-shows the same
  ~146–142 unique images more often per epoch (with different random crops/flips/color jitter each
  time). With a very small unique pool, aggressive oversampling risks the model memorizing specific
  images rather than learning generalizable features, capping how much this technique alone can
  help.

### Stage 3: DDI supplementary data fine-tuning
- **Approach:** continued training from the Stage 2 (reweighted) checkpoint, at a lower learning
  rate (5e-5, vs. 1e-4 for fresh training), using a combined train set (Fitzpatrick17k train split
  + 70% of DDI, stratified by the same diagnosis × skin-tone key). The remaining 30% of DDI was
  held out as a fully independent fairness test set never touched during training.
- **DDI integration required schema mapping:** DDI's `skin_tone` column (paired FST codes 12/34/56)
  mapped to the same light/medium/dark bands used for Fitzpatrick17k; `malignant` boolean mapped to
  `three_partition_label` (DDI has no non-neoplastic class, so DDI rows contribute only to the
  benign/malignant categories in training).
- **Growth achieved:** `malignant_dark` unique training images grew from 146 → 180 (+23%);
  `benign_dark` grew from 142 → 253 (+78%) — genuinely new, independently-sourced images, not
  resampled duplicates.
- **Result on Fitzpatrick test set:** malignant recall — light 76.1%, medium 78.1%, dark 64.5%
  (identical to Stage 2's dark recall; light/medium continued climbing). Medium-dark gap widened
  slightly to 13.6pp.
- **Result on independent DDI held-out test set:** light 40.0%, medium 54.5%, dark 42.9% — a
  markedly different pattern from the Fitzpatrick test set (recall much lower across the board,
  consistent with DDI's published difficulty as a harder, out-of-distribution benchmark for
  dermatology AI models generally).

### Stage 4: Targeted Mixup augmentation
**Motivating literature:** after Stage 2's reweighting showed limited effect, the question became
whether *synthetic* data diversity (rather than simple resampling) could help. A literature search
surfaced [Ansari, Chakraborti & Das, "Algorithmic Fairness in Lesion Classification by Mitigating
Class Imbalance and Skin Tone Bias," MICCAI 2024](https://arxiv.org/abs/2408.09442), which
introduces a mixup-based augmentation guided by an adaptive sampler specifically to synthesize
minority-class/minority-skin-tone training samples, and reports this approach mitigates skin tone
bias on benchmark datasets. A related paper, EdgeMixup (Zhao et al.), independently supports mixup
as a validated bias-mitigation technique for skin disease classification, in contrast to naive
color-space augmentation approaches (e.g., synthesizing 180 HSV color variants per image), which a
separate paper noted lacked clear justification for the chosen color transformations. This guided
the choice toward Mixup rather than naive skin-tone color-shifting.

- **Approach implemented:** same-class Mixup restricted to the two weakest groups (`malignant_dark`,
  `benign_dark`). For a sampled image from one of these groups, with 60% probability it is
  pixel-blended with a second, different image from the *same* class and *same* band
  (`lambda ~ Beta(0.4, 0.4)`, constrained so the anchor image stays dominant, `lambda >= 0.5`).
  Same-class blending avoids soft-label interpolation complexity, since the blended image's label
  is unambiguous. Continued from the Stage 3 (DDI fine-tuned) checkpoint at an even lower learning
  rate (3e-5).
- **Result on Fitzpatrick test set:** malignant recall — light 76.7%, medium 78.9%, dark 61.3% —
  back down to baseline level. Medium-dark gap widened further to 17.7pp, the worst of all four
  stages on this test set.
- **Result on independent DDI held-out test set:** dark recall rose to 57.1% (vs. 42.9% for Stage
  3), a genuine +14.3pp improvement, while light/medium stayed flat (40.0%/54.5%, unchanged) — the
  only stage where a mitigation clearly and specifically helped dark skin without also lifting the
  majority groups equally. This divergence between the two test sets is itself a finding: fairness
  outcomes here were sensitive to which data distribution was used for evaluation, not just which
  training strategy was used.



## Part 4: Grad-CAM Analysis

- **Implementation:** custom Grad-CAM hooking the final convolutional block (`features[8]`) of
  EfficientNet-B0, generating class-activation heatmaps for the malignant class score.
- **Quantitative — spatial entropy of correct predictions:** light 9.69, medium 9.71, dark 9.86
  (differences within one standard deviation — no clear evidence attention *quality* differs by
  skin tone when the model is correct).
- **Quantitative — correct vs. missed predictions, collapsed across bands:** correct predictions
  averaged higher entropy (9.75) than missed predictions (9.45). This is the opposite of the
  initial hypothesis (that missed cases would show more diffuse, "confused" attention) — instead,
  missed cases tend to show *tighter*, more confidently-localized attention on the wrong feature,
  suggesting the model's failure mode is closer to confidently attending to a misleading visual
  cue than genuine uncertainty.
- **Qualitative:** visual inspection of a stratified sample grid suggested missed dark-skin
  malignant cases were disproportionately on mucosal/oral sites (lip, tongue) compared to missed
  light/medium-skin cases, which stayed on standard cutaneous sites. An attempt to confirm this
  quantitatively via keyword-matching against the diagnosis label text returned zero matches across
  the entire dataset — Fitzpatrick17k's `label` field encodes diagnosis name only, not anatomical
  site, so this could not be statistically tested with available metadata. Reported only as a
  qualitative, unconfirmed observation.

---

## Part 5: Mechanistic Finding — Subtype Composition

Since the mucosal-site hypothesis couldn't be tested quantitatively, the diagnosis label field was
instead used to check for subtype-level composition differences by skin tone among malignant cases:

- **Kaposi sarcoma** comprised 19.4% of dark-skin malignant test cases (6/31) vs. 3.9% of
  light-skin cases (7/180) — a ~5x proportional difference, consistent with documented
  epidemiological associations between Kaposi sarcoma (particularly endemic forms) and darker-
  skinned populations.
- Classic UV-associated melanoma subtypes — superficial spreading melanoma (20 cases in light skin)
  and lentigo maligna (6 cases in light skin) — were **entirely absent** from the dark-skin
  malignant set, which contained only generically-labeled "melanoma" and "malignant melanoma."
- Kaposi-specific recall: light+medium 69.6% vs. dark 33.3% — a 36.2 percentage-point gap, by far
  the largest of any subtype checked.
- Excluding Kaposi cases from the dark-skin group entirely: recall rose from 61.3% → 68.0%,
  closing roughly 40% of the observed overall dark-skin malignant recall gap from a single subtype
  representing under 20% of that group's cases.
- **Testing the gap against subtype rarity, to distinguish "skin-tone shortcut" from "subtype
  scarcity":** if the model had learned a general skin-tone-based shortcut for malignancy, the
  recall gap should appear roughly uniformly across all malignant subtypes. Instead, the gap
  scaled with how rare/skin-tone-correlated each subtype was: squamous cell carcinoma (common,
  n=74+8), gap +4.7pp; melanoma (moderate, n=28+9), gap +11.9pp; Kaposi sarcoma (rare, n=23+6),
  gap +36.2pp.
- **Additional evidence against a shortcut explanation:** a skin-tone-based shortcut predicting
  "dark skin → malignant/Kaposi-like" would be expected to make Kaposi *easier* to catch in dark
  skin (matching the learned prior) and harder in light skin (contradicting it). The opposite
  pattern was observed — Kaposi recall was *higher* in light skin (71.4%) than dark skin (33.3%)
  — which is more consistent with the model having too few training examples of Kaposi's specific
  visual presentation to learn it reliably at all, rather than having learned a skin-tone-based
  heuristic. This is not a fully conclusive test (it does not rule out shortcut learning via other,
  more subtle visual cues, and would require dedicated counterfactual/causal probing to confirm
  definitively), but it is the available evidence and is reported as such.
- This finding also plausibly explains why none of the three mitigation attempts closed the gap:
  reweighting, DDI supplementation, and Mixup all operated at the skin-tone/class level, not the
  diagnosis-subtype level — none specifically increased Kaposi sarcoma representation, which this
  analysis suggests is where the gap is actually concentrated.

---

## Limitations

- Dark-skin malignant test set is small (n=31 on the primary test set, n=14 on the DDI held-out
  set), limiting statistical power for subgroup and per-subtype claims; several results (e.g. the
  Stage 4 DDI-holdout improvement) should be treated as suggestive rather than conclusive.
- Shortcut-learning cannot be fully ruled out without dedicated counterfactual probing (e.g.
  evaluating the model on synthetic cases where lesion morphology is held constant while skin tone
  varies); the available evidence (recall direction on Kaposi cases across bands) argues against it
  being the dominant explanation, but this is not a conclusive test.
- Model selection (early stopping, LR scheduling) at every training stage used aggregate validation
  loss, not worst-group loss — likely a contributing factor to why light/medium recall improved
  more consistently than dark-skin recall across mitigation attempts. Worst-group-aware selection
  (e.g. Group-DRO) was identified as a promising direction but not implemented, given project scope.
- The mucosal/acral lesion-site hypothesis is qualitative only; Fitzpatrick17k's metadata does not
  support quantitative testing of this observation.
- An earlier, unseeded version of this pipeline produced inconsistent fairness conclusions across
  identical reruns; while the final reported results are from a fully seeded, independently
  verified reproducible pipeline, this history is a reminder that fairness metrics computed on
  small subgroups are sensitive not just to mitigation strategy but to training randomness itself
  — a factor easy to overlook without deliberate reproducibility controls.




## Citations

- Ansari, Chakraborti & Das. "Algorithmic Fairness in Lesion Classification by Mitigating Class
  Imbalance and Skin Tone Bias." MICCAI 2024. — motivated the Stage 4 Mixup approach.
- Daneshjou et al. "Disparities in Dermatology AI Performance on a Diverse, Curated Clinical
  Image Set." Science Advances, 2022. — source of the DDI dataset and supporting evidence that
  robust training methods alone cannot correct dataset bias without diverse training data.
- Groh et al. "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the
  Fitzpatrick 17k Dataset." CVPR Workshops, 2021. — source of Fitzpatrick17k.
- Abhishek et al. SFU Skin Datasets Critique — source of the corrected/raw Fitzpatrick17k release
  and duplicate/erroneous-image documentation used in this project.

---

## Conclusion

This project set out to build a skin lesion classifier and rigorously audit it for performance
disparities across skin tone — not to assume bias exists and confirm it, but to measure it
honestly and understand its mechanism. The baseline model showed a real, clinically-relevant gap
in malignant recall for dark-skin patients, though at the sample sizes available, this could not
be confirmed as statistically significant by conventional thresholds. Three standard mitigation
techniques — class reweighting, supplementary real-world data, and literature-backed targeted
Mixup augmentation — were each implemented rigorously and evaluated honestly, and none reliably
closed the gap on the primary test set, a negative result that is itself reported rather than
hidden. A deeper investigation into diagnosis subtype composition revealed a more precise
mechanism than "skin tone" alone: the gap is concentrated in Kaposi sarcoma, a rare and
epidemiologically skin-tone-correlated cancer subtype for which the model had very little training
exposure — and the gap's near-absence in common subtypes with adequate representation supports
subtype-level data scarcity, not a general skin-tone-based shortcut, as the primary driver. This
reframes the practical takeaway: closing this gap likely requires targeted collection of
underrepresented disease subtypes, not just broader skin-tone-balanced sampling — a more specific
and actionable conclusion than the project's starting hypothesis, and one this audit process was
able to surface through careful, honestly-reported experimentation rather than assume from the
outset.
