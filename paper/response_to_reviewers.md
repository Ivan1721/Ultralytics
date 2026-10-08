# Response to Reviewers — ARTIIS 2026

**Paper:** Occlusion-Aware Fruit Instance Segmentation for Agricultural Robotics:
A Comparative Study of YOLO11, YOLO26, and Mask2Former
**Authors:** I. Garcia et al.

> **Cómo usar esta plantilla:** debajo de cada comentario hay un bloque
> `> Comentario del revisor:` — reemplázalo con la cita textual exacta tomada
> de EasyChair (no tengo el texto literal de los dos reviews en este momento,
> solo el resumen de los puntos ya resueltos). La columna "Respuesta del
> autor" ya está redactada y verificada contra el manuscrito actual. Si un
> punto viene claramente de un revisor específico, muévelo a la sección de
> ese revisor; si aplica a ambos, déjalo en "Comentarios generales".

We thank both reviewers for their careful reading and constructive feedback.
Below we address each point raised, with a reference to the specific section,
table, or figure of the revised manuscript where the change was made.

---

## Reviewer 1 (Score: 2 — Accept)

### R1.1 — Visibility/occlusion protocol not defined
> Comentario del revisor: *[pegar cita textual]*

**Respuesta del autor:** We added an explicit description of the manual
visibility-labeling protocol (each image was inspected and assigned to the
25/50/75/100% bucket based on the fraction of fruit surface occluded by
foliage/branches) and a new table quantifying the number of validation
images per visibility level (142/94/133/95 for 25/50/75/100%, respectively).
*See Section 4.4 ("Visibility and Occlusion Analysis") and Table 5
("Visibility-level counts").*

### R1.2 — Training hardware/GPU not specified
> Comentario del revisor: *[pegar cita textual]*

**Respuesta del autor:** The exact training GPU is now stated explicitly:
NVIDIA GeForce RTX 4060 Ti (8GB VRAM). *See Section 4.2 ("Training
Configuration").*

### R1.3 — Train/val split percentage mismatch
> Comentario del revisor: *[pegar cita textual]*

**Respuesta del autor:** Figure 1 incorrectly stated an 80%/20% split; this
was corrected to the actual split used (1224 training / 491 validation
images, ≈71%/29%). *See Figure 1 and Table 1.*

### R1.4 — No independent test set
> Comentario del revisor: *[pegar cita textual]*

**Respuesta del autor:** We now explicitly acknowledge, as a limitation, that
evaluation relies on a validation set rather than a held-out independent test
set, and discuss the implication for the reported numbers. *See Section 5
("Discussion"), limitations paragraph.*

### R1.5 — "Real-time" / embedded-deployment claims too strong
> Comentario del revisor: *[pegar cita textual]*

**Respuesta del autor:** Claims of real-time/embedded suitability were
moderated: we clarify that the reported latency is model-inference time only
(batch size 1, single GPU), not an end-to-end robotic pipeline measurement,
and that the YOLO-based and Mask2Former latencies come from each framework's
own benchmarking utility and are not necessarily timing the same
pre/post-processing steps. *See Section 4.2 and the Table 4 discussion.*

### R1.6 — "Best trade-off" claim for YOLO11m lacked a defined criterion
> Comentario del revisor: *[pegar cita textual]*

**Respuesta del autor:** We added an explicit, quantified trade-off criterion:
YOLO11m retains 85.2% of Mask2Former's accuracy while using only 50.8% of its
parameters and 55.1% of its FLOPs. *See Section 5 ("Discussion").*

---

## Reviewer 2 (Score: 1 — Weak Accept)

### R2.1 — General English/spelling pass needed
> Comentario del revisor: *[pegar cita textual]*

**Respuesta del autor:** The manuscript underwent an additional copy-editing
pass (grammar, leaked variable names in running text, math-notation
consistency, dash/hyphen typos). *Affects running text throughout; see e.g.
the corrected "red apple" wording (previously a leaked code identifier) and
the F-score formula notation.*

### R2.2 — Prefer published versions over preprints where available
> Comentario del revisor: *[pegar cita textual]*

**Respuesta del autor:** We checked each preprint-cited reference for a
published version and updated the bibliography accordingly, e.g. the
Mask2Former citation now points to the published CVPR 2022 version
(Cheng, Misra, Schwing, Kirillov, Girdhar) rather than the arXiv preprint,
and the YOLO26 architecture/evolution citations were corrected to their
actual (2025) publication year. *See `Ref.bib` and the reference list.*

---

## Comentarios generales (sin atribuir a un revisor específico)

*(Mueve cada ítem a la sección del revisor correspondiente si identificas
cuál de los dos lo planteó.)*

### Figure 3 color collision / legend inconsistency
> Comentario del revisor: *[pegar cita textual, si aplica]*

**Respuesta del autor:** Figure 3 originally plotted more than 10 series with
Matplotlib's default color cycle, causing color collisions between unrelated
models/visibility levels. It was regenerated with explicit, fixed colors per
model (Mask2Former = purple, YOLO11m = blue, YOLO26x = red) and distinct line
styles per visibility level (solid/dashed/dash-dot/dotted for
100/75/50/25%). The figure's own caption, which still claimed the old,
disproven "YOLO11m consistently outperforms YOLO26x" result, was also
corrected to match the body text. *See Figure 3 and Section 4.4.*

### "Real orchard" claim not matching the actual acquisition setup
> Comentario del revisor: *[pegar cita textual, si aplica]*

**Respuesta del autor:** All references to a "real orchard" environment were
corrected to accurately describe the actual setup: a controlled environment
using real fruit specimens arranged to emulate orchard-like occlusion
conditions. *See Section 3.1 ("Data Acquisition Setup") and all subsequent
mentions.*

### YOLO11 vs. YOLO26 overclaim ("YOLO11 consistently superior")
> Comentario del revisor: *[pegar cita textual, si aplica]*

**Respuesta del autor:** The claim was corrected to a size-dependent
framing: YOLO11 outperforms YOLO26 at the n/s/m sizes, while YOLO26x
outperforms both YOLO11l and YOLO11x — the comparison between the two
families is size-dependent, not uniform. *See Section 4.1.*

### YOLO26l anomalous result insufficiently explained
> Comentario del revisor: *[pegar cita textual, si aplica]*

**Respuesta del autor:** We traced YOLO26l's anomalously low score against
its own training log: validation mask mAP50–95 peaked at 50.0% by epoch 15
and then declined steadily to 41.23% by epoch 100 (the final-epoch checkpoint
used for evaluation, consistent with all other models). The manuscript now
describes this correctly as late-stage overfitting specific to this training
run, not a failure to converge or an architectural effect of model size.
*See Section 4.1.*

### Missing or off-domain citations (Mask2Former, Ultralytics, MMDetection, COCO)
> Comentario del revisor: *[pegar cita textual, si aplica]*

**Respuesta del autor:** Added missing citations for the Ultralytics YOLO
framework, MMDetection, and the COCO dataset/format, alongside the existing
Mask2Former reference. *See `Ref.bib` and the Methodology section.*

### ORCID mismatch
> Comentario del revisor: *[pegar cita textual, si aplica]*

**Respuesta del autor:** Confirmed and kept the author's existing ORCID
(0009-0005-4235-9913), which was already correct in the manuscript.

---

## Summary of changes (quick reference)

| # | Issue | Location in revised manuscript |
|---|---|---|
| 1 | Visibility protocol undefined | Sec. 4.4, Table 5 |
| 2 | GPU not specified | Sec. 4.2 |
| 3 | Split percentage wrong in Fig. 1 | Fig. 1, Table 1 |
| 4 | No independent test set | Sec. 5 (limitations) |
| 5 | "Real-time" claim too strong | Sec. 4.2, Table 4 discussion |
| 6 | "Best trade-off" undefined | Sec. 5 |
| 7 | English/spelling pass | Throughout |
| 8 | Preprints vs. published versions | `Ref.bib` |
| 9 | Figure 3 color collision + stale caption | Fig. 3 |
| 10 | "Real orchard" miscast as field conditions | Sec. 3.1, throughout |
| 11 | YOLO11 vs YOLO26 overclaim | Sec. 4.1 |
| 12 | YOLO26l anomaly unexplained | Sec. 4.1 |
| 13 | Missing citations | `Ref.bib`, Methodology |
| 14 | ORCID | Author metadata (confirmed correct) |

We believe these revisions fully address the reviewers' concerns while
keeping the paper's core contributions and conclusions unchanged. We thank
the reviewers again for helping us improve the clarity and rigor of the
manuscript.
