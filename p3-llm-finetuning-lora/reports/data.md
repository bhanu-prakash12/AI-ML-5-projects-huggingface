# Data report (small profile)

- Source: keivalya/MedQuad-MedicalQnADataset @ 5b0961f (MedQuAD; license note in DECISIONS.md)
- Raw rows 16407, after clean+dedupe 13637
- Filter: full chat text <= 384 tokens (Qwen tokenizer), seed 42

| split | rows | tokens min | median | mean | max |
|---|---|---|---|---|---|
| train | 300 | 80 | 200 | 211 | 380 |
| val | 50 | 96 | 186 | 200 | 366 |
| test | 50 | 80 | 208 | 208 | 354 |

## Samples (train)

**Q (treatment):** What are the treatments for laryngo-onycho-cutaneous syndrome ?

**A:** These resources address the diagnosis or management of laryngo-onycho-cutaneous syndrome: - Genetic Testing Registry: Laryngoonychocutaneous syndrome These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy ...

**Q (genetic changes):** What are the genetic changes related to iron-refractory iron deficiency anemia ?

**A:** Mutations in the TMPRSS6 gene cause iron-refractory iron deficiency anemia. This gene provides instructions for making a protein called matriptase-2, which helps regulate iron levels in the body. TMPRSS6 gene mutations reduce or eliminate functional matriptase-2, which disrupts iron regulation and l...

**Q (prevention):** How to prevent Heart Failure ?

**A:** Ways to prevent heart failure include - keeping your cholesterol and blood pressure levels healthy - keeping diabetes in check - maintaining a healthy weight - quitting smoking - following a heart healthy diet - limiting the amount of alcohol you drink - eating a diet low in salt because salt can ca...

## Rendered chat template (first train row)

```
<|im_start|>system
You are a careful medical information assistant. Give general, educational answers based on well-established knowledge. You are not a doctor and do not give diagnoses or personal treatment advice.<|im_end|>
<|im_start|>user
What are the treatments for laryngo-onycho-cutaneous syndrome ?<|im_end|>
<|im_start|>assistant
These resources address the diagnosis or management of laryngo-onycho-cutaneous syndrome: - Genetic Testing Registry: Laryngoonychocutaneous syndrome These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliativ
```
