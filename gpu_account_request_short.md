# GPU account request — short form (~170 words)

I am building ActiveST-Bench, a controlled benchmark testing whether active
learning can cut the cost of spatial transcriptomics by selecting which tissue
spots are worth sequencing, with Mert Sabuncu and Ruining Deng. To date I have
trained 412 models across three public cohorts, consuming 824 GPU-hours over 40
days at 87% utilisation on a single RTX 3080. The result is unexpected and worth
extending: informed selection underperforms random sampling on two cancer
cohorts, yet beats it in 47 of 48 comparisons on a kidney cohort, suggesting that
whether active learning helps depends on cohort structure rather than on the
method.

Testing that claim requires varying the predictor backbone, adding cohorts and
imaging platforms, and repeating runs across seeds — roughly 2,000 GPU-hours of
independent single-GPU jobs. Pathology foundation-model encoders carry 30–40×
the parameters of my current backbone and will not fit in 10 GB at a usable
batch size, so this work cannot run on local hardware at all.

All data are public and de-identified (HEST-1k); no PHI is involved. Output: a
SPIE Medical Imaging 2027 paper and a journal extension.
