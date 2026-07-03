# Few-shot leave-one-session-out log

Most recent runs are appended below.

## 2026-06-10 17:55:46  — Few-shot LOSO

- **Sessions (leave-one-out):** Test_EM_C7_R5, Test_EM_C7_R5_02, Test_EM_C7_R5_03
- **Seeds:** [42, 123, 456]  |  **k (shots/class):** [1, 2, 5, 10, 20]
- **Support split:** random windows  |  **Fine-tune:** 30 epochs @ lr 0.001 (full model)
- **Offline:** 10 epochs  |  Window 200/10  |  Sampling 2000  |  Batch 64/256
- **Folds:** 3 sessions × 3 seeds = 9 evals/method
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 53m 16s

**Floors (no calibration, k-independent)**

| Method | Acc |
|---|---|
| CNN (zero-shot) | 21.3% ± 8.2% |
| Proto-Episodic (generic) | 21.1% ± 6.5% |
| Proto-CE (generic) | 21.8% ± 7.2% |

**Calibrated (query accuracy vs k shots/class)**

| Method | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| CNN + fine-tune | 77.1% ± 6.5% | 83.5% ± 6.3% | 92.4% ± 1.4% | 96.0% ± 1.4% | 97.5% ± 1.1% |
| Proto-Episodic (k-shot) | 61.0% ± 7.1% | 69.3% ± 4.8% | 73.8% ± 6.0% | 77.0% ± 6.3% | 78.1% ± 5.9% |
| Proto-CE (k-shot) | 68.2% ± 7.7% | 74.6% ± 5.6% | 81.7% ± 5.9% | 84.1% ± 4.9% | 85.5% ± 3.2% |

**Calibration lift (Δ accuracy vs each method's own floor, mean)**

| Method (vs floor) | k=1 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| CNN + fine-tune | +55.8% | +62.2% | +71.1% | +74.7% | +76.2% |
| Proto-Episodic (k-shot) | +39.9% | +48.2% | +52.7% | +55.9% | +57.0% |
| Proto-CE (k-shot) | +46.4% | +52.9% | +59.9% | +62.4% | +63.7% |

