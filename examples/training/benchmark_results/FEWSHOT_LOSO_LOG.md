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

## 2026-07-15 17:47:13  — Few-shot LOSO

- **Sessions (leave-one-out):** Felix_5sessions/S_0, Felix_5sessions/S_1, Felix_5sessions/S_2, Felix_5sessions/S_3, Felix_5sessions/S_4
- **Seeds:** [42, 123, 456]  |  **k (calibration reps/class):** [1, 2, 3, 4, 5, 6, 7, 8, 9]  (1 shot = 1 repetition; 1 rep held out as query)
- **Fine-tune:** 30 epochs @ lr 0.001 (full model)
- **Offline:** 10 epochs  |  Window 200/10  |  Sampling 2000  |  Batch 64/256
- **Folds:** 5 sessions × 3 seeds = 15 evals/method
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 6h 47m 22s

**Floors (no calibration, k-independent)**

| Method | Acc |
|---|---|
| CNN (zero-shot) | 98.9% ± 1.0% |
| Proto-Episodic (generic) | 97.4% ± 4.8% |
| Proto-CE (generic) | 97.4% ± 2.6% |

**Calibrated (query accuracy vs k calibration reps/class)**

| Method | k=1 rep | k=2 rep | k=3 rep | k=4 rep | k=5 rep | k=6 rep | k=7 rep | k=8 rep | k=9 rep |
|---|---|---|---|---|---|---|---|---|---|
| CNN + fine-tune | 99.3% ± 0.5% | 99.4% ± 0.3% | 99.4% ± 0.4% | 99.4% ± 0.4% | 99.4% ± 0.4% | 99.5% ± 0.3% | 99.5% ± 0.3% | 99.5% ± 0.3% | 99.4% ± 0.3% |
| Proto-Episodic (k-shot) | 98.6% ± 2.7% | 99.0% ± 1.0% | 99.2% ± 0.7% | 99.1% ± 0.7% | 99.1% ± 0.7% | 98.8% ± 1.7% | 98.8% ± 1.9% | 98.8% ± 1.8% | 98.8% ± 1.7% |
| Proto-CE (k-shot) | 97.7% ± 2.9% | 98.0% ± 1.3% | 98.3% ± 0.9% | 98.0% ± 1.4% | 98.0% ± 1.4% | 97.9% ± 2.1% | 97.8% ± 2.2% | 97.8% ± 2.2% | 97.8% ± 2.2% |

**Calibration lift (Δ accuracy vs each method's own floor, mean)**

| Method (vs floor) | k=1 rep | k=2 rep | k=3 rep | k=4 rep | k=5 rep | k=6 rep | k=7 rep | k=8 rep | k=9 rep |
|---|---|---|---|---|---|---|---|---|---|
| CNN + fine-tune | +0.3% | +0.5% | +0.5% | +0.5% | +0.5% | +0.6% | +0.6% | +0.6% | +0.5% |
| Proto-Episodic (k-shot) | +1.2% | +1.6% | +1.8% | +1.7% | +1.7% | +1.4% | +1.4% | +1.4% | +1.4% |
| Proto-CE (k-shot) | +0.3% | +0.6% | +0.9% | +0.5% | +0.6% | +0.5% | +0.4% | +0.4% | +0.4% |

## 2026-07-17 00:43:14  — Few-shot LOSO

- **Sessions (leave-one-out):** Felix_5sessions/S_0, Felix_5sessions/S_1, Felix_5sessions/S_2, Felix_5sessions/S_3, Felix_5sessions/S_4
- **Models:** EmagerCNNProtoRingStrided, EmagerCNNProtoRingStridedPTQ, EmagerCNNProtoRingStridedPTQFp32Protos, EmagerCNNProtoRingStridedPTQSupportCalib, EmagerCNNProtoRingStridedQAT
- **Seeds:** [42, 123, 456]  |  **k (calibration reps/class):** [1, 2, 3, 4, 5, 6, 7, 8, 9]  (1 shot = 1 repetition; 1 rep held out as query)
- **Fine-tune:** _CNN baseline skipped (--skip-cnn) — see an earlier run of the same sessions for the CNN rows_
- **Offline:** 10 epochs  |  Window 200/10  |  Sampling 2000  |  Batch 64/256
- **Folds:** 5 sessions × 3 seeds = 15 evals/method
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 5h 10m 9s

**Floors (no calibration, k-independent)**

| Method | Acc |
|---|---|
| Proto-RingStrided (generic) | 97.2% ± 5.5% |
| Proto-RingStridedPTQ (generic) | 97.4% ± 5.0% |
| Proto-RingStridedPTQFp32Protos (generic) | 97.3% ± 4.9% |
| Proto-RingStridedPTQSupportCalib (generic) | n/a |
| Proto-RingStridedQAT (generic) | 97.3% ± 5.2% |

**Calibrated (query accuracy vs k calibration reps/class)**

| Method | k=1 rep | k=2 rep | k=3 rep | k=4 rep | k=5 rep | k=6 rep | k=7 rep | k=8 rep | k=9 rep |
|---|---|---|---|---|---|---|---|---|---|
| Proto-RingStrided (k-shot) | 98.7% ± 1.4% | 99.0% ± 1.0% | 99.0% ± 0.9% | 98.7% ± 1.7% | 98.8% ± 1.4% | 98.7% ± 1.8% | 98.6% ± 2.1% | 98.6% ± 2.0% | 98.6% ± 2.0% |
| Proto-RingStridedPTQ (k-shot) | 98.7% ± 1.6% | 98.8% ± 1.5% | 99.0% ± 0.9% | 98.8% ± 1.4% | 98.8% ± 1.3% | 98.9% ± 1.0% | 98.9% ± 1.0% | 99.0% ± 0.9% | 98.9% ± 1.0% |
| Proto-RingStridedPTQFp32Protos (k-shot) | 98.2% ± 2.3% | 98.7% ± 1.3% | 98.8% ± 1.0% | 98.6% ± 1.6% | 98.6% ± 1.5% | 98.5% ± 1.8% | 98.4% ± 2.1% | 98.5% ± 2.0% | 98.4% ± 2.0% |
| Proto-RingStridedPTQSupportCalib (k-shot) | 98.5% ± 1.8% | 98.5% ± 2.1% | 98.8% ± 1.3% | 98.5% ± 2.1% | 98.5% ± 2.1% | 98.4% ± 2.3% | 98.4% ± 2.5% | 98.5% ± 2.2% | 98.5% ± 2.1% |
| Proto-RingStridedQAT (k-shot) | 99.1% ± 0.6% | 98.5% ± 1.5% | 98.0% ± 2.9% | 97.9% ± 3.2% | 97.8% ± 3.1% | 98.3% ± 2.2% | 98.5% ± 2.0% | 98.5% ± 1.9% | 98.4% ± 2.1% |

**Calibration lift (Δ accuracy vs each method's own floor, mean)**

| Method (vs floor) | k=1 rep | k=2 rep | k=3 rep | k=4 rep | k=5 rep | k=6 rep | k=7 rep | k=8 rep | k=9 rep |
|---|---|---|---|---|---|---|---|---|---|
| Proto-RingStrided (k-shot) | +1.5% | +1.7% | +1.7% | +1.5% | +1.6% | +1.5% | +1.4% | +1.4% | +1.4% |
| Proto-RingStridedPTQ (k-shot) | +1.4% | +1.4% | +1.6% | +1.4% | +1.5% | +1.6% | +1.6% | +1.6% | +1.5% |
| Proto-RingStridedPTQFp32Protos (k-shot) | +0.9% | +1.3% | +1.5% | +1.2% | +1.3% | +1.2% | +1.1% | +1.2% | +1.1% |
| Proto-RingStridedPTQSupportCalib (k-shot) | +nan% | +nan% | +nan% | +nan% | +nan% | +nan% | +nan% | +nan% | +nan% |
| Proto-RingStridedQAT (k-shot) | +1.8% | +1.2% | +0.7% | +0.6% | +0.6% | +1.0% | +1.2% | +1.2% | +1.1% |

## 2026-08-13 04:42:38  — Few-shot LOSO

- **Sessions (leave-one-out):** EM_3Sessions/S_0, EM_3Sessions/S_1, EM_3Sessions/S_2
- **Models:** EmagerCNNProtoEpisodic, EmagerCNNProtoCE, EmagerCNNProtoRingStridedPTQ
- **Seeds:** [42, 123, 456]  |  **k (calibration reps/class):** [1, 2, 3, 4]  (1 shot = 1 repetition; 1 rep held out as query)
- **Fine-tune:** 30 epochs @ lr 0.001 (full model)
- **Offline:** 10 epochs  |  Window 200/10  |  Sampling 2000  |  Batch 64/256
- **Folds:** 3 sessions × 3 seeds = 9 evals/method
- **Host:** cpu · torch 2.12.1+cpu · py 3.12.10 · Windows-11-10.0.26200-SP0
- **Elapsed:** 6h 39m 19s

**Floors (no calibration, k-independent)**

| Method | Acc |
|---|---|
| CNN (zero-shot) | 18.7% ± 11.5% |
| Proto-Episodic (generic) | 22.5% ± 12.8% |
| Proto-CE (generic) | 19.2% ± 7.5% |
| Proto-RingStridedPTQ (generic) | 24.8% ± 5.8% |

**Calibrated (query accuracy vs k calibration reps/class)**

| Method | k=1 rep | k=2 rep | k=3 rep | k=4 rep |
|---|---|---|---|---|
| CNN + fine-tune | 83.2% ± 9.7% | 90.2% ± 8.7% | 90.1% ± 9.2% | 92.7% ± 7.1% |
| Proto-Episodic (k-shot) | 60.7% ± 11.6% | 66.7% ± 14.4% | 68.3% ± 14.3% | 69.2% ± 13.4% |
| Proto-CE (k-shot) | 63.1% ± 11.2% | 67.8% ± 11.5% | 68.9% ± 10.7% | 67.0% ± 10.9% |
| Proto-RingStridedPTQ (k-shot) | 62.1% ± 9.9% | 66.7% ± 12.4% | 68.6% ± 15.4% | 69.3% ± 14.3% |

**Calibration lift (Δ accuracy vs each method's own floor, mean)**

| Method (vs floor) | k=1 rep | k=2 rep | k=3 rep | k=4 rep |
|---|---|---|---|---|
| CNN + fine-tune | +64.5% | +71.5% | +71.4% | +74.0% |
| Proto-Episodic (k-shot) | +38.3% | +44.2% | +45.8% | +46.7% |
| Proto-CE (k-shot) | +43.9% | +48.6% | +49.7% | +47.8% |
| Proto-RingStridedPTQ (k-shot) | +37.3% | +41.9% | +43.7% | +44.5% |

