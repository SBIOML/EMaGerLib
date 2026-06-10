# Benchmark results log

Most recent runs are appended below.

## 2026-05-28 22:51:47

- **Datasets:** Test_EM_C7_R5
- **Seeds:** [42]
- **Epochs:** 1  |  **Window:** 200/10  |  **Sampling:** 2000  |  **Batch:** 64/256
- **Config file:** (constants in script)
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 1m 13s

| Model | Test_EM_C7_R5 | Overall | Params | Size (KB) | MACs | Lat (ms) |
|---|---|---|---|---|---|---|
| EmagerCNNBase | 98.5% ± 1.8% | 98.5% | 562,375 | 2,200.1 | 2.77M | 0.62 |
| EmagerCNNQuantized | 98.4% ± 2.0% | 98.4% | 562,375 | 35.7 | 2.77M | 0.88 |

## 2026-05-28 23:09:53

- **Datasets:** Test_EM_C7_R5
- **Seeds:** [42]
- **Epochs:** 10  |  **Window:** 200/10  |  **Sampling:** 2000  |  **Batch:** 64/256
- **Config file:** (constants in script)
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 14m 31s

| Model | Test_EM_C7_R5 | Overall | Params | Size (KB) | MACs | Lat (ms) |
|---|---|---|---|---|---|---|
| EmagerCNNCircular | 98.7% ± 0.8% | 98.7% | 562,375 | 2,200.1 | 2.77M | 0.85 |
| EmagerCNNRingStrided | 98.7% ± 1.0% | 98.7% | 44,359 | 175.1 | 152.5K | 0.59 |
| EmagerCNNStrided | 98.4% ± 1.7% | 98.4% | 44,359 | 175.1 | 152.5K | 0.42 |
| EmagerCNNBase | 98.3% ± 1.4% | 98.3% | 562,375 | 2,200.1 | 2.77M | 0.62 |
| EmagerCNNQuantized | 98.2% ± 1.5% | 98.2% | 562,375 | 35.7 | 2.77M | 0.92 |

## 2026-05-28 23:13:04

- **Datasets:** Test_EM_C7_R5
- **Seeds:** [42]
- **Epochs:** 1  |  **Window:** 200/10  |  **Sampling:** 2000  |  **Batch:** 64/256
- **Config file:** (constants in script)
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 1m 22s

| Model | Test_EM_C7_R5 | Overall | Params | Size (KB) | MACs | Lat (ms) |
|---|---|---|---|---|---|---|
| EmagerCNNQAT | 97.0% ± 4.0% | 97.0% | 562,375 | 35.7 | 2.77M | 0.90 |

## 2026-05-28 23:43:39

- **Datasets:** Test_EM_C7_R5
- **Seeds:** [42]
- **Epochs:** 10  |  **Window:** 200/10  |  **Sampling:** 2000  |  **Batch:** 64/256
- **Config file:** (constants in script)
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 19m 58s

| Model | Test_EM_C7_R5 | Overall | Params | Size (KB) | MACs | Lat (ms) |
|---|---|---|---|---|---|---|
| EmagerCNNCircular | 98.7% ± 0.8% | 98.7% | 562,375 | 2,200.1 | 2.77M | 0.86 |
| EmagerCNNRingStrided | 98.7% ± 1.0% | 98.7% | 44,359 | 175.1 | 152.5K | 0.58 |
| EmagerCNNStrided | 98.4% ± 1.7% | 98.4% | 44,359 | 175.1 | 152.5K | 0.42 |
| EmagerCNNQuantizedQAT | 98.3% ± 2.3% | 98.3% | 562,375 | 35.7 | 2.77M | 0.91 |
| EmagerCNNBase | 98.3% ± 1.4% | 98.3% | 562,375 | 2,200.1 | 2.77M | 0.73 |
| EmagerCNNQuantizedPTQ | 98.2% ± 1.5% | 98.2% | 562,375 | 35.7 | 2.77M | 1.01 |

## 2026-05-28 23:45:58

- **Datasets** (leave-one-out across reps):
    - `Test_EM_C7_R5` — 7 classes, 5 reps · ~41,815 datapoints/rep (~20.9s @ 2000 Hz)
- **Seeds:** [42]
- **Epochs:** 1  |  **Window:** 200/10  |  **Sampling:** 2000  |  **Batch:** 64/256
- **Fits:** 5 total  ·  8.7s avg/fit (incl. data load)
- **Config file:** (constants in script)
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 43s

| Model | Test_EM_C7_R5 | Overall | Params | Size (KB) | MACs | Lat (ms) |
|---|---|---|---|---|---|---|
| EmagerCNNBase | 98.5% ± 1.8% | 98.5% | 562,375 | 2,200.1 | 2.77M | 0.61 |

## 2026-05-28 23:58:42

- **Datasets** (leave-one-out across reps):
    - `Test_EM_C7_R5` — 7 classes, 5 reps · ~5,974 datapoints/rep (~3.0s @ 2000 Hz)
- **Seeds:** [42]
- **Epochs:** 10  |  **Window:** 200/10  |  **Sampling:** 2000  |  **Batch:** 64/256
- **Fits:** 5 total  ·  41.7s avg/fit (incl. data load)
- **Config file:** (constants in script)
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 3m 28s

| Model | Test_EM_C7_R5 | Overall | Params | Size (KB) | MACs | Lat (ms) |
|---|---|---|---|---|---|---|
| EmagerCNNBase | 98.3% ± 1.4% | 98.3% | 562,375 | 2,200.1 | 2.77M | 0.63 |

## 2026-05-29 00:07:17

- **Datasets** (leave-one-out across reps):
    - `Test_EM_C7_R5` — 7 classes, 5 reps · ~5,974 datapoints/rep (~3.0s @ 2000 Hz)
- **Seeds:** [42]
- **Epochs:** 1  |  **Window:** 200/10  |  **Sampling:** 2000  |  **Batch:** 64/256
- **Fits:** 5 total  ·  10.6s avg/fit (incl. data load)
- **Config file:** (constants in script)
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 52s

| Model | Test_EM_C7_R5 | Overall | Params | Size (KB) | MACs | Lat (ms) |
|---|---|---|---|---|---|---|
| EmagerCNNRingStridedQAT | 97.9% ± 1.8% | 97.9% | 44,359 | 35.7 | 152.5K | 1.13 |

## 2026-05-29 00:14:59

- **Datasets** (leave-one-out across reps):
    - `Test_EM_C7_R5` — 7 classes, 5 reps · ~5,974 datapoints/rep (~3.0s @ 2000 Hz)
- **Seeds:** [42]
- **Epochs:** 10  |  **Window:** 200/10  |  **Sampling:** 2000  |  **Batch:** 64/256
- **Fits:** 5 total  ·  53.5s avg/fit (incl. data load)
- **Config file:** (constants in script)
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 4m 27s

| Model | Test_EM_C7_R5 | Overall | Params | Size (KB) | MACs | Lat (ms) |
|---|---|---|---|---|---|---|
| EmagerCNNRingStridedQAT | 98.8% ± 0.8% | 98.8% | 44,359 | 35.7 | 152.5K | 1.15 |

## 2026-05-29 00:25:23

- **Datasets** (leave-one-out across reps):
    - `Test_EM_C7_R5` — 7 classes, 5 reps · ~5,974 datapoints/rep (~3.0s @ 2000 Hz)
- **Seeds:** [42]
- **Epochs:** 10  |  **Window:** 200/10  |  **Sampling:** 2000  |  **Batch:** 64/256
- **Fits:** 5 total  ·  30.2s avg/fit (incl. data load)
- **Config file:** (constants in script)
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 2m 30s

| Model | Test_EM_C7_R5 | Overall | Params | Size (KB) | MACs | Lat (ms) |
|---|---|---|---|---|---|---|
| EmagerCNNStrided | 98.4% ± 1.7% | 98.4% | 44,359 | 185.7 | 152.5K | 0.40 |

## 2026-06-04 13:46:09

- **Datasets** (leave-one-out across reps):
    - `Test_EM_C7_R5` — 7 classes, 5 reps · ~5,974 datapoints/rep (~3.0s @ 2000 Hz)
- **Seeds:** [42]
- **Epochs:** 10  |  **Window:** 200/10  |  **Sampling:** 2000  |  **Batch:** 64/256
- **Fits:** 10 total  ·  62.1s avg/fit (incl. data load)
- **Config file:** (constants in script)
- **Host:** cpu · torch 2.10.0+cpu · py 3.12.3 · Windows-11-10.0.26200-SP0
- **Elapsed:** 10m 21s

| Model | Test_EM_C7_R5 | Overall | Params | Size (KB) | MACs | Lat (ms) |
|---|---|---|---|---|---|---|
| EmagerCNNProtoCE | 99.5% ± 0.4% | 99.5% | 167,239 | 665.3 | 2.38M | 0.67 |
| EmagerCNNProtoEpisodic | 99.4% ± 0.5% | 99.4% | 166,784 | 665.3 | 2.38M | 0.67 |
