# Distributed experiment: gloo / cpu

Workers: 2; seed: 17; commit: 7e6c7b7e03a6e46f1720277c3e0447c4b57fcc75

| Experiment | Correctness | Slowest-rank median ms |
|---|---|---:|
| All-reduce SUM (65536 FP32 elements) | passed | 0.602700 |
| DDP forward/backward/SGD | gradients + parameters + global mean loss passed | 1.507000 |

Maximum gradient error: 2.9802322e-08; parameter error: 1.8626451e-09.

Times are synchronized host wall time; each sample uses the slowest worker. Communication excludes input reset and the preceding barrier. Training includes DDP forward, backward and SGD, excludes serial checks and the preceding barrier, and includes the first step. No speedup or scaling claim is made. CPU Gloo does not validate CUDA NCCL.
