# Persistent GEMM + reduce-scatter

`example_gemm_rs_push_waves.py` uses eight compute waves and four push waves.
Persistent CTAs hand completed FP16 tiles through reused LDS into push registers,
then spread peer writes across the next tile's GEMM. A second kernel reduces the
local inbox. Accumulation is FP32; two scalar RCCL rendezvous order buffer reuse
and peer writes.

Measured on gfx936, tile 256×256×16, grid 80, no swizzle. Times below are complete
calls in milliseconds, including synchronization and local reduce: 30 samples,
slowest rank's median. Random-input, consecutive-reuse, delayed-sender and Graph
correctness checks passed.

| Ranks | M=N | K | Non-Graph: persistent | Non-Graph: Torch MM + RCCL | Graph: persistent | Graph: Torch MM + RCCL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4096 | 4096 | 0.793600 | 0.728510 | 0.670045 | 0.686385 |
| 4 | 4096 | 4096 | 0.882547 | 0.754996 | 0.765890 | 0.714570 |
| 4 | 8192 | 4096 | 2.791429 | 2.375657 | 2.691509 | 2.331136 |
| 4 | 4096 | 8192 | 1.450541 | 1.342180 | 1.330719 | 1.302215 |
| 4 | 4096 | 16384 | 3.123552 | 2.673244 | 3.001506 | 2.637423 |
| 8 | 4096 | 4096 | 1.047042 | 0.810490 | 1.034577 | 0.761341 |
| 8 | 4096 | 16384 | 3.236398 | 2.712674 | 3.205212 | 2.678449 |
