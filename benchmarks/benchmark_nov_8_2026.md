# Full LAFAN1 throughput timings

One first-use pass and one warm pass per setup. Every pass contains all 77 motions / 496,672 original frames. Apple M1 CPU, 8 cores (4 performance + 4 efficiency), 8 GiB RAM.

| Setup | First-use full (s) | Warm processing (s) | Warm full (s) | Warm FPS | Warm µs/frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| FRDV V1 — 1 worker | 443.309 | 456.757 | 456.879 | 1,087.4 | 919.634 |
| FRDV V2 — 1 worker | 6.792 | 4.725 | 4.812 | 105,126.8 | 9.512 |
| GMR — 1 file worker | 900.338 | 900.744 | 900.873 | 551.4 | 1,813.558 |
| FRAX — 1 CPU device | 79.589 | 72.528 | 72.638 | 6,848.0 | 146.027 |
| FRDV V1 — 8 chunk workers, refinement off | 128.309 | 122.558 | 122.710 | 4,052.5 | 246.759 |
| FRDV V1 — 8 chunk workers, refinement on | 351.237 | 345.989 | 346.111 | 1,435.5 | 696.615 |
| FRDV V2 — 8 file workers | 3.750 | 0.987 | 1.079 | 503,288.6 | 1.987 |
| GMR — 8 file workers | 219.528 | 215.982 | 216.252 | 2,299.6 | 434.858 |
| FRAX — 8 CPU devices | 25.796 | 16.506 | 16.589 | 30,090.1 | 33.234 |

## What each column means

**First-use full includes any required initial compilation. All Warm columns are measured after compilation is prepared and exclude compilation time.**

| Column | Meaning | Initial compilation included? |
| --- | --- | --- |
| **Setup** | Pipeline, number of workers or CPU devices, and chunk/refinement setting. | — |
| **First-use full (s)** | Total seconds from harness entry through imports, shared model/worker setup, required compilation, loading/preparation, retargeting, enabled cleanup, and saved outputs. Installation and downloads are excluded. | **Yes, where required.** |
| **Warm processing (s)** | Seconds to process all 77 motions after reusable setup and compilation are ready. Includes input loading/forward kinematics, target preparation, per-motion initialization, retargeting, and enabled cleanup. Saving is excluded. | **No.** |
| **Warm full (s)** | Warm processing plus saving the robot trajectories. Shared startup/setup and compilation remain excluded: this is loading through saving in an initialized process. | **No.** |
| **Warm FPS** | Original frames processed per second: `496,672 / Warm processing seconds`. Uses the processing column, so saving is excluded. | **No.** |
| **Warm µs/frame** | Average processing wall time per original frame: `1,000,000 × Warm processing seconds / 496,672`. Lower is faster. | **No.** |

Compilation preparation differs by pipeline:

- **FRDV V1:** CasADi-generated C functions are compiled and cached for reuse.
- **FRDV V2:** the native C library is built on first use and cached for reuse.
- **FRAX:** JAX compiles the tested clip shapes/device configuration. An untimed warmup prepares all actual shapes; the timed warm pass is checked for **zero new reported compilation**.
- **GMR:** this BVH retargeting path uses Mink/DAQP without a separate retargeter-JIT warmup. First-use full still includes imports and model/worker setup.

Every warm measurement retargets all 77 motions again.

FPS = 496,672 / processing seconds; µs/frame = 1,000,000 × processing seconds / 496,672. Parallel values are aggregate throughput, not single-frame latency. Worker/device limits are not hard core affinity on macOS. Each pipeline retains its declared default task settings.
