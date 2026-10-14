# Benchmark boundaries

The historical classifier timing loop divides the sum by repetitions and then divides the reported average by repetitions again. It also omits explicit CUDA synchronization. Those measurements must not be reused as newly verified release performance, and no claim is made that this exact source generated the paper's reported latency.

The maintained benchmark constructs/loads a model and its inputs before timing, performs warmup, uses evaluation/no-gradient mode for PyTorch, synchronizes CUDA execution, and reads TensorFlow outputs before ending each timed interval. It reports all intervals, mean/median/spread/p95, actual batch size, and per-spectrum time at that batch size. No second repetition divisor is applied.

The model-only boundary excludes input normalization and input host/device transfer. TensorFlow output materialization and synchronization remain included. The full runner report instead includes Python worker startup, file I/O, transformation, both comparison classifier passes, and orchestration; it is not directly comparable to single-forward model latency.

For benchmarks, `--device cpu` explicitly places TensorFlow execution on CPU; `--device cuda` requires an accessible TensorFlow GPU and targets GPU 0. An unavailable GPU is an error, not a CPU timing result labeled as CUDA. For cascaded `predict`, `--device` selects the PyTorch classifier device and `--translation-device` selects TensorFlow independently. Translation defaults to CPU. Both choices are recorded; PyTorch CUDA availability does not establish TensorFlow CUDA availability.

Parameter-array bytes and serialized weight-file bytes are separate quantities. TensorFlow model counts include trainable and nontrainable graph weights; PyTorch parameter counts exclude running-statistic buffers. State-file bytes include format overhead and buffers, and may include optimizer state if saving a compiled TensorFlow model. Avoid claims of a 9 MB combined model based on the paper without checking the actual artifacts and their precision/compression.

CPU measurements from the verified runtime are software/hardware-specific. TensorFlow CUDA libraries were unavailable there; PyTorch CUDA availability is independent. No portable-spectrometer or edge-device timing is established by the laptop verification.
