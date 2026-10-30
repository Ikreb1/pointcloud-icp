# pointcloud-icp

Real-time point cloud registration (Iterative Closest Point), taken step by step from a plain C++ baseline to SIMD/multithreaded CPU code to Vulkan compute, with every step benchmarked.

The point is not a new algorithm. It is to measure where time goes in a real 3D vision workload and get it back, the same way you would profile a frame in a game.

## Plan

| Stage | What | Measure |
|---|---|---|
| 1 | Scalar C++ baseline: point-to-point ICP, brute-force nearest neighbour | correctness vs known transform, ms per iteration |
| 2 | Spatial acceleration (k-d tree or voxel grid) | speedup vs stage 1 |
| 3 | SIMD (AVX2 / NEON) + data layout (SoA) + multithreading | speedup, cache misses |
| 4 | Vulkan compute port of nearest neighbour + reduction | GPU time, occupancy, memory bandwidth |
| 5 | Vulkan viewer: draw both clouds snapping together each iteration, reading straight from the compute buffers (Vulkan 1.3 dynamic rendering) | frame time, zero-copy compute to graphics |
| 6 | Write-up with profiler captures (Tracy, Nsight / Radeon GPU Profiler) and a GIF from the viewer | |

Targets: x86 desktop (RTX 3070) and ARM (Raspberry Pi or similar) for the CPU stages.

## Learning goals

- CPU performance: data layout, SIMD intrinsics, multithreading, profiling cache behaviour
- Vulkan from scratch: compute first (buffers, descriptors, dispatch, parallel reduction), then graphics (swapchain, sync, pipelines) for the viewer
- GPU profiling and explaining performance work clearly

## Data

Public point cloud datasets (e.g. Stanford bunny, TUM RGB-D, KITTI subsets), plus synthetic clouds with known transforms for testing.

## Status

Stage 0: README only.
