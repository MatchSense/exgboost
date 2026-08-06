# CUDA Support

EXGBoost can be built against XGBoost with CUDA enabled. CUDA support is source-built; the default precompiled CPU NIF path should not be treated as a CUDA distribution mechanism.

## Host Prerequisites

For GPU execution, the host must have:

- An NVIDIA GPU.
- A working NVIDIA Linux driver; `nvidia-smi` should run on the host.
- Docker configured with the NVIDIA Container Toolkit if running in a container or devcontainer.

Install the NVIDIA Container Toolkit on the host, not inside the devcontainer. On Ubuntu/Debian hosts, follow the [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then configure Docker with:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

NVIDIA documents that `nvidia-ctk runtime configure --runtime=docker` updates `/etc/docker/daemon.json` so Docker can use the NVIDIA runtime.

Verify Docker can see the GPU from a container:

```bash
docker run --rm --gpus all \
  nvidia/cuda:13.2.1-base-ubuntu24.04 \
  nvidia-smi
```

The CUDA image tag should be compatible with the CUDA version reported by host `nvidia-smi`. The CUDA devcontainer currently uses `nvidia/cuda:13.2.1-devel-ubuntu24.04`, so a host driver that supports CUDA 13.2 or newer is expected.

Contributor devcontainer guidance lives in [CONTRIBUTING.md](../CONTRIBUTING.md#devcontainer-choice).

## Development In This Repo

Open the repo with `.devcontainer/cuda/devcontainer.json`. The CUDA devcontainer sets these environment variables automatically:

```bash
USE_CUDA=ON
USE_NCCL=OFF
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

Then compile as usual:

```bash
mix deps.get
mix compile
```

Verify the native XGBoost build was compiled with CUDA:

```bash
iex -S mix
```

```elixir
EXGBoost.xgboost_build_info()["USE_CUDA"]
```

Expected:

```elixir
true
```

Run the device smoke test with automatic device selection:

```bash
mix test test/build_device_test.exs
```

Or force the device path:

```bash
EXGBOOST_TEST_DEVICE=cpu mix test test/build_device_test.exs
EXGBOOST_TEST_DEVICE=cuda mix test test/build_device_test.exs
```

Run the Covertype benchmark from a CUDA-enabled build with:

```bash
mix run bench/covtype.exs
```

Use `DEVICE=cpu`, `DEVICE=cuda`, or `DEVICE=both` to select benchmark devices, e.g.

```bash
DEVICE=both mix run bench/covtype.exs
```

Generated benchmark data is stored under `bench/data`, which is ignored by git.

## Manual CUDA Builds

For manual builds outside the CUDA devcontainer, pass the Makefile flags yourself:

```bash
USE_CUDA=ON USE_NCCL=OFF mix compile
```

Optionally restrict the generated GPU architectures to your target cards:

```bash
USE_CUDA=ON USE_NCCL=OFF CUDA_ARCHITECTURES=89 mix compile
```

For broad Ampere-and-newer coverage, use:

```bash
USE_CUDA=ON USE_NCCL=OFF CUDA_ARCHITECTURES='80;86;89;90' mix compile
```

The same build can also be selected with `EXGBOOST_TARGET`, which is the variant name used for precompiled artifact lookup:

```bash
EXGBOOST_TARGET=cuda80_86_89_90 mix compile
```

That covers:

- `80`: Ampere data center, such as A100 and A30.
- `86`: Ampere consumer/workstation, such as RTX 30xx, A10, A40, and RTX A-series.
- `89`: Ada, such as RTX 40xx, L4, L40/L40S, and RTX Ada workstation cards.
- `90`: Hopper, such as H100, H200, and GH200.

CPU and CUDA XGBoost source builds use separate cache directories, so switching `USE_CUDA` does not overwrite the XGBoost source build cache. The final NIF under `_build/.../lib/exgboost/priv`, however, is whichever variant was compiled most recently for that Mix build.

## Using EXGBoost From Another App

When EXGBoost is a dependency of another application, CUDA must be enabled when the dependency is compiled, not just when the parent app runs. This local-build path requires the consuming app's build environment to have the full native toolchain, CMake, Ninja, GCC/G++, and the CUDA toolkit. That requirement is the main reason CUDA-specific prebuilt artifacts would be useful.

The commands below are for source-building the dependency in the consuming app. They are not equivalent to a prebuilt CUDA distribution.

In the consuming app:

```bash
CC_PRECOMPILER_PRECOMPILE_ONLY_LOCAL=true \
USE_CUDA=ON \
USE_NCCL=OFF \
CUDA_ARCHITECTURES='80;86;89;90' \
mix deps.compile exgboost --force
```

Then verify from the consuming app:

```bash
iex -S mix
```

```elixir
EXGBoost.xgboost_build_info()["USE_CUDA"]
```

Expected:

```elixir
true
```

Use CUDA explicitly when training:

```elixir
EXGBoost.train(x, y,
  device: :cuda,
  tree_method: :hist,
  objective: :reg_squarederror,
  num_boost_rounds: 100
)
```

For a specific GPU ordinal:

```elixir
EXGBoost.train(x, y,
  device: {:cuda, 0},
  tree_method: :hist
)
```

If your app builds in Docker or CI, put the environment variables in the image or release build step:

```dockerfile
ENV CC_PRECOMPILER_PRECOMPILE_ONLY_LOCAL=true
ENV USE_CUDA=ON
ENV USE_NCCL=OFF
ENV CUDA_ARCHITECTURES=80;86;89;90

RUN mix deps.get
RUN mix deps.compile exgboost --force
RUN mix compile
```

For production systems that require GPU execution, add a startup check so a later dependency rebuild cannot silently replace the CUDA build with a CPU build:

```elixir
unless EXGBoost.xgboost_build_info()["USE_CUDA"] do
  raise "EXGBoost was not compiled with CUDA"
end
```

The runtime environment must also expose compatible NVIDIA drivers, CUDA runtime libraries, and GPU devices. For Docker this usually means launching with `--gpus all`; for Kubernetes this usually means using the NVIDIA device plugin.

## CUDA Toolkit And Architecture Compatibility

The CUDA build image supplies the compiler toolchain. A machine does not need a specific GPU model to compile for that GPU architecture. For example, `nvidia/cuda:13.2.1-devel-ubuntu24.04` can build all supported Ampere-and-newer variants used by EXGBoost:

```bash
EXGBOOST_TARGET=cuda80 mix elixir_make.precompile
EXGBOOST_TARGET=cuda86 mix elixir_make.precompile
EXGBOOST_TARGET=cuda89 mix elixir_make.precompile
EXGBOOST_TARGET=cuda90 mix elixir_make.precompile
EXGBOOST_TARGET=cuda80_86_89_90 mix elixir_make.precompile
```

Those targets map to `CUDA_ARCHITECTURES=80`, `86`, `89`, `90`, or `80;86;89;90` respectively. Each target gets its own artifact variant and XGBoost source build cache.

A newer CUDA toolkit can generally compile for older supported SM architectures, but not for architectures it has dropped or does not know about. Runtime validation still requires compatible GPU hardware and a host NVIDIA driver new enough for the CUDA runtime used by the build.

## Build Without A Local GPU

A local GPU is not required to compile CUDA code. It is required to run or test CUDA execution.

A build-only environment needs the CUDA toolkit, such as an `nvidia/cuda:*devel*` image, and an explicit `CUDA_ARCHITECTURES` value:

```bash
USE_CUDA=ON USE_NCCL=OFF CUDA_ARCHITECTURES='80;86;89;90' mix compile
```

Runtime validation still requires a GPU:

```bash
EXGBOOST_TEST_DEVICE=cuda mix test test/build_device_test.exs
```

## Prebuilt CUDA Artifacts

A CUDA prebuilt artifact lets consuming apps avoid installing compilers, CMake, Ninja, and the CUDA toolkit just to compile EXGBoost.

EXGBoost uses an Evision-inspired variant naming model for CPU and CUDA artifacts. [Evision](https://github.com/cocoa-xu/evision) is an Apache-2.0 licensed Elixir OpenCV binding that uses explicit precompiled artifact variants for CPU, contrib, CUDA, and cuDNN builds. The EXGBoost implementation follows that packaging pattern without copying Evision code.

`EXGBOOST_TARGET` selects the variant used for artifact lookup and source-build fallback. Supported values are:

- `cpu`
- `cuda80`
- `cuda86`
- `cuda89`
- `cuda90`
- `cuda80_86_89_90`

The normal `elixir_make` / `cc_precompiler` artifact naming scheme distinguishes NIF ABI, platform, and package version. It does not distinguish CPU versus CUDA builds, CUDA toolkit versions, or CUDA architecture sets. EXGBoost adds the variant to the target portion of the artifact name.

Examples:

```text
exgboost-nif-2.18-x86_64-linux-gnu-cpu-0.11.0.tar.gz
exgboost-nif-2.18-x86_64-linux-gnu-cuda89-0.11.0.tar.gz
exgboost-nif-2.18-x86_64-linux-gnu-cuda80_86_89_90-0.11.0.tar.gz
```

This keeps one Hex package while avoiding CPU/CUDA artifact collisions. CPU artifacts remain the default release path. CUDA artifacts should initially be Linux-only and opt-in through `EXGBOOST_TARGET`, with runtime validation performed on GPU-capable runners or deployment hosts.
