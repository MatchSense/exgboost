# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.10.0

### Added

- Comprehensive safety validation tests in `test/nif_test.exs` covering:
  - Invalid typestr format rejection
  - Binary size validation preventing buffer overflows
  - Boolean type strictness (only `true`/`false` atoms accepted)
  - Shape overflow protection for extremely large dimensions
  - Endianness marker validation

### Changed

- **NIF Layer Safety Improvements**:
  - Buffer overflow protection: Dynamic allocation with overflow-checked arithmetic replaces fixed-size buffers in Array Interface JSON builder
  - Strict type safety: All pointer arithmetic now uses `uintptr_t` with `PRIuPTR` formatting
  - Platform-independent integer conversions: `enif_get_ulong`/`enif_make_ulong` replaced with `enif_get_uint64`/`enif_make_uint64`
  - Comprehensive input validation: Added validation for typestr format, shape dimensions, boolean values, and binary sizes
  - Memory safety: Binary size validation ensures data buffers match expected sizes before any memory operations
  - Single cleanup path: Refactored to use consistent resource management with single cleanup labels to prevent memory leaks
  - Atomic data copying: All data returned from XGBoost is now copied atomically within NIF calls, eliminating pointer lifetime issues
  - Added static assertions and overflow checks for VLA (Variable Length Array) declarations

- **ArrayInterface Optimization**:
  - Removed `address` field - pointer addresses are never exposed to Elixir for safety
  - Removed `tensor` field - eliminates duplicate memory storage; tensors now reconstructed on-demand from binary data
  - Removed `Jason.Encoder` protocol implementation - no longer needed as ArrayInterface is not serialized to JSON
  - Updated `Inspect` protocol to show `readonly` directly instead of `data: [address, readonly]`
  - `get_tensor/1` now reconstructs tensors from binary data instead of caching, trading minimal performance for significant memory savings
  - Updated `from_map/1` to ignore address values from incoming data, only extracting readonly flag

- **DMatrix Improvements**:
  - `get_quantile_cut/1` refactored to return maps with `:binary`, `:typestr`, `:shape` instead of JSON with memory addresses
  - Added internal `build_tensor_from_map/1` helper to reconstruct tensors from NIF-returned data
  - All data copying happens atomically within C before returning to Elixir

- **Unsafe APIs marked as DEPRECATED - to be removed in future releases**:
  - `EXGBoost.NIF.get_binary_from_address/2` - arbitrary memory read primitive that could crash the BEAM VM
  - `exg_get_binary_from_address` C NIF function and all declarations
  - Address-based tensor reconstruction in `ArrayInterface.get_tensor/1`
  - Tensor caching in `ArrayInterface` struct

### Fixed

- Cross-platform compatibility: Fixed `unsigned long` → `ErlNifUInt64` conversions to prevent data truncation on Windows (LP64 vs LLP64 calling conventions)
- Memory safety: Eliminated all pointer lifetime gaps where Elixir code held addresses to freed memory
- Buffer safety: All shape-to-JSON conversions now use dynamic allocation with proper bounds checking

### Security

- Eliminated arbitrary memory read primitive that allowed reading from any address
- All binary data is now validated for size before access, preventing buffer overflows
- Pointer addresses are never exposed to Elixir, preventing use-after-free vulnerabilities
- Strict type validation prevents type confusion attacks

## [0.9.1]

### Removed

- Kino and Livebook Integration; `kino` doesn't seem like it's under active development, make this a pure library.
- Mark `EXGBoost.DMatrix.from_file` as deprecated.
