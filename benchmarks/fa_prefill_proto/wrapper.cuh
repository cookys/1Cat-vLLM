// Pointer-only ABI: no torch/JIT dependency, no GPU calls until explicit launch.
extern "C" int fa_launch(
    const void* q, const void* k, const void* v, void* out, void* lse,
    const void* table, const void* lengths, int b, int h, int m, int cols, int hk,
    int64_t kb, int64_t kt, int64_t kh, int64_t vb, int64_t vt, int64_t vh,
    float scale, uint64_t stream_bits) {
  astraPROTO::astra_fa_protoPROTO_kernel<true, true, true>
      <<<dim3((m + 31) / 32, 1, b * h), 512, 0,
          reinterpret_cast<cudaStream_t>(stream_bits)>>>(
      reinterpret_cast<const __half*>(q), reinterpret_cast<const __half*>(k),
      reinterpret_cast<const __half*>(v), reinterpret_cast<__half*>(out),
      reinterpret_cast<float*>(lse), reinterpret_cast<const int*>(table),
      reinterpret_cast<const int*>(lengths), h, m, cols, hk,
      kb, kt, kh, vb, vt, vh, scale);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int fa_resources(int* regs, int* shared, int* local, int* blocks) {
  cudaFuncAttributes attrs;
  auto kernel = astraPROTO::astra_fa_protoPROTO_kernel<true, true, true>;
  cudaError_t e = cudaFuncGetAttributes(&attrs, kernel);
  if (e != cudaSuccess) return static_cast<int>(e);
  *regs = attrs.numRegs; *shared = attrs.sharedSizeBytes; *local = attrs.localSizeBytes;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks, kernel, 512, 0);
  return static_cast<int>(e);
}
