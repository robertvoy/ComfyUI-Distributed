# Compressed and Exact Result Transfer

**Distributed Collector** (`DistributedCollector`) collects images and/or audio. Its `transfer_format` and `transfer_quality` widgets control **remote image returns only**; they do not re-encode master-local images or change audio transport. The saved defaults remain `legacy_png` and quality `90`.

| `transfer_format` | Fidelity | Direct-worker transport |
|---|---|---|
| `legacy_png` (default) | 8-bit pixels | Original behaviour: one JSON request per frame, uncompressed PNG in base64. Compatible with legacy masters. |
| `png` | Lossless 8-bit pixels | Compressed PNG, binary, several frames per request. |
| `ffv1` | Lossless 8-bit pixels | FFV1 video chunks; useful for frame sequences. |
| `jpeg`, `webp` | Lossy 8-bit images | Per-frame stills. |
| `h264`, `h265` (**experimental**), `av1` | Lossy 8-bit images | Video chunks; useful for similar consecutive frames. |
| `exact` (opt-in) | Original floating-point tensors | Binary multipart chunks with one raw tensor payload and `tensor-le-v1` descriptor per frame. No 8-bit quantization. |

“Lossless” PNG/FFV1 means lossless **after 8-bit quantization**, not preservation of the source float tensor. Direct non-exact returns decode to RGB float32. `exact` preserves supported floating-point dtype, shape, channels and values: float16, bfloat16, float32 or float64, finite dense BHWC tensors with 1, 3 or 4 channels. It generally costs substantially more bandwidth.

`transfer_quality` (1–100, default 90) applies to lossy formats, not `exact`, PNG or FFV1. JPEG/WebP use it directly; video codecs map it to CRF (90 → CRF 13 for h264/h265, CRF 16 for av1; 100 → CRF 0). A quality of 100 is not a substitute for exact float transport.

## Serverless profiles

- **`comfy-media-v1`:** the same Collector supplies the endpoint's image-return policy. The saved `legacy_png` default is mapped to compressed **`png`** with the same 8-bit pixel precision; normal serverless returns are not exact by default. All other formats retain their selection, including opt-in `exact` and experimental `h265`. Missing codecs or rejected formats fail instead of selecting another codec. Typed image returns preserve channel count and carry alpha separately where needed. See [serverless images and audio](serverless-media.md).
- **`legacy-image-v1`:** retains its image-file/`SaveImage` contract, not the typed return-format controls. Selecting a legacy pool when any Collector requests `exact` is rejected before dispatch; there is no 8-bit fallback. Audio requires direct workers or `comfy-media-v1`.

## Direct-worker compatibility and limits

- Choose the format on the master; the value is sent to workers with the workflow.
- **`exact` requires support on both master and worker.** If an older master rejects an exact chunk, the worker raises an error and never resends it as `legacy_png` or another 8-bit format. An exact Collector also rejects direct-worker image returns labelled with a non-exact format. Update both sides or explicitly select a non-exact format.
- For **non-exact compressed** formats only, the specific first-request older-master rejection (`400` with `Invalid JSON payload`) causes a retry using `legacy_png`. Other rejections, including HTTP 413, remain errors.
- For direct video returns only, an unavailable encoder falls back to `h264` for lossy video if available, otherwise `png`; unavailable `ffv1` falls back to `png`. The worker logs this. This compatibility fallback does **not** apply to `exact` or typed serverless returns.
- Multipart requests use a 32 MiB chunk target (`COMFYUI_DISTRIBUTED_TRANSFER_CHUNK_BYTES`) and a 96 MiB request budget (`COMFYUI_DISTRIBUTED_TRANSFER_MAX_REQUEST_BYTES`), reserving space for the envelope and accounting separately for audio. A single large frame may exceed the chunk target but must fit the request budget. Keep the request budget below the master's `--max-upload-size` (default 100 MB).
- Non-exact compressed chunks are all encoded before sending. Exact transfer instead plans frame sizes from shape/dtype before sending and encodes one chunk at a time. An oversized individual frame is rejected before any chunks are sent; choose another format or raise the worker and master limits together.
- Audio is attached to the final image chunk when it fits, otherwise sent separately. Audio-only completions use the canonical JSON envelope. Direct audio uses float32 waveform bytes regardless of the image format; `exact` is an IMAGE option, not a new audio codec.
- Each worker logs the format and bytes sent. These request limits are for direct multipart transfers; serverless request/artifact limits are documented separately.

## Previously documented size example

The following measurements are retained from a 121-frame 1280×720 slow-pan batch (one worker's result); they are not new validation of the current implementation or exact transport.

| Format | Sent | vs legacy | PSNR |
|---|---|---|---|
| `legacy_png` | 446 MB | 1× | lossless at 8-bit precision |
| `png` | 163 MB | 2.7× | lossless at 8-bit precision |
| `ffv1` | 128 MB | 3.5× | lossless at 8-bit precision |
| `jpeg` q90 | 27 MB | 17× | 42.7 dB |
| `webp` q90 | 13 MB | 33× | 40.6 dB |
| `h264` q90 | 2.4 MB | 186× | 40.2 dB |
| `h265` q90 (experimental) | 1.0 MB | 434× | 40.1 dB |
| `av1` q90 | 0.4 MB | 1031× | 40.5 dB |

Video codecs compress this slow pan especially well; gains depend on content and motion.
