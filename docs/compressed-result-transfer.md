# Compressed Result Transfer

**Distributed Collector** has two optional widgets that control how workers send their images back to the master. This matters most when workers run in the cloud and you pay for data leaving it (egress).

| `transfer_format` | Type | Notes |
|---|---|---|
| `legacy_png` (default) | lossless | Original behaviour: one JSON request per frame, uncompressed PNG in base64. Works with any master version. |
| `png` | lossless | Compressed PNG, binary, several frames per request. Same tensors as `legacy_png`. |
| `ffv1` | lossless | Whole batch as one FFV1 video. Same tensors as `legacy_png`; best lossless choice for frame sequences. |
| `jpeg`, `webp` | lossy | Per-frame stills. Good for image batches. |
| `h264`, `h265`, `av1` | lossy | Whole batch as one video. By far the smallest for video frame batches. |

`transfer_quality` (1–100, default 90) applies to the lossy formats. JPEG/WebP use it directly; video codecs map it to CRF (90 → CRF 13 for h264/h265, CRF 16 for av1; 100 → CRF 0).

Measured on a 121-frame 1280×720 batch (one worker's result):

| Format | Sent | vs legacy | PSNR |
|---|---|---|---|
| `legacy_png` | 446 MB | 1× | lossless |
| `png` | 163 MB | 2.7× | lossless |
| `ffv1` | 128 MB | 3.5× | lossless |
| `jpeg` q90 | 27 MB | 17× | 42.7 dB |
| `webp` q90 | 13 MB | 33× | 40.6 dB |
| `h264` q90 | 2.4 MB | 186× | 40.2 dB |
| `h265` q90 | 1.0 MB | 434× | 40.1 dB |
| `av1` q90 | 0.4 MB | 1031× | 40.5 dB |

The test clip is a slow pan, which video codecs compress especially well; expect smaller (still large) gains on content with more motion.

Notes:
- Choose the format on the master; the value is sent to workers with the workflow.
- The master must run this version to accept compressed results. A worker sending to an older master gets a 400 on its first request and automatically resends with `legacy_png`.
- If a worker's PyAV build lacks the requested video encoder, it falls back to `h264` (lossy formats) or `png` (`ffv1`), and logs it.
- Lossy formats only affect frames produced by workers; the master's own frames are never re-encoded.
- Request size is bounded on the final encoded bytes. Frames are batched into requests of up to 32 MiB (`COMFYUI_DISTRIBUTED_TRANSFER_CHUNK_BYTES`); a single large frame may go alone in a request of up to 96 MiB (`COMFYUI_DISTRIBUTED_TRANSFER_MAX_REQUEST_BYTES`). Keep the latter below the master's `--max-upload-size` (default 100 MB) if you change either.
- All chunks are encoded before anything is sent. If a single frame cannot fit within the request limit, the worker fails with an error and sends nothing; use a lossy format or raise both limits.
- A rejection other than the older-master case above (for example the master's size limit, HTTP 413) is reported as an error rather than retried as `legacy_png`.
- Each worker logs what it sent, e.g. `Sent 121 image(s) as h264 in 1 request(s): 2.40 MB (raw 334.49 MB, 139.4x smaller)`.
