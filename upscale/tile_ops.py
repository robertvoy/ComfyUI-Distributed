import math, torch
from contextlib import nullcontext
from PIL import Image, ImageFilter, ImageDraw
from typing import List, Tuple
import comfy.samplers, comfy.model_management
from ..utils.logging import debug_log, log
from ..utils.image import tensor_to_pil, pil_to_tensor
from ..utils.usdu_utils import crop_cond, get_crop_region, expand_crop
from ..utils.crop_model_patch import crop_model_cond
from .conditioning import clone_conditioning


class TileOpsMixin:
    def round_to_multiple(self, value: int, multiple: int = 8) -> int:
        """Round value to nearest multiple."""
        return round(value / multiple) * multiple

    def calculate_tiles(self, image_width: int, image_height: int,
                       tile_width: int, tile_height: int, force_uniform_tiles: bool = True) -> List[Tuple[int, int]]:
        """Calculate tile positions to match Ultimate SD Upscale.

        Positions are a simple grid starting at (0,0) with steps of
        `tile_width` and `tile_height`, using ceil(rows/cols) to cover edges.
        Uniform vs non-uniform affects only crop/resize, not positions.
        """
        rows = math.ceil(image_height / tile_height)
        cols = math.ceil(image_width / tile_width)
        tiles: List[Tuple[int, int]] = []
        for yi in range(rows):
            for xi in range(cols):
                tiles.append((xi * tile_width, yi * tile_height))
        return tiles

    def extract_tile_with_padding(self, image: torch.Tensor, x: int, y: int,
                                 tile_width: int, tile_height: int, padding: int,
                                 force_uniform_tiles: bool) -> Tuple[torch.Tensor, int, int, int, int]:
        """Extract a tile region and resize to match USDU cropping logic.

        Mirrors ComfyUI_UltimateSDUpscale processing:
        - Build a mask with a white rectangle at the tile rect
        - Compute crop_region via get_crop_region(mask, padding)
        - If force_uniform_tiles: expand by crop/aspect ratio, then resize to
          fixed processing size of round_to_multiple(tile + padding)
        - Else: target is ceil(crop_size/8)*8 per dimension
        - Extract the crop and resize to target tile_size
        Returns the resized tensor and crop origin/size for blending.
        """
        _, h, w, _ = image.shape

        # Create mask and compute initial padded crop region
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x, y, x + tile_width, y + tile_height], fill=255)
        x1, y1, x2, y2 = get_crop_region(mask, padding)

        # Determine crop + processing size
        if force_uniform_tiles:
            process_w = self.round_to_multiple(tile_width + padding, 8)
            process_h = self.round_to_multiple(tile_height + padding, 8)
            crop_w = x2 - x1
            crop_h = y2 - y1
            crop_ratio = crop_w / crop_h if crop_h != 0 else 1.0
            process_ratio = process_w / process_h if process_h != 0 else 1.0
            if crop_ratio > process_ratio:
                target_w = crop_w
                target_h = round(crop_w / process_ratio) if process_ratio != 0 else crop_h
            else:
                target_w = round(crop_h * process_ratio)
                target_h = crop_h
            (x1, y1, x2, y2), _ = expand_crop((x1, y1, x2, y2), w, h, target_w, target_h)
            target_w = process_w
            target_h = process_h
        else:
            crop_w = x2 - x1
            crop_h = y2 - y1
            target_w = max(8, math.ceil(crop_w / 8) * 8)
            target_h = max(8, math.ceil(crop_h / 8) * 8)
            (x1, y1, x2, y2), (target_w, target_h) = expand_crop((x1, y1, x2, y2), w, h, target_w, target_h)

        # Actual extracted size before resizing (for blending)
        extracted_width = x2 - x1
        extracted_height = y2 - y1

        # Extract tile and resize to processing size
        tile = image[:, y1:y2, x1:x2, :]
        tile_pil = tensor_to_pil(tile, 0)
        if tile_pil.size != (target_w, target_h):
            tile_pil = tile_pil.resize((target_w, target_h), Image.LANCZOS)

        tile_tensor = pil_to_tensor(tile_pil)
        if image.is_cuda:
            tile_tensor = tile_tensor.cuda()

        return tile_tensor, x1, y1, extracted_width, extracted_height

    def extract_batch_tile_with_padding(self, images: torch.Tensor, x: int, y: int,
                                        tile_width: int, tile_height: int, padding: int,
                                        force_uniform_tiles: bool) -> Tuple[torch.Tensor, int, int, int, int]:
        """Extract a tile region for the entire batch and resize to USDU logic.

        - Computes a single crop region from a mask at (x,y,w,h) with padding
        - force_uniform_tiles controls target processing size logic
        - Returns a batched tensor [B,H',W',C] and crop origin/size for blending
        """
        batch, h, w, _ = images.shape

        # Create mask and compute initial padded crop region (same for all images)
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x, y, x + tile_width, y + tile_height], fill=255)
        x1, y1, x2, y2 = get_crop_region(mask, padding)

        # Determine crop + processing size
        if force_uniform_tiles:
            process_w = self.round_to_multiple(tile_width + padding, 8)
            process_h = self.round_to_multiple(tile_height + padding, 8)
            crop_w = x2 - x1
            crop_h = y2 - y1
            crop_ratio = crop_w / crop_h if crop_h != 0 else 1.0
            process_ratio = process_w / process_h if process_h != 0 else 1.0
            if crop_ratio > process_ratio:
                target_w = crop_w
                target_h = round(crop_w / process_ratio) if process_ratio != 0 else crop_h
            else:
                target_w = round(crop_h * process_ratio)
                target_h = crop_h
            (x1, y1, x2, y2), _ = expand_crop((x1, y1, x2, y2), w, h, target_w, target_h)
            target_w = process_w
            target_h = process_h
        else:
            crop_w = x2 - x1
            crop_h = y2 - y1
            target_w = max(8, math.ceil(crop_w / 8) * 8)
            target_h = max(8, math.ceil(crop_h / 8) * 8)
            (x1, y1, x2, y2), (target_w, target_h) = expand_crop((x1, y1, x2, y2), w, h, target_w, target_h)

        extracted_width = x2 - x1
        extracted_height = y2 - y1

        # Slice batch region
        tiles = images[:, y1:y2, x1:x2, :]

        # Resize each tile to target size
        resized_tiles = []
        for i in range(batch):
            tile_pil = tensor_to_pil(tiles, i)
            if tile_pil.size != (target_w, target_h):
                tile_pil = tile_pil.resize((target_w, target_h), Image.LANCZOS)
            resized_tiles.append(pil_to_tensor(tile_pil))
        tile_batch = torch.cat(resized_tiles, dim=0)

        if images.is_cuda:
            tile_batch = tile_batch.cuda()

        return tile_batch, x1, y1, extracted_width, extracted_height

    def process_tile(self, tile_tensor: torch.Tensor, model, positive, negative, vae,
                     seed: int, steps: int, cfg: float, sampler_name: str, 
                     scheduler: str, denoise: float, tiled_decode: bool = False,
                     batch_idx: int = 0, region: Tuple[int, int, int, int] = None,
                     image_size: Tuple[int, int] = None) -> torch.Tensor:
        """Process a single tile through SD sampling. 
        Note: positive and negative should already be pre-sliced for the current batch_idx."""
        debug_log(f"[process_tile] Processing tile for batch_idx={batch_idx}, seed={seed}, region={region}")
        
        
        # Import here to avoid circular dependencies
        from nodes import common_ksampler, VAEEncode, VAEDecode
        
        # Try to import tiled VAE nodes if available
        try:
            from nodes import VAEEncodeTiled, VAEDecodeTiled
            tiled_vae_available = True
        except ImportError:
            tiled_vae_available = False
            if tiled_decode:
                debug_log("Tiled VAE nodes not available, falling back to standard VAE")
        
        # Convert to PIL and back to ensure clean tensor without gradient tracking
        tile_pil = tensor_to_pil(tile_tensor, 0)
        clean_tensor = pil_to_tensor(tile_pil)
        
        # Ensure tensor is detached and doesn't require gradients
        clean_tensor = clean_tensor.detach()
        if hasattr(clean_tensor, 'requires_grad_'):
            clean_tensor.requires_grad_(False)
        
        # Move to correct device
        if tile_tensor.is_cuda:
            clean_tensor = clean_tensor.cuda()
            clean_tensor = clean_tensor.detach()  # Detach again after device transfer
        
        # Clone conditioning per tile (shares models, clones hints for cropping)
        positive_tile = clone_conditioning(positive, clone_hints=True)
        negative_tile = clone_conditioning(negative, clone_hints=True)
        
        # Crop conditioning to tile region if provided (assumes hints at image resolution)
        if region is not None and image_size is not None:
            init_size = image_size  # (width, height) of full image
            canvas_size = image_size
            tile_size = (tile_tensor.shape[2], tile_tensor.shape[1])  # (width, height)
            w_pad = 0  # No extra pad needed; region already includes padding
            h_pad = 0
            positive_cropped = crop_cond(positive_tile, region, init_size, canvas_size, tile_size, w_pad, h_pad)
            negative_cropped = crop_cond(negative_tile, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        else:
            # No region cropping needed, use cloned conditioning as-is
            positive_cropped = positive_tile
            negative_cropped = negative_tile
        
        # Encode to latent (always non-tiled, matching original node)
        latent = VAEEncode().encode(vae, clean_tensor)[0]
        
        # Sample with model patch cropping parity (ControlNet patch hints)
        if region is not None and image_size is not None:
            model_ctx = crop_model_cond(
                model,
                region,
                image_size,
                image_size,
                (clean_tensor.shape[2], clean_tensor.shape[1]),
            )
        else:
            model_ctx = nullcontext(model)
        with model_ctx as model_for_sampling:
            samples = common_ksampler(
                model_for_sampling, seed, steps, cfg, sampler_name, scheduler,
                positive_cropped, negative_cropped, latent, denoise=denoise
            )[0]
        
        # Decode back to image
        if tiled_decode and tiled_vae_available:
            image = VAEDecodeTiled().decode(vae, samples, tile_size=512)[0]
        else:
            image = VAEDecode().decode(vae, samples)[0]
        
        return image

    def process_tiles_batch(self, tile_batch: torch.Tensor, model, positive, negative, vae,
                            seed: int, steps: int, cfg: float, sampler_name: str,
                            scheduler: str, denoise: float, tiled_decode: bool,
                            region: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> torch.Tensor:
        """Process a batch of tiles together (USDU behavior).

        tile_batch: [B, H, W, C]
        Returns image batch tensor [B, H, W, C]
        """
        # Import locally to avoid circular deps
        from nodes import common_ksampler, VAEEncode, VAEDecode
        try:
            from nodes import VAEEncodeTiled, VAEDecodeTiled
            tiled_vae_available = True
        except ImportError:
            tiled_vae_available = False

        # Detach and move device
        clean = tile_batch.detach()
        if hasattr(clean, 'requires_grad_'):
            clean.requires_grad_(False)
        if tile_batch.is_cuda:
            clean = clean.cuda().detach()

        # Clone/crop conditioning once for the region
        positive_tile = clone_conditioning(positive, clone_hints=True)
        negative_tile = clone_conditioning(negative, clone_hints=True)

        init_size = image_size
        canvas_size = image_size
        tile_size = (clean.shape[2], clean.shape[1])  # (W,H)
        w_pad = 0
        h_pad = 0
        positive_cropped = crop_cond(positive_tile, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        negative_cropped = crop_cond(negative_tile, region, init_size, canvas_size, tile_size, w_pad, h_pad)

        # Encode -> Sample -> Decode
        latent = VAEEncode().encode(vae, clean)[0]
        with crop_model_cond(model, region, image_size, image_size, tile_size) as model_for_sampling:
            samples = common_ksampler(
                model_for_sampling, seed, steps, cfg, sampler_name, scheduler,
                positive_cropped, negative_cropped, latent, denoise=denoise
            )[0]
        if tiled_decode and tiled_vae_available:
            image = VAEDecodeTiled().decode(vae, samples, tile_size=512)[0]
        else:
            image = VAEDecode().decode(vae, samples)[0]

        return image

    def create_tile_mask(self, image_width: int, image_height: int,
                        x: int, y: int, tile_width: int, tile_height: int, 
                        mask_blur: int) -> Image.Image:
        """Create a mask for blending tiles - matches Ultimate SD Upscale approach.
        
        Creates a black image with a white rectangle at the tile position,
        then applies blur to create soft edges.
        """
        # Create a full-size mask matching the image dimensions
        mask = Image.new('L', (image_width, image_height), 0)  # Black background
        
        # Draw white rectangle at tile position
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x, y, x + tile_width, y + tile_height], fill=255)
        
        # Apply blur to soften edges
        if mask_blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))
        
        return mask

    def blend_tile(self, base_image: Image.Image, tile_image: Image.Image,
                  x: int, y: int, extracted_size: Tuple[int, int], 
                  mask: Image.Image, padding: int) -> Image.Image:
        """Blend a processed tile back into the base image using Ultimate SD Upscale's exact approach.
        
        This follows the exact method from ComfyUI_UltimateSDUpscale/modules/processing.py
        """
        extracted_width, extracted_height = extracted_size
        
        # Debug logging (uncomment if needed)
        # debug_log(f"[Blend] Placing tile at ({x}, {y}), size: {extracted_width}x{extracted_height}")
        
        # Calculate the crop region that was used for extraction
        crop_region = (x, y, x + extracted_width, y + extracted_height)
        
        # The mask is already full-size, no need to crop
        
        # Resize the processed tile back to the extracted size
        if tile_image.size != (extracted_width, extracted_height):
            tile_resized = tile_image.resize((extracted_width, extracted_height), Image.LANCZOS)
        else:
            tile_resized = tile_image
        
        # Follow Ultimate SD Upscale blending approach:
        # Put the tile into position
        image_tile_only = Image.new('RGBA', base_image.size)
        image_tile_only.paste(tile_resized, crop_region[:2])
        
        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        temp.putalpha(mask)  # Use the full image mask
        image_tile_only.paste(temp, image_tile_only)
        
        # Add back the tile to the initial image according to the mask in the alpha channel
        result = base_image.convert('RGBA')
        result.alpha_composite(image_tile_only)
        
        # Convert back to RGB
        return result.convert('RGB')

    def _slice_conditioning(self, positive, negative, batch_idx):
        """Helper to slice conditioning for a specific batch index."""
        # Clone and slice conditioning properly, including ControlNet hints
        positive_sliced = clone_conditioning(positive)
        negative_sliced = clone_conditioning(negative)
        
        for cond_list in [positive_sliced, negative_sliced]:
            for i in range(len(cond_list)):
                emb, cond_dict = cond_list[i]
                if emb.shape[0] > 1:
                    cond_list[i][0] = emb[batch_idx:batch_idx+1]
                if 'control' in cond_dict:
                    control = cond_dict['control']
                    while control is not None:
                        hint = control.cond_hint_original
                        if hint.shape[0] > 1:
                            control.cond_hint_original = hint[batch_idx:batch_idx+1]
                        control = control.previous_controlnet
                if 'mask' in cond_dict and cond_dict['mask'].shape[0] > 1:
                    cond_dict['mask'] = cond_dict['mask'][batch_idx:batch_idx+1]
        
        return positive_sliced, negative_sliced

    def slice_conditioning(self, positive, negative, batch_idx):
        """Public conditioning-slice API."""
        return self._slice_conditioning(positive, negative, batch_idx)

    def _process_and_blend_tile(self, tile_idx, tile_pos, upscaled_image, result_image,
                               model, positive, negative, vae, seed, steps, cfg,
                               sampler_name, scheduler, denoise, tile_width, tile_height,
                               padding, mask_blur, image_width, image_height, force_uniform_tiles,
                               tiled_decode, batch_idx: int = 0):
        """Process a single tile and blend it into the result image."""
        x, y = tile_pos
        
        # Extract and process tile
        tile_tensor, x1, y1, ew, eh = self.extract_tile_with_padding(
            upscaled_image, x, y, tile_width, tile_height, padding, force_uniform_tiles
        )
        
        processed_tile = self.process_tile(tile_tensor, model, positive, negative, vae,
                                         seed, steps, cfg, sampler_name, 
                                         scheduler, denoise, tiled_decode, batch_idx=batch_idx,
                                         region=(x1, y1, x1 + ew, y1 + eh), image_size=(image_width, image_height))
        
        # Convert and blend
        processed_pil = tensor_to_pil(processed_tile, 0)
        # Create mask for this specific tile (no cache here; only used in single-tile path)
        tile_mask = self.create_tile_mask(image_width, image_height, x, y, tile_width, tile_height, mask_blur)
        # Use extraction position and size for blending
        result_image = self.blend_tile(result_image, processed_pil, 
                                     x1, y1, (ew, eh), tile_mask, padding)
        
        return result_image

    def process_and_blend_tile(
        self,
        tile_idx,
        tile_pos,
        upscaled_image,
        result_image,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        tile_width,
        tile_height,
        padding,
        mask_blur,
        image_width,
        image_height,
        force_uniform_tiles,
        tiled_decode,
        batch_idx: int = 0,
    ):
        """Public tile-processing API."""
        return self._process_and_blend_tile(
            tile_idx,
            tile_pos,
            upscaled_image,
            result_image,
            model,
            positive,
            negative,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            tile_width,
            tile_height,
            padding,
            mask_blur,
            image_width,
            image_height,
            force_uniform_tiles,
            tiled_decode,
            batch_idx=batch_idx,
        )

    def _process_single_tile(self, global_idx, num_tiles_per_image, upscaled_image, all_tiles,
                                  model, positive, negative, vae, seed, steps, cfg, sampler_name,
                                  scheduler, denoise, tiled_decode, tile_width, tile_height, padding,
                                  width, height, force_uniform_tiles, sliced_conditioning_cache):
        """Process a single tile."""
        # Calculate which image and tile this corresponds to
        batch_idx = global_idx // num_tiles_per_image
        tile_idx = global_idx % num_tiles_per_image
        
        # Skip if batch_idx is out of range
        if batch_idx >= upscaled_image.shape[0]:
            debug_log(f"Warning: Calculated batch_idx {batch_idx} exceeds batch size {upscaled_image.shape[0]}")
            return None
        
        # Get or create sliced conditioning for this batch index
        if batch_idx not in sliced_conditioning_cache:
            positive_sliced, negative_sliced = self._slice_conditioning(positive, negative, batch_idx)
            sliced_conditioning_cache[batch_idx] = (positive_sliced, negative_sliced)
        else:
            positive_sliced, negative_sliced = sliced_conditioning_cache[batch_idx]
        
        x, y = all_tiles[tile_idx]
        
        # Extract tile from the specific image in the batch
        tile_tensor, x1, y1, ew, eh = self.extract_tile_with_padding(
            upscaled_image[batch_idx:batch_idx+1], x, y, tile_width, tile_height, padding, force_uniform_tiles
        )
        
        # Process tile through SD with the exact seed (USDU parity)
        image_seed = seed
        processed_tile = self.process_tile(tile_tensor, model, positive_sliced, negative_sliced, vae,
                                         image_seed, steps, cfg, sampler_name,
                                         scheduler, denoise, tiled_decode, batch_idx=batch_idx,
                                         region=(x1, y1, x1 + ew, y1 + eh), image_size=(width, height))
        
        return {
            'tile': processed_tile,
            'global_idx': global_idx,
            'batch_idx': batch_idx,
            'tile_idx': tile_idx,
            'x': x1,
            'y': y1,
            'extracted_width': ew,
            'extracted_height': eh
        }
