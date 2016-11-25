#include "radv_private.h"
#include "sid.h"
#include "vk_format.h"

static VkFormat get_format_from_aspect_mask(VkImageAspectFlags aspectMask,
					    VkFormat format)
{
	if (aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT)
		format = vk_format_depth_only(format);
	else if (aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT)
		format = vk_format_stencil_only(format);
	return format;
}

static const struct radeon_surf_level *get_base_level_info(const struct radv_image *img,
							   VkImageAspectFlags aspectMask, int base_mip_level)
{
	if (aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT)
		return &img->surface.stencil_level[base_mip_level];
	return &img->surface.level[base_mip_level];
}

static void get_image_info(struct radv_cmd_buffer *cmd_buffer,
			   const struct radv_image *img,
			   const VkImageSubresourceLayers *subres,
			   uint64_t *va_p, uint32_t *bpp_p, uint32_t *pitch, uint32_t *slice_pitch)
{
	const struct radeon_surf_level *base_level = get_base_level_info(img, subres->aspectMask,
									 subres->mipLevel);
	VkFormat format = get_format_from_aspect_mask(subres->aspectMask, img->vk_format);
	uint32_t bpp = vk_format_get_blocksize(format);
	uint64_t va = cmd_buffer->device->ws->buffer_get_va(img->bo);

	va += img->offset;
	*pitch = base_level->nblk_x;
	*slice_pitch = base_level->slice_size / bpp;
	if (bpp_p)
		*bpp_p = bpp;
	*va_p = va;
}

/* L2L buffer->image + image->buffer */
static void
radv_cik_dma_copy_one_lin_to_lin(struct radv_cmd_buffer *cmd_buffer,
				 struct radv_buffer *buffer,
				 struct radv_image *image,
				 const VkBufferImageCopy *region,
				 bool buf2img)
{
	uint64_t buf_va, img_va;
	uint64_t src_va, dst_va;
	unsigned depth;
	unsigned zoffset;
	uint32_t bpp, pitch, slice_pitch;

	get_image_info(cmd_buffer, image, &region->imageSubresource, &img_va,
		       &bpp, &pitch, &slice_pitch);

	buf_va = cmd_buffer->device->ws->buffer_get_va(buffer->bo);
	buf_va += buffer->offset;
	buf_va += region->bufferOffset;

	if (image->type == VK_IMAGE_TYPE_3D) {
		depth = region->imageExtent.depth;
		zoffset = region->imageOffset.z;
	} else {
		depth = region->imageSubresource.layerCount;
		zoffset = region->imageSubresource.baseArrayLayer;
	}

	src_va = buf2img ? buf_va : img_va;
	dst_va = buf2img ? img_va : buf_va;

	radeon_emit(cmd_buffer->cs, CIK_SDMA_PACKET(CIK_SDMA_OPCODE_COPY,
						    CIK_SDMA_COPY_SUB_OPCODE_LINEAR_SUB_WINDOW, 0) |
		    (util_logbase2(bpp) << 29));
	radeon_emit(cmd_buffer->cs, src_va);
	radeon_emit(cmd_buffer->cs, src_va >> 32);
	radeon_emit(cmd_buffer->cs, 0);
	radeon_emit(cmd_buffer->cs, (region->bufferRowLength << 16));
	radeon_emit(cmd_buffer->cs, (region->bufferImageHeight));
	radeon_emit(cmd_buffer->cs, dst_va);
	radeon_emit(cmd_buffer->cs, dst_va >> 32);
	radeon_emit(cmd_buffer->cs, region->imageOffset.x | (region->imageOffset.y << 16));
	radeon_emit(cmd_buffer->cs, zoffset | (pitch << 16));
	radeon_emit(cmd_buffer->cs, slice_pitch);
	if (cmd_buffer->device->instance->physicalDevice.rad_info.chip_class == CIK) {
		radeon_emit(cmd_buffer->cs, region->imageExtent.width | (region->imageExtent.height << 16));
		radeon_emit(cmd_buffer->cs, depth);
	} else {
		radeon_emit(cmd_buffer->cs, (region->imageExtent.width -1) | ((region->imageExtent.height - 1) << 16));
		radeon_emit(cmd_buffer->cs, (depth - 1));
	}
}

/* L2T buffer->image + image->buffer */
static void
radv_cik_dma_copy_one_lin_to_tiled(struct radv_cmd_buffer *cmd_buffer,
				   struct radv_buffer *buffer,
				   struct radv_image *image,
				   const VkBufferImageCopy *region,
				   bool buf2img)
{
	uint64_t buf_va, img_va;
	unsigned depth;
	unsigned zoffset;
	unsigned pitch, slice_pitch, bpp;

	get_image_info(cmd_buffer, image, &region->imageSubresource, &img_va,
		       &bpp, &pitch, &slice_pitch);

	if (image->type == VK_IMAGE_TYPE_3D) {
		depth = region->imageExtent.depth;
		zoffset = region->imageOffset.z;
	} else {
		depth = region->imageSubresource.layerCount;
		zoffset = region->imageSubresource.baseArrayLayer;
	}

	buf_va = cmd_buffer->device->ws->buffer_get_va(buffer->bo);
	buf_va += buffer->offset;
	buf_va += region->bufferOffset;

	radeon_emit(cmd_buffer->cs, CIK_SDMA_PACKET(CIK_SDMA_OPCODE_COPY,
						    CIK_SDMA_COPY_SUB_OPCODE_TILED_SUB_WINDOW, 0) |
		    buf2img ? 0 : (1u << 31));
	radeon_emit(cmd_buffer->cs, img_va);
	radeon_emit(cmd_buffer->cs, img_va >> 32);
	radeon_emit(cmd_buffer->cs, region->imageOffset.x | (region->imageOffset.y << 16));
	radeon_emit(cmd_buffer->cs, zoffset | (pitch << 16));
	radeon_emit(cmd_buffer->cs, slice_pitch);
	radeon_emit(cmd_buffer->cs, 0/*tileinfo*/);
	radeon_emit(cmd_buffer->cs, buf_va);
	radeon_emit(cmd_buffer->cs, buf_va >> 32);
	radeon_emit(cmd_buffer->cs, 0/*x,y*/);
	radeon_emit(cmd_buffer->cs, (region->bufferRowLength << 16));
	radeon_emit(cmd_buffer->cs, region->bufferImageHeight);
	if (cmd_buffer->device->instance->physicalDevice.rad_info.chip_class == CIK) {
		radeon_emit(cmd_buffer->cs, region->imageExtent.width | (region->imageExtent.height << 16));
		radeon_emit(cmd_buffer->cs, depth);
	} else {
		radeon_emit(cmd_buffer->cs, (region->imageExtent.width -1) | ((region->imageExtent.height - 1) << 16));
		radeon_emit(cmd_buffer->cs, (depth - 1));
	}
}

/* T2T */

void radv_cik_dma_copy_buffer_to_image(struct radv_cmd_buffer *cmd_buffer,
				       struct radv_buffer *src_buffer,
				       struct radv_image *dest_image,
				       uint32_t region_count,
				       const VkBufferImageCopy *pRegions)
{
	uint32_t r;
	for (r = 0; r < region_count; r++) {
		const VkBufferImageCopy *region = &pRegions[r];
		if (dest_image->surface.level[region->imageSubresource.mipLevel].mode == RADEON_SURF_MODE_LINEAR_ALIGNED) {
			/* L -> L  */
			radv_cik_dma_copy_one_lin_to_lin(cmd_buffer, src_buffer, dest_image,
							 region, true);
		} else {
			/* L -> T */
			radv_cik_dma_copy_one_lin_to_tiled(cmd_buffer, src_buffer, dest_image,
							   region, true);
		}
	}
}

void radv_cik_dma_copy_image_to_buffer(struct radv_cmd_buffer *cmd_buffer,
				       struct radv_image *src_image,
				       struct radv_buffer *dest_buffer,
				       uint32_t region_count,
				       const VkBufferImageCopy *pRegions)
{
	uint32_t r;
	for (r = 0; r < region_count; r++) {
		const VkBufferImageCopy *region = &pRegions[r];

		if (src_image->surface.level[region->imageSubresource.mipLevel].mode == RADEON_SURF_MODE_LINEAR_ALIGNED) {
			/* L -> L */
			radv_cik_dma_copy_one_lin_to_lin(cmd_buffer, dest_buffer, src_image,
							 region, false);
		} else {
			/* L -> T */
			radv_cik_dma_copy_one_lin_to_tiled(cmd_buffer, dest_buffer, src_image,
							   region, false);
		}
	}
}

/* L2L buffer->image */
static void
radv_cik_dma_copy_one_image_lin_to_lin(struct radv_cmd_buffer *cmd_buffer,
				       struct radv_image *src_image,
				       struct radv_image *dst_image,
				       const VkImageCopy *region)

{
	uint64_t src_va, dst_va;
	unsigned src_pitch, src_slice_pitch, src_zoffset;
	unsigned dst_pitch, dst_slice_pitch, dst_zoffset;
	unsigned depth;
	unsigned bpp;

	get_image_info(cmd_buffer, src_image, &region->srcSubresource, &src_va,
		       &bpp, &src_pitch, &src_slice_pitch);
	get_image_info(cmd_buffer, dst_image, &region->dstSubresource, &dst_va,
		       NULL, &dst_pitch, &dst_slice_pitch);

	if (src_image->type == VK_IMAGE_TYPE_3D) {
		depth = region->extent.depth;
		src_zoffset = region->srcOffset.z;
	} else {
		depth = region->srcSubresource.layerCount;
		src_zoffset = region->srcSubresource.baseArrayLayer;
	}

	if (dst_image->type == VK_IMAGE_TYPE_3D) {
		dst_zoffset = region->dstOffset.z;
	} else {
		dst_zoffset = region->dstSubresource.baseArrayLayer;
	}

	radeon_emit(cmd_buffer->cs, CIK_SDMA_PACKET(CIK_SDMA_OPCODE_COPY,
						    CIK_SDMA_COPY_SUB_OPCODE_LINEAR_SUB_WINDOW, 0) |
		    (util_logbase2(bpp) << 29));
	radeon_emit(cmd_buffer->cs, src_va);
	radeon_emit(cmd_buffer->cs, src_va >> 32);
	radeon_emit(cmd_buffer->cs, region->srcOffset.x | (region->srcOffset.y << 16));
	radeon_emit(cmd_buffer->cs, src_zoffset | (src_pitch << 16));
	radeon_emit(cmd_buffer->cs, src_slice_pitch);
	radeon_emit(cmd_buffer->cs, dst_va);
	radeon_emit(cmd_buffer->cs, dst_va >> 32);
	radeon_emit(cmd_buffer->cs, region->dstOffset.x | (region->dstOffset.y << 16));
	radeon_emit(cmd_buffer->cs, dst_zoffset | (dst_pitch << 16));
	radeon_emit(cmd_buffer->cs, dst_slice_pitch);
	if (cmd_buffer->device->instance->physicalDevice.rad_info.chip_class == CIK) {
		radeon_emit(cmd_buffer->cs, region->extent.width | (region->extent.height << 16));
		radeon_emit(cmd_buffer->cs, depth);
	} else {
		radeon_emit(cmd_buffer->cs, (region->extent.width -1) | ((region->extent.height - 1) << 16));
		radeon_emit(cmd_buffer->cs, (depth - 1));
	}
}

void radv_cik_dma_copy_image(struct radv_cmd_buffer *cmd_buffer,
			     struct radv_image *src_image,
			     VkImageLayout src_image_layout,
			     struct radv_image *dest_image,
			     VkImageLayout dest_image_layout,
			     uint32_t region_count,
			     const VkImageCopy *pRegions)
{
	uint32_t r;
	for (r = 0; r < region_count; r++) {
		const VkImageCopy *region = &pRegions[r];
		bool src_is_linear = src_image->surface.level[region->srcSubresource.mipLevel].mode == RADEON_SURF_MODE_LINEAR_ALIGNED;
		bool dst_is_linear = dest_image->surface.level[region->dstSubresource.mipLevel].mode == RADEON_SURF_MODE_LINEAR_ALIGNED;

		/* X -> X */
		if (src_is_linear && dst_is_linear) {
			radv_cik_dma_copy_one_image_lin_to_lin(cmd_buffer,
							       src_image,
							       dest_image,
							       region);
			/* L -> L */
		} else if (!src_is_linear && dst_is_linear) {
			/* T -> L */
		} else if (src_is_linear && !dst_is_linear) {
			/* L -> T */
		} else {
			/* T -> T */
		}
	}

}

static void
radv_cik_sdma_do_copy_buffer_one(struct radv_cmd_buffer *cmd_buffer,
				 struct radv_buffer *src_buffer,
				 struct radv_buffer *dst_buffer,
				 const VkBufferCopy *region)
{
	unsigned ncopy, i;
	uint64_t src_va, dst_va;
	VkDeviceSize size = region->size;

	src_va = cmd_buffer->device->ws->buffer_get_va(src_buffer->bo);
	dst_va = cmd_buffer->device->ws->buffer_get_va(dst_buffer->bo);

	src_va += src_buffer->offset;
	dst_va += dst_buffer->offset;
	ncopy = DIV_ROUND_UP(region->size, CIK_SDMA_COPY_MAX_SIZE);

	src_va += region->srcOffset;
	dst_va += region->dstOffset;

	for (i = 0; i < ncopy; i++) {
		unsigned csize = MIN2(size, CIK_SDMA_COPY_MAX_SIZE);

		radeon_emit(cmd_buffer->cs, CIK_SDMA_PACKET(CIK_SDMA_OPCODE_COPY,
							    CIK_SDMA_COPY_SUB_OPCODE_LINEAR,
							    0));

		radeon_emit(cmd_buffer->cs, csize);
		radeon_emit(cmd_buffer->cs, 0);
		radeon_emit(cmd_buffer->cs, src_va);
		radeon_emit(cmd_buffer->cs, src_va >> 32);
		radeon_emit(cmd_buffer->cs, dst_va);
		radeon_emit(cmd_buffer->cs, dst_va >> 32);
		dst_va += csize;
		src_va += csize;
		size -= csize;
	}
}

void radv_cik_dma_copy_buffer(struct radv_cmd_buffer *cmd_buffer,
			      struct radv_buffer *src_buffer,
			      struct radv_buffer *dest_buffer,
			      uint32_t region_count,
			      const VkBufferCopy *pRegions)
{
	int r;

	for (r = 0; r < region_count; r++)
		radv_cik_sdma_do_copy_buffer_one(cmd_buffer,
						 src_buffer,
						 dest_buffer,
						 &pRegions[r]);
}
