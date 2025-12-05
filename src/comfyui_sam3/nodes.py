from inspect import cleandoc
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
from sam3.model.sam3_image_processor import Sam3Processor
import os

try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False

class SAM3Segmentation:
    """
    SAM3 Segmentation Node
    
    Performs image segmentation using SAM3 (Segment Anything Model 3) with text prompts.
    Takes an image and a text prompt, and outputs a visualization with segmentation masks,
    bounding boxes, and confidence scores.
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.use_video_model = None
    
    def _find_sam3_checkpoint(self):
        """Look for sam3.pt checkpoint in ComfyUI/models/sam3/ folder"""
        if not COMFYUI_AVAILABLE:
            return None
        
        # Try to find sam3.pt in models/sam3/ directory
        try:
            # Get the base models folder from ComfyUI
            model_paths = folder_paths.get_folder_paths("checkpoints")
            
            # Look for models/sam3/sam3.pt relative to the models folder
            for model_path in model_paths:
                # Go up one level from checkpoints to models, then into sam3
                models_base = os.path.dirname(model_path)
                sam3_checkpoint = os.path.join(models_base, "sam3", "sam3.pt")
                if os.path.exists(sam3_checkpoint):
                    print(f"SAM3: Found checkpoint at {sam3_checkpoint}")
                    return sam3_checkpoint
        except Exception as e:
            print(f"SAM3: Error searching for checkpoint: {e}")
        
        return None
    
    def load_model(self, use_video_model=False):
        """Force reload the model every time for video model to avoid state issues"""
        # Find checkpoint if available
        checkpoint_path = self._find_sam3_checkpoint()
        
        # Always reload video model to ensure clean state
        if use_video_model:
            print("Loading SAM3 video model...")
            self.unload_model()
            
            if checkpoint_path:
                self.model = build_sam3_video_model(checkpoint_path=checkpoint_path)
            else:
                self.model = build_sam3_video_model()
            # For video model, create processor using the detector's backbone
            self.processor = Sam3Processor(self.model.detector)
            print("SAM3 video model loaded successfully")
            self.use_video_model = use_video_model
        # Only reload image model if needed
        elif self.model is None or self.use_video_model != use_video_model:
            print("Loading SAM3 image model...")
            self.unload_model()
            
            if checkpoint_path:
                self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            else:
                self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            print("SAM3 image model loaded successfully")
            self.use_video_model = use_video_model

    def unload_model(self):
        """Completely unload SAM3 from memory (CPU and GPU)."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("SAM3: Model unloaded from memory")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to segment"}),
                "prompt": ("STRING", {
                    "multiline": False,
                    "default": "person",
                    "tooltip": "Text prompt describing what to segment (e.g., 'person', 'car', 'dog')"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum confidence score threshold (0.0 to 1.0)"
                }),
                "min_width_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "Minimum bounding box width in pixels"
                }),
                "min_height_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "Minimum bounding box height in pixels"
                }),
                "use_video_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use SAM3 video model instead of image model"
                }),
                "unload_after_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, unload SAM3 from GPU after this node finishes"
                }),
            },
            "optional": {
                "object_ids": ("STRING", {
                    "default": "",
                    "tooltip": "(Video model only) Comma-separated list of object IDs to track (e.g., '0,1,2'). Leave empty to track all objects."
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("segmented_image", "masks", "mask_combined")
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "segment"
    CATEGORY = "SAM3"

    def segment(self, image, prompt, threshold, min_width_pixels, min_height_pixels, use_video_model, unload_after_run=False, object_ids=""):
        """
        Perform segmentation on the input image using the text prompt
        
        Args:
            image: Input image tensor in ComfyUI format [B, H, W, C] with values in [0, 1]
            prompt: Text description of objects to segment
            threshold: Minimum confidence score threshold
            min_width_pixels: Minimum bounding box width in pixels
            min_height_pixels: Minimum bounding box height in pixels
            use_video_model: Whether to use SAM3 video model instead of image model
            object_ids: Comma-separated list of object IDs to track (video model only)
            
        Returns:
            Tuple of (segmented_image, masks, mask_combined)
        """
        # Load model if not already loaded
        self.load_model(use_video_model=use_video_model)
        
        if use_video_model:
            result = self._segment_with_video_model(
                image, prompt, threshold, min_width_pixels, min_height_pixels, object_ids
            )
        else:
            result = self._segment_with_image_model(
                image, prompt, threshold, min_width_pixels, min_height_pixels
            )

        # Unload if requested
        if unload_after_run:
            self.unload_model()

        return result

    def _segment_with_image_model(self, image, prompt, threshold, min_width_pixels, min_height_pixels):
        """Process images independently using the image model"""
        batch_size = image.shape[0]
        all_result_images = []
        all_masks = []
        all_combined_masks = []
        
        # Process each image in the batch
        for batch_idx in range(batch_size):
            # Convert ComfyUI image format [B, H, W, C] to PIL Image
            # ComfyUI images are in range [0, 1], convert to [0, 255]
            image_np = (image[batch_idx].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            
            # Set up inference state
            inference_state = self.processor.set_image(pil_image)
            
            # Run segmentation with text prompt
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)

            # Get results
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            # Filter by threshold, minimum width, and minimum height
            filtered_indices = []
            for i, (score, box) in enumerate(zip(scores, boxes)):
                if score >= threshold:
                    # Calculate box dimensions
                    x1, y1, x2, y2 = box
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width >= min_width_pixels and box_height >= min_height_pixels:
                        filtered_indices.append(i)
            
            masks = [masks[i] for i in filtered_indices]
            boxes = [boxes[i] for i in filtered_indices]
            scores = [scores[i] for i in filtered_indices]
            
            print(f"SAM3 [Image {batch_idx + 1}/{batch_size}] found {len(masks)} object(s) matching '{prompt}' with score >= {threshold}, width >= {min_width_pixels}px, height >= {min_height_pixels}px")

            
            # Create visualization
            result_img = pil_image.copy()
            
            # Process each detection
            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                # Convert mask to numpy if it's a tensor
                if torch.is_tensor(mask):
                    mask = mask.cpu().numpy()
                
                # Squeeze mask to remove extra dimensions
                mask = np.squeeze(mask)
                
                # Resize mask if needed to match image dimensions
                if mask.shape != (pil_image.size[1], pil_image.size[0]):
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_img = mask_img.resize(pil_image.size, Image.NEAREST)
                    mask = np.array(mask_img) > 0
                
                # Create colored overlay for mask
                mask_array = np.zeros((pil_image.size[1], pil_image.size[0], 4), dtype=np.uint8)
                
                # Apply mask with semi-transparent color
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                mask_color = colors[i % len(colors)]
                mask_array[mask] = (*mask_color, 100)  # Semi-transparent
                
                mask_overlay = Image.fromarray(mask_array, 'RGBA')
                result_img = Image.alpha_composite(result_img.convert('RGBA'), mask_overlay).convert('RGB')
                
                # Draw bounding box
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(result_img)
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=mask_color, width=3)
                
                # Draw score text
                score_text = f"{score:.2f}"
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # Draw text background
                text_bbox = draw.textbbox((x1, y1 - 25), score_text, font=font)
                draw.rectangle(text_bbox, fill=mask_color)
                draw.text((x1, y1 - 25), score_text, fill=(255, 255, 255), font=font)
            
            # Convert result back to ComfyUI format [H, W, C] with values in [0, 1]
            result_np = np.array(result_img).astype(np.float32) / 255.0
            all_result_images.append(result_np)
            
            # Prepare mask batch - ComfyUI masks are [B, H, W] with values in [0, 1]
            mask_list = []
            for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                # Convert mask to numpy if it's a tensor
                if torch.is_tensor(mask):
                    mask = mask.cpu().numpy()
                
                # Squeeze mask to remove extra dimensions
                mask = np.squeeze(mask)
                
                # Resize mask if needed to match image dimensions
                if mask.shape != (pil_image.size[1], pil_image.size[0]):
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_img = mask_img.resize(pil_image.size, Image.NEAREST)
                    mask = (np.array(mask_img) / 255.0).astype(np.float32)
                else:
                    mask = mask.astype(np.float32)
                
                mask_list.append(mask)
            
            # Store masks for this image
            if len(mask_list) > 0:
                all_masks.extend(mask_list)
                # Create combined mask by taking maximum across all masks
                combined_mask = np.maximum.reduce(mask_list)
                all_combined_masks.append(combined_mask)
            else:
                # Return empty mask if no detections for this image
                all_combined_masks.append(np.zeros((pil_image.size[1], pil_image.size[0]), dtype=np.float32))
        
        # Stack all results into batch tensors
        result_tensor = torch.from_numpy(np.stack(all_result_images, axis=0))
        
        # Stack masks into batch tensor [B, H, W]
        if len(all_masks) > 0:
            masks_tensor = torch.from_numpy(np.stack(all_masks, axis=0))
        else:
            # Return empty mask if no detections across all images
            masks_tensor = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
        
        # Stack combined masks [B, H, W]
        combined_mask_tensor = torch.from_numpy(np.stack(all_combined_masks, axis=0))
        
        return (result_tensor, masks_tensor, combined_mask_tensor)
    
    def _segment_with_video_model(self, image, prompt, threshold, min_width_pixels, min_height_pixels, object_ids=""):
        """Process images as video frames using the video model with temporal tracking"""
        # Parse object IDs if provided
        track_object_ids = None
        if object_ids and object_ids.strip():
            try:
                track_object_ids = [int(id.strip()) for id in object_ids.split(',') if id.strip()]
                print(f"SAM3 Video: Filtering for object IDs: {track_object_ids}")
            except ValueError:
                print(f"SAM3 Video: Warning - Invalid object_ids format '{object_ids}', tracking all objects")
                track_object_ids = None
        import tempfile
        import os
        import shutil
        
        batch_size = image.shape[0]
        
        # Convert batch of images to list of PIL images for video processing
        pil_images = []
        for batch_idx in range(batch_size):
            image_np = (image[batch_idx].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_images.append(pil_image)
        
        # Use a persistent temp directory that we clean manually
        temp_dir = tempfile.mkdtemp()
        try:
            
            # Save images to temporary directory for video model
            for idx, pil_img in enumerate(pil_images):
                pil_img.save(os.path.join(temp_dir, f"frame_{idx:05d}.jpg"))
            
            # Initialize inference state with video frames
            inference_state = self.model.init_state(resource_path=temp_dir)
            
            # Add text prompt on the first frame
            frame_idx, initial_output = self.model.add_prompt(
                inference_state=inference_state,
                frame_idx=0,
                text_str=prompt
            )
            
            print(f"SAM3 Video: Initial detection on frame 0 with prompt '{prompt}'")
            
            # Propagate across all frames to track objects
            video_segments = {}
            for frame_idx, outputs in self.model.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=batch_size,
                reverse=False
            ):
                video_segments[frame_idx] = outputs
            
            # Process results for each frame
            all_result_images = []
            all_masks = []
            all_combined_masks = []
            
            for batch_idx in range(batch_size):
                pil_image = pil_images[batch_idx]
                
                if batch_idx not in video_segments:
                    # No detections for this frame
                    print(f"SAM3 Video [Frame {batch_idx + 1}/{batch_size}] no objects detected")
                    result_np = np.array(pil_image).astype(np.float32) / 255.0
                    all_result_images.append(result_np)
                    all_combined_masks.append(np.zeros((pil_image.size[1], pil_image.size[0]), dtype=np.float32))
                    continue
                
                outputs = video_segments[batch_idx]
                out_obj_ids = outputs['out_obj_ids']
                out_binary_masks = outputs['out_binary_masks']
                out_probs = outputs['out_probs']
                out_boxes_xywh = outputs['out_boxes_xywh']
                
                # Filter by threshold, size, and object IDs
                filtered_indices = []
                for i, (obj_id, prob, box_xywh) in enumerate(zip(out_obj_ids, out_probs, out_boxes_xywh)):
                    # Filter by object ID if specified
                    if track_object_ids is not None and obj_id not in track_object_ids:
                        continue
                    
                    if prob >= threshold:
                        # box_xywh is in [x, y, w, h] format
                        box_width = box_xywh[2]
                        box_height = box_xywh[3]
                        if box_width >= min_width_pixels and box_height >= min_height_pixels:
                            filtered_indices.append(i)
                
                filtered_obj_ids = [out_obj_ids[i] for i in filtered_indices]
                filtered_masks = [out_binary_masks[i] for i in filtered_indices]
                filtered_probs = [out_probs[i] for i in filtered_indices]
                filtered_boxes = [out_boxes_xywh[i] for i in filtered_indices]
                
                print(f"SAM3 Video [Frame {batch_idx + 1}/{batch_size}] tracking {len(filtered_obj_ids)} object(s) with IDs {filtered_obj_ids}")
                
                # Create visualization
                result_img = pil_image.copy()
                
                # Convert to RGB for drawing
                result_img_rgb = result_img.convert('RGB')
                
                from PIL import ImageDraw, ImageFont
                
                for i, (obj_id, mask, prob, box_xywh) in enumerate(zip(filtered_obj_ids, filtered_masks, filtered_probs, filtered_boxes)):
                    # mask is already in video resolution and binary
                    mask = mask.astype(bool)
                    
                    # Create colored overlay
                    mask_array = np.zeros((pil_image.size[1], pil_image.size[0], 4), dtype=np.uint8)
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    mask_color = colors[obj_id % len(colors)]
                    mask_array[mask] = (*mask_color, 100)
                    
                    mask_overlay = Image.fromarray(mask_array, 'RGBA')
                    result_img_rgb = Image.alpha_composite(result_img_rgb.convert('RGBA'), mask_overlay).convert('RGB')
                
                # Draw bounding boxes and labels on the result image
                draw = ImageDraw.Draw(result_img_rgb)
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                for i, (obj_id, mask, prob, box_xywh) in enumerate(zip(filtered_obj_ids, filtered_masks, filtered_probs, filtered_boxes)):
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    mask_color = colors[obj_id % len(colors)]
                    
                    # box_xywh is in relative coordinates [x, y, w, h] - convert to pixels
                    img_w, img_h = pil_image.size
                    x, y, w, h = box_xywh
                    x1 = int(x * img_w)
                    y1 = int(y * img_h)
                    x2 = int((x + w) * img_w)
                    y2 = int((y + h) * img_h)
                    
                    # Draw bounding box with thick line
                    draw.rectangle([x1, y1, x2, y2], outline=mask_color, width=5)
                    
                    # Draw object ID and score label
                    label_text = f"ID:{obj_id} {prob:.2f}"
                    
                    # Position text above the box
                    text_y = max(y1 - 35, 5)
                    text_bbox = draw.textbbox((x1, text_y), label_text, font=font)
                    # Draw background rectangle for text
                    draw.rectangle(text_bbox, fill=mask_color)
                    draw.text((x1, text_y), label_text, fill=(255, 255, 255), font=font)
                
                result_img = result_img_rgb
                
                # Convert result back to ComfyUI format
                result_np = np.array(result_img).astype(np.float32) / 255.0
                all_result_images.append(result_np)
                
                # Prepare masks
                mask_list = []
                for mask in filtered_masks:
                    mask_float = mask.astype(np.float32)
                    mask_list.append(mask_float)
                
                if len(mask_list) > 0:
                    all_masks.extend(mask_list)
                    combined_mask = np.maximum.reduce(mask_list)
                    all_combined_masks.append(combined_mask)
                else:
                    all_combined_masks.append(np.zeros((pil_image.size[1], pil_image.size[0]), dtype=np.float32))
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            print("SAM3 Video: Cleaned up temporary files")
            # Reset inference state to clear all accumulated state for next run
            self.model.reset_state(inference_state)
            print("SAM3 Video: Inference state reset")
        
        # Stack results
        result_tensor = torch.from_numpy(np.stack(all_result_images, axis=0))
        
        if len(all_masks) > 0:
            masks_tensor = torch.from_numpy(np.stack(all_masks, axis=0))
        else:
            masks_tensor = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
        
        combined_mask_tensor = torch.from_numpy(np.stack(all_combined_masks, axis=0))
        
        return (result_tensor, masks_tensor, combined_mask_tensor)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SAM3Segmentation": SAM3Segmentation
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3Segmentation": "SAM3 Segmentation"
}
