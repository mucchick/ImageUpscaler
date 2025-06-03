
import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from config import Config
from models import ESRGANGenerator


class ESRGANInference:
    def __init__(self, model_path, device=None):
        """Initialize ESRGAN inference"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.transform = transforms.ToTensor()

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)

        # Create generator
        self.generator = ESRGANGenerator(
            num_rrdb=Config.NUM_RRDB_BLOCKS,
            scale=Config.SCALE_FACTOR
        ).to(device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()

        print(f"Model loaded successfully on {device}")

    def process_image(self, image_path, output_path=None, tile_size=512, tile_overlap=32):
        """
        Process a single image with optional tiling for large images

        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        print(f"Processing image: {image_path} (Size: {image.size})")

        # Convert to tensor
        lr_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Check if image is too large for GPU memory
            _, _, h, w = lr_tensor.shape

            if h > tile_size or w > tile_size:
                print(f"Large image detected, using tiled processing...")
                sr_tensor = self._process_with_tiles(lr_tensor, tile_size, tile_overlap)
            else:
                sr_tensor = self.generator(lr_tensor).clamp(0, 1)

        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(image_path)
            output_path = os.path.join(output_dir, f"{base_name}_sr.png")

        # Save result
        save_image(sr_tensor[0], output_path)
        print(f"Super-resolved image saved to: {output_path}")

        return output_path

    def _process_with_tiles(self, lr_tensor, tile_size, overlap):
        """Process large images using tiled approach"""
        _, _, h, w = lr_tensor.shape
        scale = Config.SCALE_FACTOR

        # Calculate output dimensions
        output_h = h * scale
        output_w = w * scale

        # Initialize output tensor
        sr_tensor = torch.zeros(1, 3, output_h, output_w, device=self.device)

        # Calculate tile positions
        h_tiles = (h - overlap) // (tile_size - overlap) + 1
        w_tiles = (w - overlap) // (tile_size - overlap) + 1

        for i in range(h_tiles):
            for j in range(w_tiles):
                # Calculate tile boundaries
                start_h = i * (tile_size - overlap)
                end_h = min(start_h + tile_size, h)
                start_w = j * (tile_size - overlap)
                end_w = min(start_w + tile_size, w)

                # Extract tile
                tile = lr_tensor[:, :, start_h:end_h, start_w:end_w]

                # Process tile
                sr_tile = self.generator(tile).clamp(0, 1)

                # Calculate output positions
                out_start_h = start_h * scale
                out_end_h = end_h * scale
                out_start_w = start_w * scale
                out_end_w = end_w * scale

                # Handle overlap blending
                if i > 0 or j > 0:
                    # Simple averaging for overlapping regions
                    sr_tensor[:, :, out_start_h:out_end_h, out_start_w:out_end_w] = \
                        (sr_tensor[:, :, out_start_h:out_end_h, out_start_w:out_end_w] + sr_tile) / 2
                else:
                    sr_tensor[:, :, out_start_h:out_end_h, out_start_w:out_end_w] = sr_tile

        return sr_tensor

    def process_directory(self, input_dir, output_dir=None, supported_formats=None):
        """
        Process all images in a directory
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        if output_dir is None:
            output_dir = os.path.join(input_dir, 'super_resolved')

        os.makedirs(output_dir, exist_ok=True)

        # Find all images
        image_files = []
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                image_files.append(file)

        if not image_files:
            print(f"No supported images found in {input_dir}")
            return

        print(f"Found {len(image_files)} images to process")

        # Process each image
        for i, image_file in enumerate(image_files, 1):
            input_path = os.path.join(input_dir, image_file)

            # Generate output filename
            base_name = os.path.splitext(image_file)[0]
            output_filename = f"{base_name}_sr.png"
            output_path = os.path.join(output_dir, output_filename)

            print(f"[{i}/{len(image_files)}] Processing {image_file}...")

            try:
                self.process_image(input_path, output_path)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        print(f"Batch processing completed. Results saved to {output_dir}")

    def create_comparison(self, image_path, output_path=None):
        """Create a side-by-side comparison image"""
        # Process image
        sr_path = self.process_image(image_path)

        # Load original and super-resolved images
        original = Image.open(image_path).convert('RGB')
        super_resolved = Image.open(sr_path).convert('RGB')

        # Create comparison
        comparison_width = original.width + super_resolved.width
        comparison_height = max(original.height, super_resolved.height)

        comparison = Image.new('RGB', (comparison_width, comparison_height), (255, 255, 255))
        comparison.paste(original, (0, 0))
        comparison.paste(super_resolved, (original.width, 0))

        # Save comparison
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(image_path)
            output_path = os.path.join(output_dir, f"{base_name}_comparison.png")

        comparison.save(output_path)
        print(f"Comparison saved to: {output_path}")

        return output_path


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='ESRGAN Inference')
    parser.add_argument('--model', required=True, type=str,
                        help='Path to trained ESRGAN model')
    parser.add_argument('--input', required=True, type=str,
                        help='Input image or directory path')
    parser.add_argument('--output', type=str,
                        help='Output path (optional)')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Tile size for processing large images')
    parser.add_argument('--tile-overlap', type=int, default=32,
                        help='Overlap between tiles')
    parser.add_argument('--comparison', action='store_true',
                        help='Create side-by-side comparison image')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to use for inference')

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("ESRGAN Inference")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Device: {device}")
    print("=" * 50)

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        return

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input {args.input} not found!")
        return

    # Initialize inference
    try:
        esrgan = ESRGANInference(args.model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process input
    try:
        if os.path.isfile(args.input):
            # Single image
            if args.comparison:
                esrgan.create_comparison(args.input, args.output)
            else:
                esrgan.process_image(args.input, args.output,
                                     args.tile_size, args.tile_overlap)
        elif os.path.isdir(args.input):
            # Directory
            esrgan.process_directory(args.input, args.output)
        else:
            print(f"Error: {args.input} is neither a file nor a directory!")

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main()