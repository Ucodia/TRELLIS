import os
os.environ['ATTN_BACKEND'] = 'flash-attn'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert an image to a 3D GLB file using TRELLIS')
    parser.add_argument('input_image', type=str, help='Path to the input image')
    parser.add_argument('--output', type=str, help='Path to save the output GLB file (optional, defaults to input filename with .glb extension)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for generation')
    parser.add_argument('--ss_guidance_strength', type=float, default=7.5, help='Guidance strength for sparse structure generation')
    parser.add_argument('--ss_sampling_steps', type=int, default=12, help='Sampling steps for sparse structure generation')
    parser.add_argument('--slat_guidance_strength', type=float, default=3.0, help='Guidance strength for structured latent generation')
    parser.add_argument('--slat_sampling_steps', type=int, default=12, help='Sampling steps for structured latent generation')
    parser.add_argument('--mesh_simplify', type=float, default=0.95, help='Mesh simplification factor')
    parser.add_argument('--texture_size', type=int, default=1024, help='Texture resolution')
    
    args = parser.parse_args()

    # If no output is specified, use the input filename with .glb extension
    if args.output is None:
        input_base = os.path.splitext(args.input_image)[0]
        args.output = f"{input_base}.glb"

    # Initialize the pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    # Load the input image
    image = Image.open(args.input_image)

    # Run the pipeline with the specified parameters
    outputs = pipeline.run(
        image,
        num_samples=1,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": args.ss_sampling_steps,
            "cfg_strength": args.ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": args.slat_sampling_steps,
            "cfg_strength": args.slat_guidance_strength,
        },
        formats=["gaussian", "mesh"],
        preprocess_image=True,
    )

    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{input_base}.mp4", video, fps=30)

    # Generate GLB file
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=args.mesh_simplify,
        texture_size=args.texture_size,
    )
    glb.export(args.output)
    print(f"GLB file: {args.output}")

if __name__ == "__main__":
    main()