import os
os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'auto'
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import imageio
import argparse
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils
from pydantic import BaseModel
from typing import Optional
import uuid

app = FastAPI(title="TRELLIS Image to 3D API")

# Initialize the pipeline once at startup
pipeline = None
output_dir = "./output"

class ConversionParams(BaseModel):
    seed: Optional[int] = 1
    ss_guidance_strength: Optional[float] = 7.5
    ss_sampling_steps: Optional[int] = 12
    slat_guidance_strength: Optional[float] = 3.0
    slat_sampling_steps: Optional[int] = 12
    mesh_simplify: Optional[float] = 0.95
    texture_size: Optional[int] = 1024

class ConversionRequest(BaseModel):
    image_path: str
    params: Optional[ConversionParams] = ConversionParams()

class ConversionResponse(BaseModel):
    model_path: str
    video_path: str

@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

@app.post("/convert", response_model=ConversionResponse)
async def convert_image_to_3d(request: ConversionRequest):
    """
    Convert an image to a 3D GLB file and generate a preview video.
    
    Parameters:
    - image_path: Path to the input image file
    - params: Optional conversion parameters with default values
    
    Returns:
    - JSON response containing paths to the generated model and video files
    """
    try:
        # Check if file exists
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=400, detail="Image file not found")
            
        # Generate unique ID for this conversion
        conversion_id = str(uuid.uuid4())
        
        # Create paths for output files
        model_path = os.path.join(output_dir, f"{conversion_id}.glb")
        video_path = os.path.join(output_dir, f"{conversion_id}.mp4")
        
        # Load the image
        image = Image.open(request.image_path)
        
        # Run the pipeline
        outputs = pipeline.run(
            image,
            num_samples=1,
            seed=request.params.seed,
            sparse_structure_sampler_params={
                "steps": request.params.ss_sampling_steps,
                "cfg_strength": request.params.ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": request.params.slat_sampling_steps,
                "cfg_strength": request.params.slat_guidance_strength,
            },
            formats=["gaussian", "mesh"],
            preprocess_image=True,
        )
        
        # Generate video preview
        video = render_utils.render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(video_path, video, fps=30)
        
        # Generate GLB file
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=request.params.mesh_simplify,
            texture_size=request.params.texture_size,
        )
        glb.export(model_path)
        
        # Return paths to the generated files
        return ConversionResponse(
            model_path=model_path,
            video_path=video_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def parse_args():
    parser = argparse.ArgumentParser(description='TRELLIS Image to 3D API Server')
    parser.add_argument('--output', type=str, default='./output',
                      help='Output directory for generated files (default: ./output)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to bind the server to (default: 8000)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port) 