import os
os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'auto'
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

import asyncio
import websockets
import json
import os
import uuid
from typing import Dict, Any, Literal
import argparse
from PIL import Image
import imageio
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils

class WebSocketServer:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.pipeline = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the pipeline
        self._init_pipeline()

    def _init_pipeline(self):
        """Initialize the TRELLIS pipeline"""
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline.cuda()

    async def send_progress(self, websocket: websockets.WebSocketServerProtocol, message: str, status: str, data: dict = None):
        """Send a progress message to the client"""
        msg = {
            "status": status,
            "message": message
        }
        if data:
            msg.update(data)
        await websocket.send(json.dumps(msg))

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        client_id = str(id(websocket))
        self.clients[client_id] = websocket
        print(f"Client connected: {client_id}")

        try:
            async for message in websocket:
                try:
                    # Parse the incoming message
                    data = json.loads(message)
                    
                    # Validate the message format
                    if "image_path" not in data:
                        await self.send_progress(websocket, "Missing required field: image_path", "error")
                        continue

                    # Get output type (default to model)
                    output_type = data.get("output_type", "model")  # "video" or "model"
                    if output_type not in ["video", "model"]:
                        await self.send_progress(websocket, "Invalid output_type. Must be 'video' or 'model'", "error")
                        continue

                    # Get parameters with defaults
                    params = data.get("params", {})
                    seed = params.get("seed", 1)
                    ss_guidance_strength = params.get("ss_guidance_strength", 7.5)
                    ss_sampling_steps = params.get("ss_sampling_steps", 12)
                    slat_guidance_strength = params.get("slat_guidance_strength", 3.0)
                    slat_sampling_steps = params.get("slat_sampling_steps", 12)
                    mesh_simplify = params.get("mesh_simplify", 0.95)
                    texture_size = params.get("texture_size", 1024)

                    # Check if file exists
                    if not os.path.exists(data["image_path"]):
                        await self.send_progress(websocket, "Image file not found", "error")
                        continue

                    # Generate unique ID for this conversion
                    conversion_id = str(uuid.uuid4())
                    
                    # Create paths for output files
                    model_path = os.path.join(self.output_dir, f"{conversion_id}.glb")
                    video_path = os.path.join(self.output_dir, f"{conversion_id}.mp4")

                    # Load the image
                    await self.send_progress(websocket, "Loading image...", "progress")
                    image = Image.open(data["image_path"])
                    
                    # Run the pipeline
                    await self.send_progress(websocket, "Running TRELLIS pipeline...", "progress")
                    outputs = self.pipeline.run(
                        image,
                        num_samples=1,
                        seed=seed,
                        sparse_structure_sampler_params={
                            "steps": ss_sampling_steps,
                            "cfg_strength": ss_guidance_strength,
                        },
                        slat_sampler_params={
                            "steps": slat_sampling_steps,
                            "cfg_strength": slat_guidance_strength,
                        },
                        formats=["gaussian", "mesh"] if output_type == "model" else ["gaussian"],
                        preprocess_image=True,
                    )
                    
                    # Generate video preview
                    await self.send_progress(websocket, "Generating video preview...", "progress")
                    video = render_utils.render_video(outputs['gaussian'][0])['color']
                    imageio.mimsave(video_path, video, fps=30)
                    await self.send_progress(websocket, "Video preview ready", "video_ready", {"video_path": video_path})
                    
                    # Generate GLB file if model output is requested
                    if output_type == "model":
                        await self.send_progress(websocket, "Generating 3D model...", "progress")
                        glb = postprocessing_utils.to_glb(
                            outputs['gaussian'][0],
                            outputs['mesh'][0],
                            simplify=mesh_simplify,
                            texture_size=texture_size,
                        )
                        glb.export(model_path)
                        await self.send_progress(websocket, "3D model ready", "model_ready", {"model_path": model_path})
                    
                    # Send completion message
                    completion_data = {"video_path": video_path}
                    if output_type == "model":
                        completion_data["model_path"] = model_path
                    
                    await self.send_progress(websocket, "Processing complete", "complete", completion_data)

                except json.JSONDecodeError:
                    await self.send_progress(websocket, "Invalid JSON format", "error")
                except Exception as e:
                    await self.send_progress(websocket, f"Processing error: {str(e)}", "error")

        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {client_id}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]

    async def start(self, host: str = "localhost", port: int = 8765):
        server = await websockets.serve(self.handle_client, host, port)
        print(f"WebSocket server started on ws://{host}:{port}")
        await server.wait_closed()

def parse_args():
    parser = argparse.ArgumentParser(description='TRELLIS WebSocket Server')
    parser.add_argument('--host', type=str, default='localhost',
                      help='Host to bind the WebSocket server to (default: localhost)')
    parser.add_argument('--port', type=int, default=8765,
                      help='Port to bind the WebSocket server to (default: 8765)')
    parser.add_argument('--output', type=str, default='./output',
                      help='Output directory for generated files (default: ./output)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    server = WebSocketServer(output_dir=args.output)
    
    # Start the WebSocket server
    asyncio.get_event_loop().run_until_complete(server.start(args.host, args.port)) 