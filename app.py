import os
import torch
import json
import uvicorn
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse,FileResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
import hashlib
import time
from typing import Optional

#---CONFIGURATION---
class Config:
    def __init__(self):
        # Media and file paths
        self.MEDIA_FOLDER = os.getenv("MEDIA_FOLDER", "./media")
        self.THUMBNAIL_FOLDER = os.getenv("THUMBNAIL_FOLDER", "./thumbnails")
        
        # Model configuration
        self.MODEL_NAME = os.getenv("MODEL_NAME", "clip-ViT-B-32")
        self.DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        
        # File extensions
        self.IMAGE_EXTENSIONS = os.getenv("IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.bmp,.tiff,.heic").split(",")
        self.VIDEO_EXTENSIONS = os.getenv("VIDEO_EXTENSIONS", ".mp4,.avi,.mov,.mkv").split(",")
        
        # Indexing settings
        self.DEFAULT_FRAMES_TO_EXTRACT = int(os.getenv("DEFAULT_FRAMES_TO_EXTRACT", "10"))
        self.THUMBNAIL_SIZE = tuple(map(int, os.getenv("THUMBNAIL_SIZE", "400,400").split(",")))
        
        # Search defaults
        self.DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "50"))
        self.DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.2"))
        
        # Server settings
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8000"))
        
        # Index files
        self.INDEX_FILE_EMBEDDINGS = os.getenv("INDEX_FILE_EMBEDDINGS", "indexed_embeddings.pt")
        self.INDEX_FILE_FILENAMES = os.getenv("INDEX_FILE_FILENAMES", "indexed_filenames.json")
        self.INDEX_FILE_METADATA = os.getenv("INDEX_FILE_METADATA", "indexed_files_metadata.json")

# Initialize configuration
config = Config()

# Global variables for the index
indexed_embeddings = None
indexed_filenames = []
model = None

# Use config values (backward compatibility)
MEDIA_FOLDER = config.MEDIA_FOLDER
THUMBNAIL_FOLDER = config.THUMBNAIL_FOLDER
MODEL_NAME = config.MODEL_NAME
DEVICE = config.DEVICE
IMAGE_EXTENTIONS = config.IMAGE_EXTENSIONS
VIDEO_EXTENTIONS = config.VIDEO_EXTENSIONS
DEFAULT_FRAMES_TO_EXTRACT = config.DEFAULT_FRAMES_TO_EXTRACT
INDEX_FILE_EMBEDDINGS = config.INDEX_FILE_EMBEDDINGS
INDEX_FILE_FILENAMES = config.INDEX_FILE_FILENAMES
INDEX_FILE_METADATA = config.INDEX_FILE_METADATA
THUMBNAIL_SIZE = config.THUMBNAIL_SIZE

def validate_file_path(file_path: str, allowed_directory: str) -> bool:
    """
    Validate that a file path is safe to access.
    For actual files: ensures they're within the allowed directory
    For symlinks: allows symlinks within allowed directory but validates they don't point to dangerous locations
    """
    try:
        # Get the absolute path (resolves .. but not symlinks)
        abs_path = os.path.abspath(file_path)
        abs_allowed = os.path.abspath(allowed_directory)
        
        # First check: the file/link itself must be within allowed directory
        if not abs_path.startswith(abs_allowed):
            return False
        
        # If it's a symlink, check where it points
        if os.path.islink(abs_path):
            # Get the real path the symlink points to
            real_path = os.path.realpath(abs_path)
            
            # Block dangerous system directories
            dangerous_paths = [
                "/etc", "/usr", "/bin", "/sbin", "/boot", "/sys", "/proc",
                "/var/log", "/var/run", "/tmp", "/dev"
            ]
            
            for dangerous in dangerous_paths:
                if real_path.startswith(dangerous):
                    print(f"[SECURITY] Blocked symlink pointing to system directory: {real_path}")
                    return False
            
            # Allow user directories and mounted drives (your use case)
            # This allows symlinks to ~/Downloads, /mnt/usb-images, etc.
            allowed_target_prefixes = [
                "/home/", "/Users/", "/mnt/", "/media/", "/data/"
            ]
            
            # If it doesn't start with allowed prefixes, block it
            if not any(real_path.startswith(prefix) for prefix in allowed_target_prefixes):
                print(f"[SECURITY] Blocked symlink to untrusted location: {real_path}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[SECURITY] Error validating path {file_path}: {e}")
        return False

def load_file_metadata():
    """Load file modification times from metadata file."""
    if os.path.exists(INDEX_FILE_METADATA):
        try:
            with open(INDEX_FILE_METADATA, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load file metadata: {e}")
    return {}

def save_file_metadata(metadata):
    """Save file modification times to metadata file."""
    try:
        with open(INDEX_FILE_METADATA, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not save file metadata: {e}")

def get_file_mod_time(file_path):
    """Get file modification time, follows symlinks to get real file's mod time."""
    try:
        # Get the real path to follow symlinks
        real_path = os.path.realpath(file_path)
        return os.path.getmtime(real_path)
    except Exception:
        return 0

def is_safe_media_file(filename: str, media_folder: str) -> bool:
    """
    Check if a file is safe to process - validates extension and path security.
    """
    file_path = os.path.join(media_folder, filename)
    
    # Check if it's a file and exists
    if not os.path.isfile(file_path):
        return False
    
    # Validate path security (prevents symlink attacks)
    if not validate_file_path(file_path, media_folder):
        print(f"[SECURITY] Blocked potentially unsafe file: {filename}")
        return False
    
    # Check file extension
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENTIONS or ext in VIDEO_EXTENTIONS

class searchRequest(BaseModel):
    query: str
    top_k: int = config.DEFAULT_TOP_K
    score_threshold: float = config.DEFAULT_THRESHOLD

os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

#---END CONFIGURATION---

# =========================================================================
# 1. CORE UTILITY FUNCTIONS 
# =========================================================================

def generate_thumbnail(image_path):
  """Generate and save a thumbnail for the given image."""
  filename_hash = hashlib.md5(image_path.encode()).hexdigest()
  thumbnail_path = os.path.join(THUMBNAIL_FOLDER, f"{filename_hash}.jpg")
  if os.path.exists(thumbnail_path):
     return thumbnail_path
  try:
      with Image.open(image_path) as img:
        img.thumbnail(THUMBNAIL_SIZE,Image.Resampling.LANCZOS)
        img.convert("RGB").save(thumbnail_path, "JPEG",quality=85)
      return thumbnail_path
  except Exception as e:
      print(f"Error generating thumbnail for {image_path}: {e}")
      return None

def generate_video_thumbnail(video_path):
  """Generate and save a thumbnail for the given video."""
  filename_hash = hashlib.md5(video_path.encode()).hexdigest()
  thumbnail_path = os.path.join(THUMBNAIL_FOLDER, f"{filename_hash}.jpg")
  if os.path.exists(thumbnail_path):
     return thumbnail_path
  try:
      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
          print(f"ERROR: Cannot open video file {video_path} - file may be corrupted")
          cap.release()
          return None
      ret, frame = cap.read()
      cap.release()
      if ret:
          rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          pil_image = Image.fromarray(rgb_frame)
          pil_image.thumbnail(THUMBNAIL_SIZE,Image.Resampling.LANCZOS)
          pil_image.convert("RGB").save(thumbnail_path, "JPEG",quality=85)
      else:
        print(f"ERROR: Cannot read frame from {video_path} - file may be corrupted or incomplete")
      return thumbnail_path
  except Exception as e:
      print(f"Error generating thumbnail for {video_path}: {e}")
      return None


def extract_frames(video_path,desired_frames_per_media:int = DEFAULT_FRAMES_TO_EXTRACT):
  """
  Docstring for extract_frames
  
  :param video_path: Description
  :param desired_frames_per_media: Description
  :type desired_frames_per_media: int
  """
  """Extract frames from a video file at regular intervals."""
  
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    return
  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  if total_frames < desired_frames_per_media:
    desired_frames_per_media = total_frames
  frames = int(desired_frames_per_media)
  frames_to_skip = total_frames // desired_frames_per_media
  print(f"to get {frames} frames per video we need {frames_to_skip} frames to skip")
  frame_count = 0
  extracted_frames = 0
  for i in range (desired_frames_per_media):
    current_frame_index = i * frames_to_skip
    if current_frame_index > total_frames:
      print("finished extraction")
      break
    cap.set(cv2.CAP_PROP_POS_FRAMES,current_frame_index)
    ret,frame = cap.read()
    if ret:
      rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
      pil_image = Image.fromarray(rgb_frame)
      extracted_frames += 1
      yield pil_image
    else:
      print("Error : couldn't read frame")
      break
  print(f"{extracted_frames} frames extracted")
  cap.release()

# =========================================================================
# 2. INDEXING AND MODEL SETUP
# =========================================================================

app = FastAPI(title="Multimedia Semantic Search API")
app.mount("/static", StaticFiles(directory="static"),name="static")

@app.get("/",response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    with open("static/index.html", "r") as f:
        return f.read()
    
model = None
def index_all_media(media_folder: str, video_frames_to_extract: int = DEFAULT_FRAMES_TO_EXTRACT, force_reindex: bool = False):
    """Indexes all media files in the specified folder with incremental support."""
    global indexed_embeddings, indexed_filenames, model
    
    # Load existing file metadata
    existing_metadata = load_file_metadata()
    new_metadata = {}
    
    elements = os.listdir(media_folder)
    print(f"\n[INDEXING] Found {len(elements)} files in '{media_folder}' folder.")
    
    # Load existing index if not forcing reindex
    existing_embeddings = None
    existing_labels = []
    if not force_reindex and os.path.exists(INDEX_FILE_EMBEDDINGS) and os.path.exists(INDEX_FILE_FILENAMES):
        try:
            print("Loading existing index from disk...")
            existing_embeddings = torch.load(INDEX_FILE_EMBEDDINGS, map_location=DEVICE)
            with open(INDEX_FILE_FILENAMES, "r") as f:
                existing_labels = json.load(f)
            print(f"Loaded {len(existing_labels)} existing embeddings.")
        except Exception as e:
            print(f"[WARNING] Could not load existing index: {e}. Will rebuild.")
            existing_embeddings = None
            existing_labels = []
    
    all_embeddings = []
    all_labels = []
    files_processed = 0
    files_skipped = 0
    
    # Process files one at a time to avoid memory issues
    for idx, filename in enumerate(elements):
        full_path = os.path.join(media_folder, filename)
        
        # Security check - skip if not a safe media file
        if not is_safe_media_file(filename, media_folder):
            print(f"[{idx+1}/{len(elements)}] Skipping unsafe file: {filename}")
            continue
        
        # Get file modification time
        current_mod_time = get_file_mod_time(full_path)
        new_metadata[filename] = current_mod_time
        
        # Check if file needs reindexing
        if not force_reindex and filename in existing_metadata:
            if existing_metadata[filename] == current_mod_time:
                # File hasn't changed, use existing embedding if available
                if existing_embeddings is not None:
                    # Find existing embeddings for this file
                    for i, existing_filename in enumerate(existing_labels):
                        if existing_filename == filename:
                            all_embeddings.append(existing_embeddings[i].unsqueeze(0))
                            all_labels.append(filename)
                            files_skipped += 1
                            break
                    else:
                        # File in metadata but not in existing index, need to process
                        pass
                    continue
                else:
                    continue
        
        print(f"[{idx+1}/{len(elements)}] Processing: {filename}")
        files_processed += 1
        
        images_to_encode = []
        
        # Process videos
        if filename.lower().endswith(tuple(VIDEO_EXTENTIONS)):
            try:
                frames_generator = extract_frames(full_path, video_frames_to_extract)
                for frame in frames_generator:
                    images_to_encode.append(frame)
            except Exception as e:
                print(f"  Error processing video: {e}")
                continue
        
        # Process images
        elif filename.lower().endswith(tuple(IMAGE_EXTENTIONS)):
            try:
                images_to_encode.append(Image.open(full_path))
            except Exception as e:
                print(f"  Error processing image: {e}")
                continue
        
        else:
            continue  # This should not happen due to security check, but keep as fallback
        
        if not images_to_encode:
            continue
        
        # Encode this file's frames immediately
        print(f"  Encoding {len(images_to_encode)} frames...")
        
        try:
            file_embeddings = model.encode(
                images_to_encode,
                batch_size=8,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=DEVICE
            )
            
            # Move to CPU to save GPU memory
            all_embeddings.append(file_embeddings.cpu())
            all_labels.extend([filename] * len(images_to_encode))
            
            # Free memory immediately
            del images_to_encode
            del file_embeddings
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Garbage collection every 10 files
            if (idx + 1) % 10 == 0:
                import gc
                gc.collect()
                print(f"  Memory cleanup done ({idx+1} files processed)")
        
        except Exception as e:
            print(f"  Error encoding: {e}")
            del images_to_encode
            continue
    
    # Combine all embeddings
    if all_embeddings:
        print("\nCombining all embeddings...")
        indexed_embeddings = torch.cat(all_embeddings, dim=0).to(DEVICE)
        indexed_filenames = all_labels
        
        print("\n--- Indexing Complete ---")
        print(f"Files processed: {files_processed}")
        print(f"Files skipped (unchanged): {files_skipped}")
        print(f"Total Vectors: {indexed_embeddings.shape[0]}")
        print(f"Vector Dimensions: {indexed_embeddings.shape[1]}")
        
        print("[PERSISTENCE] Saving index to disk...")
        torch.save(indexed_embeddings, INDEX_FILE_EMBEDDINGS)
        with open(INDEX_FILE_FILENAMES, "w") as f:
            json.dump(all_labels, f)
        save_file_metadata(new_metadata)
        print(f"Index saved to '{INDEX_FILE_EMBEDDINGS}', '{INDEX_FILE_FILENAMES}', and '{INDEX_FILE_METADATA}'")
        print("-------------------------\n")
    else:
        if existing_embeddings is not None:
            print("No new or changed files found. Using existing index.")
            indexed_embeddings = existing_embeddings.to(DEVICE)
            indexed_filenames = existing_labels
        else:
            print("No images or videos found in the folder.")
            indexed_embeddings = None
            indexed_filenames = []
    
    return None

@app.on_event("startup")
async def startup_event():
    """Load the model and index media files on startup with incremental support."""
    global model, indexed_embeddings, indexed_filenames
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Model '{MODEL_NAME}' loaded on {DEVICE}.")
    
    # Use incremental indexing (only process new/changed files)
    index_all_media(MEDIA_FOLDER)

# =========================================================================
# 3. SEARCH API ENDPOINTS
# =========================================================================

def search_media(query:str,top_k:int = 50,score_threshold:float = 0.21):
  """Searches the indexed media files based on the text query."""
  global indexed_embeddings,indexed_filenames,model

  if indexed_embeddings is None or len(indexed_filenames) == 0:
    return {"query": query, "results": [], "message": "Index is empty. Run indexing first."}
  print(f"Searching for: '{query}'")
  query_embedding = model.encode(query,convert_to_tensor=True,device=DEVICE)
  k_search = 100
  hits = util.semantic_search(query_embedding,indexed_embeddings,top_k = k_search)[0]
  final_results = {}
  for hit in hits:

    index = hit["corpus_id"]
    score = float(hit["score"])
    filename = indexed_filenames[index]
    if score < score_threshold:
      continue
    is_video = filename.lower().endswith(tuple(VIDEO_EXTENTIONS))
    if is_video:
      thumb_path = generate_video_thumbnail(os.path.join(MEDIA_FOLDER,filename))
    else:
      thumb_path = generate_thumbnail(os.path.join(MEDIA_FOLDER,filename))
    full_path = os.path.join(MEDIA_FOLDER,filename)
    thumb_hash = hashlib.md5(full_path.encode()).hexdigest() if thumb_path else None
    if filename not in final_results or score > final_results[filename]["score"]:
     final_results[filename] = {
         "score":score,
         "filename":filename,
         "thumbnail":thumb_hash,
         "source": "Video" if is_video else "Image"
     }

     sorted_results = sorted(final_results.values(),key=lambda x:x["score"],reverse=True)
  return {"query": query, "total_matches": len(sorted_results), "results": sorted_results[:top_k]}
    
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and status."""
    try:
        # Check if model is loaded
        model_status = model is not None
        
        # Check index status
        index_status = indexed_embeddings is not None and len(indexed_filenames) > 0
        indexed_count = len(indexed_filenames) if indexed_filenames else 0
        
        # Check media folder accessibility
        media_accessible = os.path.exists(MEDIA_FOLDER) and os.path.isdir(MEDIA_FOLDER)
        
        # Check thumbnail folder
        thumbnail_accessible = os.path.exists(THUMBNAIL_FOLDER) and os.path.isdir(THUMBNAIL_FOLDER)
        
        # Count files in media folder
        media_files = 0
        if media_accessible:
            try:
                media_files = len([f for f in os.listdir(MEDIA_FOLDER) if os.path.isfile(os.path.join(MEDIA_FOLDER, f))])
            except Exception:
                media_files = -1  # Error counting files
        
        # Overall health status
        overall_healthy = model_status and index_status and media_accessible and thumbnail_accessible
        
        health_data = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": time.time(),
            "checks": {
                "model_loaded": model_status,
                "index_ready": index_status,
                "media_accessible": media_accessible,
                "thumbnail_accessible": thumbnail_accessible
            },
            "stats": {
                "indexed_files": indexed_count,
                "total_media_files": media_files,
                "device": DEVICE,
                "model_name": MODEL_NAME
            }
        }
        
        # Return appropriate HTTP status
        status_code = 200 if overall_healthy else 503
        
        return JSONResponse(content=health_data, status_code=status_code)
        
    except Exception as e:
        error_data = {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }
        return JSONResponse(content=error_data, status_code=500)

@app.get("/api/all-media")
async def get_all_media():
    """Get all indexed media files for browsing."""
    try:
        if indexed_filenames is None or len(indexed_filenames) == 0:
            return JSONResponse(
                content={"status": "error", "message": "No media indexed yet"},
                status_code=404
            )
        
        # Create media items with proper thumbnail generation
        all_media = []
        for filename in indexed_filenames:
            # Determine if it's a video
            ext = os.path.splitext(filename)[1].lower()
            is_video = ext in VIDEO_EXTENTIONS
            
            # Generate thumbnail like the search endpoint does
            full_path = os.path.join(MEDIA_FOLDER, filename)
            if is_video:
                thumb_path = generate_video_thumbnail(full_path)
            else:
                thumb_path = generate_thumbnail(full_path)
            
            # Use the same hash method as search endpoint
            thumb_hash = hashlib.md5(full_path.encode()).hexdigest() if thumb_path else None
            
            all_media.append({
                "filename": filename,
                "thumbnail": thumb_hash,
                "source": "Video" if is_video else "Image",
                "score": 1.0  # Default score for browsing
            })
        
        return JSONResponse(content={"media": all_media, "total": len(all_media)})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
async def force_reindex():
    """Force reindexing of all media files."""
    try:
        print("[API] Force reindex requested")
        index_all_media(MEDIA_FOLDER, force_reindex=True)
        return JSONResponse(content={"status": "success", "message": "Reindexing completed"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def api_search(request: searchRequest):
    """API endpoint to search media files based on a text query."""
    try:
        results = search_media(request.query, request.top_k, request.score_threshold)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/media/{filename}")
async def get_media_file(filename: str):
    """API endpoint to retrieve a media file by filename."""
    try:
        # Security check - validate filename to prevent path traversal
        if "/" in filename or "\\" in filename or filename.startswith(".."):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = os.path.join(MEDIA_FOLDER, filename)
        
        # Additional security check
        if not validate_file_path(file_path, MEDIA_FOLDER):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check file size to prevent serving extremely large files
        file_size = os.path.getsize(file_path)
        if file_size > 2 * 1024 * 1024 * 1024:  # 2GB limit
            raise HTTPException(status_code=413, detail="File too large")
        
        return FileResponse(path=file_path, filename=filename)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving media file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/thumbnail/{filename}")
async def get_thumbnail_file(filename: str):
    """API endpoint to retrieve a thumbnail for a media file by filename."""
    try:
        # Security check - validate filename to prevent path traversal
        if "/" in filename or "\\" in filename or filename.startswith(".."):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # The filename parameter should be the hash of the original media file
        thumb_path = os.path.join(THUMBNAIL_FOLDER, f"{filename}.jpg")
        
        # Security check - ensure thumbnail is within thumbnail directory
        if not validate_file_path(thumb_path, THUMBNAIL_FOLDER):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.isfile(thumb_path):
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        
        return FileResponse(path=thumb_path, media_type="image/jpeg", filename=f"{filename}.jpg")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving thumbnail {filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


