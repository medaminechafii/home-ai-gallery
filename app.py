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

#---CONFIGURATION---
MEDIA_FOLDER = "./media"
THUMBNAIL_FOLDER = "./thumbnails"
#Global Variables for the index
indexed_embeddings = None
indexed_filenames = []
MODEL_NAME = "clip-ViT-B-32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTENTIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
VIDEO_EXTENTIONS = [".mp4", ".avi", ".mov", ".mkv"]
DEFAULT_FRAMES_TO_EXTRACT = 10

INDEX_FILE_EMBEDDINGS = "indexed_embeddings.pt"
INDEX_FILE_FILENAMES = "indexed_filenames.json"

class searchRequest(BaseModel):
    query: str
    top_k: int = 50
    score_threshold: float = 0.2

THUMBNAIL_SIZE = (400,400)
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
    
app.mount("/static", StaticFiles(directory="static"),name="static")
@app.get("/",response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    with open("static/index.html", "r") as f:
        return f.read()
    
model = None
def index_all_media(media_folder: str, video_frames_to_extract: int = DEFAULT_FRAMES_TO_EXTRACT):
    """Indexes all media files in the specified folder."""
    global indexed_embeddings, indexed_filenames, model
    
    elements = os.listdir(media_folder)
    print(f"\n[INDEXING] Found {len(elements)} files in '{media_folder}' folder.")
    
    all_embeddings = []
    all_labels = []
    
    # Process files one at a time to avoid memory issues
    for idx, filename in enumerate(elements):
        full_path = os.path.join(media_folder, filename)
        
        if not os.path.isfile(full_path):
            continue
        
        print(f"[{idx+1}/{len(elements)}] Processing: {filename}")
        
        images_to_encode = []
        
        # Process videos
        if filename.endswith(tuple(VIDEO_EXTENTIONS)):
            try:
                frames_generator = extract_frames(full_path, video_frames_to_extract)
                for frame in frames_generator:
                    images_to_encode.append(frame)
            except Exception as e:
                print(f"  Error processing video: {e}")
                continue
        
        # Process images
        elif filename.endswith(tuple(IMAGE_EXTENTIONS)):
            try:
                images_to_encode.append(Image.open(full_path))
            except Exception as e:
                print(f"  Error processing image: {e}")
                continue
        
        else:
            continue  # Skip non-media files
        
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
        print(f"Total Vectors: {indexed_embeddings.shape[0]}")
        print(f"Vector Dimensions: {indexed_embeddings.shape[1]}")
        
        print("[PERSISTENCE] Saving index to disk...")
        torch.save(indexed_embeddings, INDEX_FILE_EMBEDDINGS)
        with open(INDEX_FILE_FILENAMES, "w") as f:
            json.dump(all_labels, f)
        print(f"Index saved to '{INDEX_FILE_EMBEDDINGS}' and '{INDEX_FILE_FILENAMES}'")
        print("-------------------------\n")
    else:
        print("No images or videos found in the folder.")
        indexed_embeddings = None
        indexed_filenames = []
    
    return None

@app.on_event("startup")
async def startup_event():
    """Load the model and index media files on startup."""
    global model, indexed_embeddings, indexed_filenames
    if os.path.exists(INDEX_FILE_EMBEDDINGS) and os.path.exists(INDEX_FILE_FILENAMES):
       try:
          print("Loading existing index from disk...")
          indexed_embeddings = torch.load(INDEX_FILE_EMBEDDINGS, map_location=DEVICE)
          with open(INDEX_FILE_FILENAMES, "r") as f:
              indexed_filenames = json.load(f)
          print("\n--- Index Loaded Successfully ---")
          print(f"Total Vectors: {indexed_embeddings.shape[0]}")
          print(f"Vector Dimensions: {indexed_embeddings.shape[1]}")
          print("Loading model...")
          model = SentenceTransformer(MODEL_NAME, device=DEVICE)
          print(f"Model '{MODEL_NAME}' loaded on {DEVICE}.")
          print("---------------------------------\n")
          return
       except Exception as e:
          print(f"[ERROR] Could not load saved index ({e}). Recalculating index...")
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Model '{MODEL_NAME}' loaded on {DEVICE}.")

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
    file_path = os.path.join(MEDIA_FOLDER, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename)

@app.get("/thumbnail/{filename}")
async def get_thumbnail_file(filename: str):
  """API endpoint to retrieve a thumbnail for a media file by filename."""
  thumb_path = os.path.join(THUMBNAIL_FOLDER, f"{filename}.jpg")
  if not os.path.isfile(thumb_path):
      raise HTTPException(status_code=404, detail="File not found")
  return FileResponse(path = thumb_path,media_type="image/jpeg",filename=filename)


