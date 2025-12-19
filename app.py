import os
import torch
import json
import uvicorn
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse,FileResponse,FileResponse
from sentence_transformers import SentenceTransformer, util  
from fastapi.staticfiles import StaticFiles 
from pydantic import BaseModel


#---CONFIGURATION---
MEDIA_FOLDER = "./media"
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

#---END CONFIGURATION---

# =========================================================================
# 1. CORE UTILITY FUNCTIONS (Your Optimized Code)
# =========================================================================

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
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")
model = None
def index_all_media(media_folder:str,video_frames_to_extract:int = DEFAULT_FRAMES_TO_EXTRACT):
   
  """Indexes all media files in the specified folder."""

  """Docstring for index_all_media
    :param media_folder: Description
    :param video_frames_to_extract: Description
    :type video_frames_to_extract: int
    """
  global indexed_embeddings,indexed_filenames,model
  elements = os.listdir(media_folder)
  print(f"\n[INDEXING] Found {len(elements)} files in '{media_folder}' folder.")
  images_to_encode = []
  label = []
  for filename in elements:

    full_path = os.path.join(media_folder, filename)
    if not os.path.isfile(full_path):
            continue
    
    if filename.endswith(tuple(VIDEO_EXTENTIONS)):
      try:
        frames_generator = extract_frames(full_path,video_frames_to_extract)

        for i,frame in enumerate(frames_generator):
          images_to_encode.append(frame)
          label.append(filename)
      except Exception as e:
        print(f"Error processing video {filename}: {e}")
        continue

    if filename.endswith(tuple(IMAGE_EXTENTIONS)):
      try:
        images_to_encode.append(Image.open(full_path))
        label.append(filename)
      except Exception as e:
        print(f"Error processing image {filename}: {e}")
        continue

  if not images_to_encode:
    print("No images or videos found in the folder.")
    indexed_embeddings = None
    indexed_filenames = []
    return None
  
  indexed_filenames = label
  print(f"{len(images_to_encode)} images or videos found.")
  print(f"Encoding {len(images_to_encode)} total frames/images on {DEVICE}...")
  print("Encoding images...")

  indexed_embeddings = model.encode(images_to_encode,batch_size=64,
                                    convert_to_tensor=True,
                                    show_progress_bar=True,
                                    device = DEVICE)
  
  print("\n--- Indexing Complete ---")
  print(f"Total Vectors: {indexed_embeddings.shape[0]}")
  print(f"Vector Dimensions: {indexed_embeddings.shape[1]}")
  print("[PERSISTENCE] Saving index to disk...")
  torch.save(indexed_embeddings, INDEX_FILE_EMBEDDINGS)
  with open(INDEX_FILE_FILENAMES, "w") as f:
    json.dump(label, f)
  print(f"Index saved to '{INDEX_FILE_EMBEDDINGS}' and '{INDEX_FILE_FILENAMES}'")
  print("-------------------------\n")
  return None
@app.on_event("startup")
async def startup_event():
    """Load the model and index media files on startup."""
    global model
    if os.path.exists(INDEX_FILE_EMBEDDINGS) and os.path.exists(INDEX_FILE_FILENAMES):
       try:
          print("Loading existing index from disk...")
          indexed_embeddings = torch.load(INDEX_FILE_EMBEDDINGS, map_location=DEVICE)
          with open(INDEX_FILE_FILENAMES, "r") as f:
              indexed_filenames = json.load(f)
          print("\n--- Index Loaded Successfully ---")
          print(f"Total Vectors: {indexed_embeddings.shape[0]}")
          print(f"Vector Dimensions: {indexed_embeddings.shape[1]}")
          print("---------------------------------\n")
       except Exception as e:
          print(f"[ERROR] Could not load saved index ({e}). Recalculating index...")
          pass
   
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
    if filename not in final_results or score > final_results[filename]["score"]:
     final_results[filename] = {
         "score":score,
         "filename":filename,
         "source": "Video Frame" if filename.lower().endswith(tuple(VIDEO_EXTENTIONS)) else "Image"
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


