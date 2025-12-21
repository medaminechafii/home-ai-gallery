# home gallery

I wanted to create my personal google drive by having a home server that has all my images and videos and being able to connect to it and download into my main device whatever media I want. 
I use tailscale for connecting my devices via VPN so that connecting to the web app is secure and personal of use. I will update this README whenver I have the type to explain exactly how to run the web app.

in this repo I will explain how to set this up so that you have your own home gallery too using your home server.
In my case, my home server is an archlinux machine, but I do think it works with any other machine.
![App Interface]<img width="1468" height="794" alt="Screenshot 2025-12-21 at 22 32 48" src="https://github.com/user-attachments/assets/b0e302ff-07df-4801-ab3b-36e804db7260" />


## Features

- **AI-Powered** - Uses OpenAI's CLIP model for intelligent visual understanding
-  **Fast Indexing** - Efficient media indexing with frame extraction for videos
-  **Similarity Scores** - Visual similarity percentages for each result
-  **Video Support** - Full support for video files with thumbnail generation
-  **Download Options** - Easy download functionality for all media

## Prerequisites

- Python 3.8+
- tailscale
- pip (Python package manager)

## Installation

### 1. Install System Dependencies
1.install tailscale on all you machines and login using the same account
**Arch Linux:**
```bash
sudo pacman -S tailscale
sudo systemctl enable --now tailscaled
```
for login :
```bash
sudo tailscale up
```

### 2. Clone the Repository

```bash
git clone https://github.com/medaminechafii/home-ai-gallery.git
cd home-ai-gallery
```

### 3. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up Media Directory

Create a `media` directory and add your images and videos:

```bash
mkdir media
# Add your images and videos to this directory
# Supports: jpg, jpeg, png, gif, webp, mp4, avi, mov, mkv
```

**Note:** The app supports symlinks, so you can link to existing media directories:
```bash
ln -s /path/to/your/media/* media/
```

## Usage

### 1. Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
you can use gpu for a little bit of acceleration if needed

The server will:
- Index all media files in the `media` directory
- Generate thumbnails for videos
- Start a FastAPI server on `http://localhost:8000` (FASTAPI by default uses port 8000)

### 2. Open the App

Navigate to `http://tailscaleIP:8000` in your web browser.

 replace tailscaleIP by your host IP address provided by tailscale
### 3. Search

Type a natural language query like:
- "person smiling at the beach"
- "cats sitting by the window"
- "sunset over mountains"
- "people celebrating"

The AI will find visually similar images and videos from your media collection.

## Project Structure

```
.
├── app.py       # FastAPI backend with CLIP model
├── static/
│   ├── script.js          # Frontend JavaScript
│   └── style.css          # Modern UI styling
├── index.html             # Main HTML interface
├── media/                 # Your images and videos
├── thumbnails/            # Auto-generated video thumbnails
├── indexed_embeddings.pt  #media embedding vectors for caching
├──indexed_filenames.json  #filenames database
└── README.md             # This file
```

## Configuration

### Change Media Directory

Edit the `MEDIA_FOLDER` variable in `app.py`:

```python
MEDIA_FOLDER = "path/to/your/media"
```


### Video Frame Extraction

By default, 10 frames are extracted from each video. To change this, edit `DEFAULT_FRAMES_TO_EXTRACT` in `app.py`:

```python
DEFAULT_FRAMES_TO_EXTRACT = 10  # Change this value
```

## Dependencies

- **FastAPI** - Web framework
- **PyTorch** - Deep learning framework
- **CLIP** - OpenAI's vision-language model
- **OpenCV** - Video processing
- **Pillow** - Image processing

## Points to improve

- **Download** -needs some changes for a faster download
- **Finetuning** -the clip model does make some mistakes so it needs some finetuning
- **interface** -the interface lacks a little bit of functionalities(like choosing the threshold and top_k in search)

## Troubleshooting

### FFmpeg Errors

If you see FFmpeg errors about corrupted videos:
1. Ensure FFmpeg is installed: `ffmpeg -version`
2. Re-encode problematic videos: `ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4`
3. The app will skip corrupted files and continue processing

### Slow Indexing

For large media collections:
- Indexing runs on startup
- Progress is shown in terminal
- Consider reducing video frame extraction count

### Memory Issues

For systems with limited RAM:
- Reduce batch size in image processing
- Process fewer video frames
- Use a smaller CLIP model variant

## License

MIT License - feel free to use this project for personal or commercial purposes.

## Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) - For the amazing vision-language model
- [Flask](https://flask.palletsprojects.com/) - For the web framework
- [FFmpeg](https://ffmpeg.org/) - For video processing capabilities

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.



