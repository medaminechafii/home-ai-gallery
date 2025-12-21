# AI-Powered Media Search

I wanted to create my personal google drive by having a home server that has all my images and videos and being able to connect to it and download into my main device whatever media I want. 
I use tailscale for connecting my devices via VPN so that connecting to the web app is secure and personal of use. I will update this README whenver I have the type to explain exactly how to run the web app.


![App Interface]

## Features

- 🔍 **Semantic Search** - Search images and videos using natural language descriptions
- 🤖 **AI-Powered** - Uses OpenAI's CLIP model for intelligent visual understanding
- 🎨 **Modern UI** - Beautiful gradient backgrounds with glassmorphism effects
- ⚡ **Fast Indexing** - Efficient media indexing with frame extraction for videos
- 📊 **Similarity Scores** - Visual similarity percentages for each result
- 🎥 **Video Support** - Full support for video files with thumbnail generation
- 📥 **Download Options** - Easy download functionality for all media
- 🔄 **Real-time Search** - Instant results as you type

## Prerequisites

- Python 3.8+
- tailscale
- pip (Python package manager)

## Installation

### 1. Install System Dependencies

**Arch Linux:**
```bash
sudo pacman -S ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-media-search.git
cd ai-media-search
```

### 3. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Python Dependencies

```bash
pip install flask torch torchvision pillow opencv-python clip
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
python search-request.py
```

The server will:
- Index all media files in the `media` directory
- Generate thumbnails for videos
- Start a Flask server on `http://localhost:5000`

### 2. Open the App

Navigate to `http://localhost:5000` in your web browser.

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
├── search-request.py       # Flask backend with CLIP model
├── static/
│   ├── script.js          # Frontend JavaScript
│   └── style.css          # Modern UI styling
├── index.html             # Main HTML interface
├── media/                 # Your images and videos
├── thumbnails/            # Auto-generated video thumbnails
└── README.md             # This file
```

## Configuration

### Change Media Directory

Edit the `MEDIA_DIR` variable in `search-request.py`:

```python
MEDIA_DIR = "path/to/your/media"
```

### Change Port

Edit the last line in `search-request.py`:

```python
app.run(debug=True, port=5000)  # Change port here
```

### Video Frame Extraction

By default, 5 frames are extracted from each video. To change this, edit `extract_frames()` in `search-request.py`:

```python
NUM_FRAMES = 5  # Change this value
```

## Dependencies

- **Flask** - Web framework
- **PyTorch** - Deep learning framework
- **CLIP** - OpenAI's vision-language model
- **OpenCV** - Video processing
- **Pillow** - Image processing

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

## Screenshots

Add your screenshots here:
- Main search interface
- Search results
- Video playback

---

