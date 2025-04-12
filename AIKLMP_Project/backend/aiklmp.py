# AI Video Generator - Python Web App using Flask, HuggingFace, and FFmpeg

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import subprocess
import requests


# Safe directory creation
temp_path = "temp"
videos_path = os.path.join(temp_path, "videos")

# Fix issue if 'temp' is a file
if os.path.exists(temp_path) and not os.path.isdir(temp_path):
    os.remove(temp_path)

# Now safely make the videos folder
os.makedirs(videos_path, exist_ok=True)

print("✅ temp/videos directory is ready.")


app = Flask(__name__)
CORS(app)

# Create directories to store temporary files
os.makedirs("temp/videos", exist_ok=True)
os.makedirs("temp/audio", exist_ok=True)

# Endpoint: Generate AI video from text
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    video_id = str(uuid.uuid4())
    video_path = f"temp/videos/{video_id}.mp4"
    audio_path = f"temp/audio/{video_id}.wav"
    output_path = f"temp/videos/{video_id}_final.mp4"

    # Step 1: Generate video from text using ModelScope API (HuggingFace hosted)
    print("[INFO] Generating video...")
    modelscope_url = "https://api-inference.huggingface.co/models/damo-vilab/modelscope-text-to-video-synthesis"
    hf_token = os.getenv("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.post(modelscope_url, headers=headers, json={"inputs": prompt})
    if response.status_code != 200:
        return jsonify({"error": "Video generation failed", "details": response.text}), 500

    # Save video
    with open(video_path, "wb") as f:
        f.write(response.content)

    # Step 2: Generate audio using Bark (open-source TTS)
    print("[INFO] Generating audio...")
    tts_output = subprocess.run([
        "python3", "bark_infer.py",
        prompt,
        audio_path
    ], capture_output=True)
    if tts_output.returncode != 0:
        return jsonify({"error": "Audio generation failed", "details": tts_output.stderr.decode()}), 500

    # Step 3: Combine video and audio with FFmpeg
    print("[INFO] Combining video and audio...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return send_file(output_path, mimetype="video/mp4")

if __name__ == "__main__":
    app.run(debug=True, port=5000)