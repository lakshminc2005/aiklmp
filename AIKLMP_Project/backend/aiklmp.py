from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import subprocess
import requests

app = Flask(__name__)

# ✅ CORS config — only allow your frontend domain
CORS(app, resources={r"/*": {"origins": "https://aiklmp.netlify.app"}})

# Safe directory creation
temp_path = "temp"
videos_path = os.path.join(temp_path, "videos")
audio_path = os.path.join(temp_path, "audio")

# If 'temp' exists as file, remove it
if os.path.exists(temp_path) and not os.path.isdir(temp_path):
    os.remove(temp_path)

# Make temp/videos and temp/audio
os.makedirs(videos_path, exist_ok=True)
os.makedirs(audio_path, exist_ok=True)

print("✅ temp/videos and temp/audio directories are ready.")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    video_id = str(uuid.uuid4())
    video_path = f"{videos_path}/{video_id}.mp4"
    audio_file_path = f"{audio_path}/{video_id}.wav"
    output_path = f"{videos_path}/{video_id}_final.mp4"

    # Step 1: Generate video using ModelScope API
    print("[INFO] Generating video from HuggingFace...")
    modelscope_url = "https://api-inference.huggingface.co/models/damo-vilab/modelscope-text-to-video-synthesis"
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        return jsonify({"error": "HF_TOKEN is missing in environment variables"}), 500

    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.post(modelscope_url, headers=headers, json={"inputs": prompt})

    print(f"[INFO] HuggingFace response status: {response.status_code}")
    if response.status_code != 200:
        return jsonify({"error": "Video generation failed", "details": response.text}), 500

    with open(video_path, "wb") as f:
        f.write(response.content)

    # Step 2: Generate audio using Bark
    print("[INFO] Generating audio using Bark...")
    tts_output = subprocess.run([
        "python3", "bark_infer.py",
        prompt,
        audio_file_path
    ], capture_output=True)

    if tts_output.returncode != 0:
        return jsonify({
            "error": "Audio generation failed",
            "details": tts_output.stderr.decode()
        }), 500

    # Step 3: Merge video + audio with FFmpeg
    print("[INFO] Merging video and audio with FFmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_file_path,
        "-c:v", "copy",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return send_file(output_path, mimetype="video/mp4")


# ✅ For Render compatibility (bind to 0.0.0.0 and dynamic port)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
