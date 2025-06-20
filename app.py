import os
import tempfile
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

app = Flask(__name__)

# Load Whisper model with GPU (fallback to CPU if needed)
try:
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
except Exception as e:
    print(f"CUDA initialization failed: {e}. Falling back to CPU.")
    model = WhisperModel("large-v3", device="cpu", compute_type="float32")


def transcribe_audio(audio_path):
    """Run transcription and return text."""
    try:
        segments, info = model.transcribe(audio_path)
        transcription = " ".join([segment.text for segment in segments])
        os.unlink(audio_path)  # Delete temp file
        return transcription.strip()
    except Exception as e:
        raise Exception(f"Transcription error: {str(e)}")


@app.route("/health", methods=["GET"])
def health():
    """Check service status."""
    return jsonify({"status": "healthy"}), 200


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Accept audio file, save temp, and transcribe."""
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in request"}), 400
    audio_file = request.files["file"]

    # Save the uploaded audio to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_file.save(temp_file.name)

    try:
        transcription = transcribe_audio(temp_file.name)
        return jsonify({"transcription": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)

#STT model