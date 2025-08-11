# app.py (v4.5 - Proof-of-Concept: Limited to 10 scenes)

import requests
import tempfile
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pytesseract
from flask import Flask, request, jsonify

# Import PySceneDetect components
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector

# --- CONFIGURATION CONSTANTS ---
COLOR_PALETTE_SIZE = 5
OCR_CONFIDENCE_THRESHOLD = 65
MOTION_THRESHOLD = 1.0

# --- HELPER FUNCTIONS ---

def analyze_colors(frame, k=COLOR_PALETTE_SIZE):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixels = cv2.resize(frame_rgb, (100, 100)).reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(pixels)
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
        return hex_colors
    except Exception: return []

def analyze_shot_type(frame, face_cascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
    if len(faces) == 0: return "wide / landscape"
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    (x, y, w, h) = largest_face
    face_area, frame_area = w * h, frame.shape[0] * frame.shape[1]
    return "close-up" if (face_area / frame_area) > 0.1 else "medium shot"

def analyze_text_overlays(frame):
    try:
        ocr_data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
        text_overlays = []
        for i in range(len(ocr_data['level'])):
            if int(ocr_data['conf'][i]) > OCR_CONFIDENCE_THRESHOLD and ocr_data['text'][i].strip():
                text = ocr_data['text'][i]
                (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                text_overlays.append({"text": text, "position": {"x": x, "y": y, "width": w, "height": h}})
        return text_overlays
    except Exception: return []

def analyze_motion(prev_frame, curr_frame):
    if prev_frame is None or curr_frame is None: return "static"
    prev_gray, curr_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return "motion" if np.mean(magnitude) > MOTION_THRESHOLD else "static"

# --- FLASK APPLICATION ---

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/analyze', methods=['POST'])
def analyze_video_endpoint():
    print("Received a request to /analyze")
    data = request.get_json()
    if not data or 'video_url' not in data:
        return jsonify({"success": False, "error": "Missing 'video_url' in request body"}), 400

    video_url = data['video_url']

    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video_file:
        try:
            print(f"Downloading video: {video_url}")
            with requests.get(video_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192): temp_video_file.write(chunk)
            
            video_path = temp_video_file.name
            
            print("Detecting scenes...")
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector())
            scene_manager.add_detector(ThresholdDetector())
            scene_manager.detect_scenes(video=video)
            scene_list = scene_manager.get_scene_list()
            print(f"Found {len(scene_list)} scenes.")

            print("Analyzing each scene for details...")
            analyzed_scenes, prev_frame = [], None
            for i, scene in enumerate(scene_list):
                # ---- THE NEW CHANGE TO PREVENT TIMEOUTS ON FREE TIER ----
                if i >= 10:
                    print("Limiting analysis to first 10 scenes for free tier.")
                    break # Stop the loop
                # -----------------------------------------------------------

                start_time, end_time = scene[0], scene[1]
                
                middle_frame_num = int((start_time.get_frames() + end_time.get_frames()) / 2)
                video.seek(middle_frame_num)
                middle_frame = video.read()
                
                video.seek(start_time.get_frames())
                curr_frame = video.read()

                scene_data = {
                    "scene_number": i + 1,
                    "duration_seconds": (end_time - start_time).get_seconds(),
                    "start_timecode": start_time.get_timecode(),
                    "dominant_colors": [], "shot_type": "unknown", "text_overlays": [], "camera_motion": "static"
                }
                
                if middle_frame is not None:
                    scene_data["dominant_colors"] = analyze_colors(middle_frame)
                    scene_data["shot_type"] = analyze_shot_type(middle_frame, face_cascade)
                    scene_data["text_overlays"] = analyze_text_overlays(middle_frame)

                if curr_frame is not None:
                    scene_data["camera_motion"] = analyze_motion(prev_frame, curr_frame)

                analyzed_scenes.append(scene_data)
                prev_frame = curr_frame

            final_response = {"success": True, "video_profile": {"scenes": analyzed_scenes}}
            print("Analysis complete.")
            return jsonify(final_response)

        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback; traceback.print_exc()
            return jsonify({"success": False, "error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)