# Make ABSOLUTE SURE to read the instructions provided.
# Note: This program was made in Visual Studio Code, so I don't know if it'll work elsewhere.

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")
import difflib
import requests
import json
import speech_recognition as sr
import random
import os
import re
import string
import atexit
import cv2
from ultralytics import YOLO
import pytesseract
from datetime import datetime
import face_recognition
import whisper

# ==============================
# LOAD CONFIG
# ==============================
with open("config.json", "r") as f:
    CONFIG = json.load(f)

WAKE_WORDS = CONFIG.get("wake_words", [])
WAKE_PREFIXES = CONFIG.get("wake_prefixes", ["", "ok ", "hay ", "hey "])
USER_NAME = CONFIG["user_name"]
PERSONAL_GREETINGS = CONFIG["personal_greetings"]
FAREWELLS = CONFIG["farewells"]
HOLIDAYS = CONFIG["holiday_greetings"]

pytesseract.pytesseract.tesseract_cmd = CONFIG["tesseract_path"]
MEMORY_FILE = CONFIG["memory_file"]
SYSTEM_PROMPT_FILE = CONFIG["system_prompt_file"]
MICROPHONE_INDEX = CONFIG["microphone_index"]
yolo_model = YOLO(CONFIG["yolo_model"])

# ==============================
# GLOBAL VARIABLES
# ==============================
known_face_encodings = []
known_face_names = []
current_speaker = "Unknown"
temp_memory_cache = set()

whisper_model = whisper.load_model("large")  # You can change in config if needed

# ==============================
# LOAD KNOWN FACES
# ==============================
def load_known_faces(folder="Faces"):
    global known_face_encodings, known_face_names
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[INFO] Created '{folder}' folder. Place your known face images inside (e.g., {USER_NAME}.jpg).")
        return

    for file_name in os.listdir(folder):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(file_name)[0]
            try:
                image = face_recognition.load_image_file(os.path.join(folder, file_name))
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                else:
                    print(f"[WARNING] No face detected in {file_name}")
            except Exception as e:
                print(f"[ERROR] Could not process {file_name}: {e}")

load_known_faces()

# ==============================
# HYBRID SPEECH RECOGNITION
# ==============================
def hybrid_recognize_speech(prompt="", device_index=MICROPHONE_INDEX, timeout=None, phrase_time_limit=10):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        mic = sr.Microphone(device_index=device_index) if device_index is not None else sr.Microphone()
        with mic as source:
            if prompt:
                print(prompt)
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            audio_data = audio.get_wav_data()
    except:
        return None

    google_text, whisper_text = "", ""

    # Google STT
    try:
        google_text = recognizer.recognize_google(audio).lower()
    except:
        google_text = ""

    # Whisper STT
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data)
        result = whisper_model.transcribe("temp_audio.wav", fp16=False)
        whisper_text = result["text"].lower().strip()
    except:
        whisper_text = ""

    # Combine Results
    if not whisper_text and not google_text:
        return None
    if whisper_text and not google_text:
        return whisper_text
    if google_text and not whisper_text:
        return google_text

    whisper_words = whisper_text.split()
    google_words = google_text.split()

    merged_words = []
    for i, word in enumerate(whisper_words):
        if i < len(google_words):
            g_word = google_words[i]
            similarity = difflib.SequenceMatcher(None, word, g_word).ratio()
            merged_words.append(word if similarity > 0.7 else word)
        else:
            merged_words.append(word)

    return " ".join(merged_words)

# ==============================
# MEMORY FUNCTIONS
# ==============================
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def auto_update_memory(user_input, ai_reply):
    global temp_memory_cache
    memory = load_memory()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def entry_exists(role, content):
        return any(isinstance(m, dict) and m.get("role") == role and m.get("content") == content for m in memory)

    new_entries = []
    cleaned_user_input = user_input.strip()
    if 5 < len(cleaned_user_input) < 300 and not entry_exists("user", cleaned_user_input):
        new_entries.append({"role": "user", "content": cleaned_user_input, "time": now})
        temp_memory_cache.add(cleaned_user_input)

    cleaned_ai_reply = ai_reply.strip()
    if 5 < len(cleaned_ai_reply) < 300 and not entry_exists("ai", cleaned_ai_reply):
        new_entries.append({"role": "ai", "content": cleaned_ai_reply, "time": now})
        temp_memory_cache.add(cleaned_ai_reply)

    if new_entries:
        memory.extend(new_entries)
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)

@atexit.register
def clear_memory_on_exit():
    global temp_memory_cache
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                memory = json.load(f)
            updated = [m for m in memory if m["content"] not in temp_memory_cache]
            with open(MEMORY_FILE, "w") as f:
                json.dump(updated, f, indent=2)
        except:
            pass

# ==============================
# AI CHAT
# ==============================
def load_system_prompt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return ""

def get_chat_response(user_input):
    system_prompt_content = load_system_prompt(SYSTEM_PROMPT_FILE)
    if not isinstance(user_input, str):
        user_input = str(user_input)

    data = {
        "model": "cyn-model",
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_input}
        ]
    }

    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
        timeout=60
    )

    if response.status_code == 200:
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "No content returned.")
    else:
        return ""

# ==============================
# CAMERA + VISION
# ==============================
def capture_webcam_frame():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    frame = None
    for _ in range(35):
        ret, latest = cap.read()
        if ret:
            frame = latest
    cap.release()
    return frame if frame is not None else None

def read_text_from_frame(frame):
    if frame is None:
        return ""
    cv2.imwrite("debug_ocr_image.png", frame)
    text = pytesseract.image_to_string(frame, config='--psm 6')
    return text.strip()

def detect_objects_from_frame(frame):
    if frame is None:
        return "No objects detected."
    results = yolo_model(frame, verbose=False)[0]
    names = results.names
    class_ids = [int(cls) for cls in results.boxes.cls]
    labels = [names[c] for c in class_ids]
    if not labels:
        return "No objects detected."
    summary = {}
    for label in labels:
        summary[label] = summary.get(label, 0) + 1
    return ', '.join([f"{count} {label if count==1 else label+'s'}" for label, count in summary.items()])

def describe_faces_in_frame(frame):
    global current_speaker

    if frame is None:
        return "I can’t see anything."

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)

    if not face_encodings:
        return "No faces detected."

    summary = {}
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
        if True in matches:
            name = known_face_names[matches.index(True)]
            summary[name] = summary.get(name, 0) + 1
            if name == USER_NAME:
                current_speaker = USER_NAME
        else:
            summary["Unknown"] = summary.get("Unknown", 0) + 1

    return "I see " + ', '.join([f"{count} {name}" if count > 1 else name for name, count in summary.items()]) + "."

def recognize_face_greeting():
    global current_speaker
    frame = capture_webcam_frame()
    if frame is None:
        return random.choice(["Camera not working.", "I can’t see anything."])

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)

    if not face_encodings:
        return random.choice(["No one’s here.", "I don’t see anyone."])

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
        if True in matches:
            name = known_face_names[matches.index(True)]
            if name == USER_NAME:
                current_speaker = USER_NAME
            return random.choice([greet.replace("{name}", name) for greet in PERSONAL_GREETINGS])
    return random.choice(["I don’t recognize you.", "Stranger detected."])

# ==============================
# TEXT / WAKEWORD
# ==============================
def precise_wakeword_detected(text):
    wake_phrases = [prefix + word for prefix in WAKE_PREFIXES for word in WAKE_WORDS]
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    for phrase in wake_phrases:
        similarity = difflib.SequenceMatcher(None, text, phrase).ratio()
        if similarity > 0.85 or phrase in text:
            return True
    return False

def check_for_farewell(user_text):
    if any(word in user_text for word in ["bye", "goodbye", "see you", "stop"]):
        print(random.choice(FAREWELLS))
        return True
    return False

# ==============================
# MAIN LOOP
# ==============================
def main():
    global current_speaker
    in_conversation = False

    while True:
        if not in_conversation:
            wake_text = hybrid_recognize_speech(timeout=None, phrase_time_limit=6)
            if wake_text and precise_wakeword_detected(wake_text):
                in_conversation = True
                if current_speaker == "Unknown":
                    current_speaker = USER_NAME
                print(recognize_face_greeting())
            continue

        user_text = hybrid_recognize_speech(timeout=8, phrase_time_limit=22.5)
        if not user_text:
            continue

        user_text_lower = user_text.lower()

        if check_for_farewell(user_text_lower):
            in_conversation = False
            current_speaker = "Unknown"
            continue

        elif re.search(r"\b(what time|current time)\b", user_text_lower):
            now = datetime.now().strftime("%I:%M %p").lstrip("0")
            response = f"The current time is {now}."
            print(response)
            auto_update_memory(user_text, response)

        elif re.search(r"\b(what day|date)\b", user_text_lower):
            now = datetime.now()
            month_day = now.strftime("%m-%d")
            holiday_greeting = HOLIDAYS.get(month_day, "")
            date_str = now.strftime("%B %#d") if os.name == "nt" else now.strftime("%B %-d")
            response = f"It's {date_str}."
            if holiday_greeting:
                response += f" {holiday_greeting}"
            print(response)
            auto_update_memory(user_text, response)

        elif re.search(r"\b(read|writing)\b", user_text_lower):
            frame = capture_webcam_frame()
            text = read_text_from_frame(frame).strip()
            response = f"It says: {text}" if text else "I can't see any readable text."
            print(response)
            auto_update_memory(user_text, response)

        elif re.search(r"\b(who do you see|who is there)\b", user_text_lower):
            frame = capture_webcam_frame()
            faces_description = describe_faces_in_frame(frame)
            print(faces_description)
            auto_update_memory(user_text, faces_description)

        elif re.search(r"\b(look|objects?)\b", user_text_lower):
            frame = capture_webcam_frame()
            faces_description = describe_faces_in_frame(frame)
            objects_description = detect_objects_from_frame(frame)
            response = f"{faces_description} I also see: {objects_description}"
            print(response)
            auto_update_memory(user_text, response)

        else:
            response = get_chat_response(user_text)
            print(response)
            auto_update_memory(user_text, response)

if __name__ == "__main__":
    main()
