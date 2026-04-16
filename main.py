from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time
import threading

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. SETUP HARDWARE (BCM PIN NUMBERS)
# ==========================================
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# L298N Motor Pins (DC Motors)
IN1, IN2, IN3, IN4 = 17, 27, 22, 23 
# TB6600 Stepper Pins (Camera Slider)
PUL, DIR = 18, 24

dc_pins = [IN1, IN2, IN3, IN4]
all_pins = [IN1, IN2, IN3, IN4, PUL, DIR]

for pin in all_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)

# Setup PWM Heartbeat for the Stepper Motor (1000 Hz)
stepper_pwm = GPIO.PWM(PUL, 1000)

# ==========================================
# 2. LOAD AI MODEL & CAMERA THREAD
# ==========================================
model = YOLO("tomato_model.pt") 
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

# --- Anti-Crash Camera Threading ---
# This prevents the stream and the AI from fighting over the webcam
latest_frame = None
frame_lock = threading.Lock()

def capture_frames():
    global latest_frame
    while True:
        success, frame = camera.read()
        if success:
            with frame_lock:
                latest_frame = frame.copy()
        time.sleep(0.03) # Limits to ~30fps to save Pi CPU

# Start the background camera thread immediately
threading.Thread(target=capture_frames, daemon=True).start()

def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            frame_to_stream = latest_frame
            
        if frame_to_stream is not None:
            _, buffer = cv2.imencode('.jpg', frame_to_stream)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05) # Stream to app at ~20fps

# ==========================================
# 3. ENDPOINTS
# ==========================================
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    cmd = data.get("command") 
    
    # --- Rover Driving Logic (DC Motors) ---
    if cmd == "forward":
        GPIO.output(IN1, True);  GPIO.output(IN2, False)
        GPIO.output(IN3, True);  GPIO.output(IN4, False)
    elif cmd == "backward":
        GPIO.output(IN1, False); GPIO.output(IN2, True)
        GPIO.output(IN3, False); GPIO.output(IN4, True)
    elif cmd == "left":
        GPIO.output(IN1, False); GPIO.output(IN2, True)
        GPIO.output(IN3, True);  GPIO.output(IN4, False)
    elif cmd == "right":
        GPIO.output(IN1, True);  GPIO.output(IN2, False)
        GPIO.output(IN3, False); GPIO.output(IN4, True)
    
    # --- Camera Slider Logic (Stepper) ---
    elif cmd == "up":
        GPIO.output(DIR, True) 
        stepper_pwm.start(50) # Start smooth movement (50% duty cycle pulse)
    elif cmd == "down":
        GPIO.output(DIR, False) 
        stepper_pwm.start(50) # Start smooth movement in reverse
        
    # --- Universal Stop ---
    elif cmd == "stop":
        for pin in dc_pins: 
            GPIO.output(pin, False) # Stop wheels
        stepper_pwm.stop()          # Stop camera slider

    return jsonify({"status": "success", "command": cmd})

@app.route('/detect', methods=['GET'])
def detect():
    global latest_frame
    with frame_lock:
        frame_to_scan = latest_frame
        
    if frame_to_scan is None: 
        return jsonify({"error": "Camera failed or warming up"})
    
    # Run AI Model on the safely grabbed frame
    results = model(frame_to_scan)[0]
    
    # Check for Classification Results
    if results.probs is not None:
        class_id = results.probs.top1
        label = results.names[class_id]
        conf = float(results.probs.top1conf)
        return jsonify({"disease": label, "confidence": f"{conf*100:.1f}%"})
    
    return jsonify({"disease": "Unknown", "confidence": "0.0%"})

# ==========================================
# 4. SERVER EXECUTION & SAFETY CLEANUP
# ==========================================
if __name__ == "__main__":
    print("🚀 Rover Server starting on http://192.168.10.222:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down safely...")
    finally:
        # This guarantees motors turn off if the script crashes or is stopped!
        print("🧹 Cleaning up GPIO pins...")
        camera.release()
        stepper_pwm.stop()
        for pin in all_pins:
            GPIO.output(pin, False)
        GPIO.cleanup()
        print("Done.")