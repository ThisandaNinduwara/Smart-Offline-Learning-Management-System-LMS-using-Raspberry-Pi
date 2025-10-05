import cv2
from deepface import DeepFace
import time
import numpy as np
from collections import Counter


class QuizEmotionAnalyzer:
    def __init__(self):
        # Use DroidCam video feed URL instead of webcam
        droidcam_url = "http://10.30.250.231:4747/video"
        self.cap = cv2.VideoCapture(droidcam_url)

        # Set fixed resolution for stability
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            # Fallback to regular webcam if DroidCam fails
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("❌ Camera not available")

        # Initial sensitivity value - optimized for confident/focused detection
        self.SENSITIVITY = 1.5  # Balanced for stable detection

        # Quiz-specific emotion mapping - HEAVILY FAVOR CONFIDENT AND FOCUSED
        self.quiz_emotions = {
            'happy': {'display': 'CONFIDENT', 'color': (0, 255, 0), 'status': 'Ready to answer', 'boost': 2.2},
            'neutral': {'display': 'FOCUSED', 'color': (200, 200, 200), 'status': 'Concentrating', 'boost': 2.0},
            'sad': {'display': 'CONFUSED', 'color': (0, 100, 255), 'status': 'Thinking', 'boost': 0.4},
            'surprise': {'display': 'CONFIDENT', 'color': (0, 255, 0), 'status': 'Interested', 'boost': 1.3},
            'fear': {'display': 'FOCUSED', 'color': (200, 200, 200), 'status': 'Attentive', 'boost': 0.5},
            'angry': {'display': 'FOCUSED', 'color': (200, 200, 200), 'status': 'Determined', 'boost': 0.5},
            'disgust': {'display': 'FOCUSED', 'color': (200, 200, 200), 'status': 'Analytical', 'boost': 0.5},
            'not_focused': {'display': 'NOT FOCUSED', 'color': (255, 50, 50), 'status': 'Looking away', 'boost': 1.0}
        }

        # Fixed professional settings
        self.emotion_history = []
        self.max_history = 10
        self.last_analysis_time = 0
        self.analysis_interval = 0.7
        self.emotion_threshold = 35.0
        self.current_emotion_data = None
        self.last_emotion = None
        self.min_confused_chance = 0.1
        self.not_focused_counter = 0
        self.max_not_focused_frames = 3  # Number of frames to confirm not focused

        # Professional color scheme
        self.bg_color = (30, 30, 40)
        self.text_color = (255, 255, 255)
        self.accent_color = (0, 150, 255)

        # Dialog box position (bottom right)
        self.dialog_width = 350
        self.dialog_height = 120
        self.dialog_margin = 20

    def adjust_sensitivity(self, increase=True):
        """Adjust sensitivity in real-time"""
        if increase:
            self.SENSITIVITY = min(self.SENSITIVITY + 0.1, 3.0)
        else:
            self.SENSITIVITY = max(self.SENSITIVITY - 0.1, 0.5)

        return self.SENSITIVITY

    def is_looking_away(self, emotion_data, frame):
        """Check if participant is looking away or shaking head"""
        if not emotion_data or 'region' not in emotion_data:
            return True

        region = emotion_data['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # Check if face is too small (looking away)
        if w < 100 or h < 100:
            return True

        # Check if face is near edges of screen (looking away)
        height, width = frame.shape[:2]
        if x < 50 or x + w > width - 50 or y < 50 or y + h > height - 100:
            return True

        # Check confidence - if too low, probably looking away
        if emotion_data['emotion'][emotion_data['dominant_emotion']] < 25:
            return True

        return False

    def apply_emotion_boost(self, emotion_scores):
        """Apply heavy boosting for confident and focused, suppress confused"""
        boosted_scores = emotion_scores.copy()

        # HEAVILY boost confident and focused
        boosted_scores['happy'] = boosted_scores.get('happy', 0) * 2.2 * self.SENSITIVITY
        boosted_scores['neutral'] = boosted_scores.get('neutral', 0) * 2.0 * self.SENSITIVITY

        # SUPPRESS confused emotions very heavily
        boosted_scores['sad'] = boosted_scores.get('sad', 0) * 0.4 * self.SENSITIVITY
        boosted_scores['fear'] = boosted_scores.get('fear', 0) * 0.3 * self.SENSITIVITY
        boosted_scores['angry'] = boosted_scores.get('angry', 0) * 0.3 * self.SENSITIVITY
        boosted_scores['disgust'] = boosted_scores.get('disgust', 0) * 0.3 * self.SENSITIVITY

        # Map surprise to confident with moderate boost
        boosted_scores['surprise'] = boosted_scores.get('surprise', 0) * 1.3 * self.SENSITIVITY

        # Ensure minimum values for stability
        for emotion in boosted_scores:
            boosted_scores[emotion] = max(boosted_scores[emotion], 1.0)

        # Additional: If confident or focused is already leading, boost it more
        confident_score = boosted_scores.get('happy', 0)
        focused_score = boosted_scores.get('neutral', 0)
        confused_score = boosted_scores.get('sad', 0)

        # If confident is already strong, make it stronger
        if confident_score > 25:
            boosted_scores['happy'] = confident_score * 1.3

        # If focused is already strong, make it stronger
        if focused_score > 25:
            boosted_scores['neutral'] = focused_score * 1.3

        # HEAVILY suppress confused if it tries to appear too much
        if confused_score > 20:
            boosted_scores['sad'] = confused_score * 0.2

        # Normalize scores
        total = sum(boosted_scores.values())
        if total > 0:
            for emotion in boosted_scores:
                boosted_scores[emotion] = (boosted_scores[emotion] / total) * 100

        return boosted_scores

    def analyze_emotion(self, frame):
        """Analyze emotions with heavy emphasis on CONFIDENT and FOCUSED"""
        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                align=True,
                silent=True
            )

            if results and 'emotion' in results[0]:
                # Check if looking away first
                if self.is_looking_away(results[0], frame):
                    self.not_focused_counter += 1
                    if self.not_focused_counter >= self.max_not_focused_frames:
                        # Create not focused result
                        not_focused_result = {
                            'dominant_emotion': 'not_focused',
                            'emotion': {'not_focused': 85.0},
                            'region': results[0].get('region', {'x': 0, 'y': 0, 'w': 100, 'h': 100})
                        }
                        return not_focused_result
                else:
                    self.not_focused_counter = 0  # Reset counter if focused

                boosted_scores = self.apply_emotion_boost(results[0]['emotion'])
                dominant_emotion = max(boosted_scores.items(), key=lambda x: x[1])[0]
                confidence = boosted_scores[dominant_emotion]

                # FINAL CHECK: Very rarely allow confused
                if dominant_emotion == 'sad' and confidence < 40:
                    if np.random.random() > self.min_confused_chance:
                        dominant_emotion = 'neutral'
                        confidence = boosted_scores['neutral'] * 1.1

                results[0]['dominant_emotion'] = dominant_emotion
                results[0]['emotion'] = boosted_scores
                return results[0]

        except Exception:
            pass

        # If no face detected for multiple frames, show not focused
        self.not_focused_counter += 1
        if self.not_focused_counter >= self.max_not_focused_frames:
            return {
                'dominant_emotion': 'not_focused',
                'emotion': {'not_focused': 90.0},
                'region': {'x': 0, 'y': 0, 'w': 100, 'h': 100}
            }

        return self.current_emotion_data

    def draw_professional_interface(self, frame):
        """Draw professional quiz competition interface"""
        # Create a clean background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), self.bg_color, -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        # Draw header
        header_height = 60
        cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), (20, 20, 30), -1)

        header_text = ("QUIZ COMPETITION PARTICIPANT STATUS"
                       " ANALYZER")
        text_size = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, header_text, (text_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.accent_color, 2)

        if self.current_emotion_data and 'region' in self.current_emotion_data:
            region = self.current_emotion_data['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            emotion = self.current_emotion_data['dominant_emotion']
            confidence = self.current_emotion_data['emotion'].get(emotion, 85.0)

            # Get quiz-specific emotion info
            emotion_info = self.quiz_emotions.get(emotion, self.quiz_emotions['not_focused'])
            display_name = emotion_info['display']
            color = emotion_info['color']
            status = emotion_info['status']

            # Draw face bounding box (only if not looking away)
            if emotion != 'not_focused':
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
            else:
                # Draw warning symbol for not focused
                center_x, center_y = x + w // 2, y + h // 2
                cv2.circle(frame, (center_x, center_y), 25, color, 3)
                cv2.line(frame, (center_x - 15, center_y - 15), (center_x + 15, center_y + 15), color, 3)
                cv2.line(frame, (center_x + 15, center_y - 15), (center_x - 15, center_y + 15), color, 3)

            # Draw BOTTOM-RIGHT dialog box
            dialog_x = frame.shape[1] - self.dialog_width - self.dialog_margin
            dialog_y = frame.shape[0] - self.dialog_height - self.dialog_margin - 40

            # Dialog background
            cv2.rectangle(frame, (dialog_x, dialog_y),
                          (dialog_x + self.dialog_width, dialog_y + self.dialog_height),
                          (40, 40, 50), -1)
            cv2.rectangle(frame, (dialog_x, dialog_y),
                          (dialog_x + self.dialog_width, dialog_y + self.dialog_height),
                          color, 2)

            # Dialog content
            cv2.putText(frame, "PARTICIPANT STATUS",
                        (dialog_x + 10, dialog_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)

            cv2.putText(frame, display_name,
                        (dialog_x + 10, dialog_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

            cv2.putText(frame, status,
                        (dialog_x + 10, dialog_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

            # Confidence indicator (only for real emotions)
            if emotion != 'not_focused':
                conf_text = f"{confidence:.0f}% certainty"
                conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.putText(frame, conf_text,
                            (dialog_x + self.dialog_width - conf_size[0] - 10, dialog_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        else:
            # No face detected - centered message
            text = "AWAITING PARTICIPANT"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.accent_color, 2)

        # Draw footer with sensitivity controls
        footer_height = 50
        cv2.rectangle(frame, (0, frame.shape[0] - footer_height),
                      (frame.shape[1], frame.shape[0]), (20, 20, 30), -1)

        # Sensitivity display
        sens_text = f"Sensitivity: {self.SENSITIVITY:.1f}x"
        cv2.putText(frame, sens_text, (20, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Control instructions
        controls_text = "Press '+' to increase sensitivity | 'Q' to exit"
        controls_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(frame, controls_text,
                    (frame.shape[1] - controls_size[0] - 10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def calculate_fps(self, last_time):
        """Calculate FPS"""
        current_time = time.time()
        try:
            fps = 1.0 / (current_time - last_time)
        except ZeroDivisionError:
            fps = 0
        return fps, current_time

    def run(self):
        """Main professional analysis loop"""
        last_frame_time = time.time()

        # Create window
        window_name = 'Quiz Emotion Analyzer - Focus Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 800)

        print("🎯 Optimized for CONFIDENT and FOCUSED detection!")
        print("👀 Will detect when participant looks away or shakes head")
        print("🚫 Shows 'NOT FOCUSED' when not looking at screen")
        print("➕ Press '+' to increase sensitivity")
        print("➖ Press '-' to decrease sensitivity")
        print("⏹️  Press 'Q' to quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                # If frame reading fails, try to reconnect to the camera
                print("Failed to read frame, trying to reconnect...")
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture("http://10.30.250.231:4747/video")
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(1)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(0)
                continue

            frame = cv2.flip(frame, 1)
            current_time = time.time()

            # Calculate FPS
            fps, last_frame_time = self.calculate_fps(last_frame_time)

            # Analyze at intervals
            if current_time - self.last_analysis_time > self.analysis_interval:
                new_emotion_data = self.analyze_emotion(frame)

                if new_emotion_data:
                    self.current_emotion_data = new_emotion_data
                    emotion = new_emotion_data['dominant_emotion']

                    # Add to history for smoothing
                    self.emotion_history.append(emotion)
                    if len(self.emotion_history) > self.max_history:
                        self.emotion_history.pop(0)

                    self.last_emotion = emotion

                self.last_analysis_time = current_time

            # Draw professional interface
            frame = self.draw_professional_interface(frame)

            # Display FPS in top-right corner
            fps_text = f"FPS: {fps:.1f}"
            fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame, fps_text, (frame.shape[1] - fps_size[0] - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # Display the professional interface
            cv2.imshow(window_name, frame)

            # Handle key press for real-time sensitivity adjustment
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == 27:  # ESC key
                break
            elif key == ord('+') or key == ord('='):
                new_sens = self.adjust_sensitivity(True)
                print(f"➕ Sensitivity increased to: {new_sens:.1f}x")
            elif key == ord('-') or key == ord('_'):
                new_sens = self.adjust_sensitivity(False)
                print(f"➖ Sensitivity decreased to: {new_sens:.1f}x")

    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Emotion detection stopped")


# Run the professional analyzer
if __name__ == "__main__":
    try:
        analyzer = QuizEmotionAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'analyzer' in locals():
            analyzer.cleanup()
