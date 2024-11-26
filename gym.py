import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize Mediapipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Workout Variables
right_arm_reps = 0
left_arm_reps = 0
squat_reps = 0
calories_burned = 0
exercise_start_time = time.time()

# States for detecting transitions
right_arm_state = 'down'
left_arm_state = 'down'
squat_state = 'up'

# Buffers for stability
right_arm_buffer = []
left_arm_buffer = []
squat_buffer = []

# Store progress data
progress_data = {"Time (s)": [], "Calories Burned": []}

# Helper Function to Calculate Angle
def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Rep Counting Function with Stability
def count_reps_stable(angle, threshold_down, threshold_up, state, rep_counter, buffer):
    """Count reps using stable transition detection."""
    buffer.append(angle)

    # Keep the buffer size manageable
    if len(buffer) > 10:  # Using the last 10 frames for stability
        buffer.pop(0)

    # Compute the average angle from the buffer
    avg_angle = np.mean(buffer)

    if avg_angle > threshold_up and state == 'down':
        state = 'up'
    elif avg_angle < threshold_down and state == 'up':
        state = 'down'
        rep_counter += 1
    return state, rep_counter, buffer

# Calorie Calculation Function
def calculate_calories(reps, met, weight=70):
    """Estimate calories burned based on reps, MET value, and weight (kg)."""
    duration_minutes = (time.time() - exercise_start_time) / 60
    duration_hours = duration_minutes / 60
    return met * weight * duration_hours * reps / 50

# Progress Visualization Function
def plot_progress():
    plt.figure(figsize=(8, 5))
    plt.plot(progress_data["Time (s)"], progress_data["Calories Burned"], marker='o', label="Calories Burned")
    plt.title("Workout Progress")
    plt.xlabel("Time (s)")
    plt.ylabel("Calories Burned")
    plt.legend()
    plt.grid()
    plt.savefig("progress_chart.png")
    plt.show()

# Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for arms and squats
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

        # Calculate angles
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        squat_angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Count reps with stability
        right_arm_state, right_arm_reps, right_arm_buffer = count_reps_stable(
            right_arm_angle, 30, 160, right_arm_state, right_arm_reps, right_arm_buffer)
        left_arm_state, left_arm_reps, left_arm_buffer = count_reps_stable(
            left_arm_angle, 30, 160, left_arm_state, left_arm_reps, left_arm_buffer)
        squat_state, squat_reps, squat_buffer = count_reps_stable(
            squat_angle, 70, 160, squat_state, squat_reps, squat_buffer)

        # Calculate calories burned
        calories_burned = calculate_calories(right_arm_reps + left_arm_reps + squat_reps, 5.0)

        # Update progress data
        progress_data["Time (s)"].append(int(time.time() - exercise_start_time))
        progress_data["Calories Burned"].append(calories_burned)

        # Draw skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display metrics
        cv2.putText(image, f"Right Arm Reps: {right_arm_reps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, f"Left Arm Reps: {left_arm_reps}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, f"Squats: {squat_reps}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, f"Calories Burned: {calories_burned:.2f} kcal", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error: {e}")

    # Render
    cv2.imshow('Workout Tracker', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save and display progress
plot_progress()
