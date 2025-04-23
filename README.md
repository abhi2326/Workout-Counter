# 🏋️ Workout-Counter

The **Workout-Counter** is a Python-based computer vision application designed to **automatically count your workout reps** using your webcam. Whether you're doing squats, push-ups, or other exercises, this tool helps track your performance in real-time — no manual counting needed!

## 🚀 Features

- 🎥 **Real-time video feed** from your webcam  
- 🤖 **Pose detection and analysis** to track movement  
- 🔢 **Accurate rep counter** based on movement patterns  
- 📝 **Customizable for different workouts**  
- 💡 Beginner-friendly and open source

## 🛠️ Technologies Used

- **Python 3**
- **OpenCV** – for webcam input and frame processing
- **MediaPipe** – for pose detection (if used)
- **NumPy** – for mathematical operations
- *(Optional)* **PyGame / Tkinter** – for GUI (if added)

## 📁 Project Structure

```
Workout-Counter/
├── gym.py         # Main script for running the workout counter
└── README.md      # Project documentation
```

## ▶️ How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/abhi2326/Workout-Counter.git
   cd Workout-Counter
   ```

2. **Install Dependencies**
   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. **Run the Program**
   ```bash
   python gym.py
   ```

4. **Start Working Out!**
   - Ensure your full body is visible to the camera.
   - Begin your reps and watch the counter do its magic 💪

## 🔄 How It Works

- Captures video using OpenCV
- Detects key body landmarks (e.g., knees, elbows)
- Calculates angles or positions to determine movement
- Increments count on complete rep motions (down → up)

## 🧩 Customization

You can adjust:
- Detection thresholds
- Angle tolerances
- Target workout type (e.g., jumping jacks, lunges)

Feel free to tweak `gym.py` to suit your needs!

## 📌 Future Improvements

- Add GUI for easier interaction  
- Support multiple workouts with a menu  
- Save workout data to file or database  
- Audio feedback for rep counts

## 👨‍💻 Author

**Abhijeet Srivastava**  
[LinkedIn](https://www.linkedin.com/in/abhijeet-sri11/)  
📧 *abhijeet.sri11@gmail.com* (replace with your real email if you want)

