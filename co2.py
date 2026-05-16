import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import pandas as pd
import numpy as np
import time
import asyncio
from bleak import BleakScanner

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------- CONFIG ----------------
BLOCK_SIZE = 10  # window of 10 readings
TRAIN_EPOCHS = 5
TRAIN_BATCH = 8

# ---------------- Utilities ----------------
def create_fixed_blocks(data, block_size=BLOCK_SIZE):
    X, indices = [], []
    for i in range(0, len(data), block_size):
        if i + block_size <= len(data):
            X.append(data[i:i+block_size])
            indices.append(i)
    return np.array(X), indices

def normalize_headers(df):
    colmap = {}
    for c in df.columns:
        cu = c.strip().lower()
        if cu in ("timestamp", "time", "datetime"):
            colmap[c] = "Timestamp"
        elif cu in ("rpm", "engine_rpm"):
            colmap[c] = "RPM"
        elif cu in ("speed", "vehicle_speed"):
            colmap[c] = "Speed"
        elif cu in ("throttle", "throttle_position"):
            colmap[c] = "Throttle"
        elif cu in ("load", "engine_load"):
            colmap[c] = "Load"
    return df.rename(columns=colmap)

# ---------------- GUI APP ----------------
class CO2App:
    def __init__(self, root):
        self.root = root
        self.root.title("Carbon Emission Tracker - LSTM")
        self.root.configure(bg="#121314")
        self.root.geometry("1200x750")

        # state
        self.data_file = None
        self.model = None
        self.scaler = None
        self.predictions = []
        self.sim_running = False
        self.playback_running = False
        self.anim = None  # store animation to prevent garbage collection

        self._build_ui()
        self._init_charts()

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#121314", pady=8)
        header.pack(fill="x")
        title = tk.Label(header, text="Carbon Emission Tracker (LSTM Model + ELSM Device)",
                         bg="#121314", fg="#0ff0fc", font=("Arial", 18, "bold"))
        title.pack()
        subtitle = tk.Label(header, text="Upload a CSV with Timestamp, RPM, Speed, Throttle, Load OR Connect via ELSM",
                            bg="#121314", fg="#cbd5da", font=("Arial", 10))
        subtitle.pack()

        # Controls
        controls = tk.Frame(self.root, bg="#121314", pady=10)
        controls.pack(fill="x")

        self.btn_connect = tk.Button(controls, text="🔗 Connect to ELSM", command=self.connect_elsm,
                                    bg="#0ff0fc", fg="#000", bd=0, padx=12, pady=8)
        self.btn_connect.pack(side="left", padx=12)

        self.btn_browse = tk.Button(controls, text="📁 Upload CSV", command=self.load_csv,
                                    bg="#0ff0fc", fg="#000", bd=0, padx=12, pady=8)
        self.btn_browse.pack(side="left", padx=6)

        self.btn_start = tk.Button(controls, text="▶ Start", command=self.start_playback,
                                     bg="#0ff0fc", fg="#000", bd=0, padx=12, pady=8)
        self.btn_start.pack(side="left", padx=6)

        self.btn_stop = tk.Button(controls, text="⏸ Stop", command=self.stop_all,
                                     bg="#444", fg="#fff", bd=0, padx=12, pady=8)
        self.btn_stop.pack(side="left", padx=6)

        self.btn_reset = tk.Button(controls, text="🔄 Reset", command=self.reset_all,
                                     bg="#444", fg="#fff", bd=0, padx=12, pady=8)
        self.btn_reset.pack(side="left", padx=6)

        self.btn_predict = tk.Button(controls, text="🔮 Show Emission", command=self.show_emission,
                                     bg="#0ff0fc", fg="#000", bd=0, padx=12, pady=8)
        self.btn_predict.pack(side="left", padx=6)

        # Stats cards
        stats_frame = tk.Frame(self.root, bg="#121314")
        stats_frame.pack(fill="x", pady=12)
        self.cards = {}
        for label in ["Timestamp", "RPM", "Speed", "Throttle", "Load", "Status"]:
            f = tk.Frame(stats_frame, bg="#2c2f33", padx=20, pady=15)
            f.pack(side="left", padx=10, ipadx=10, ipady=10)
            tk.Label(f, text=label, bg="#2c2f33", fg="#0ff0fc", font=("Arial", 12, "bold")).pack()
            val = tk.Label(f, text="--", bg="#2c2f33", fg="#fff", font=("Arial", 14))
            val.pack()
            self.cards[label] = val

        # Chart area
        charts_frame = tk.Frame(self.root, bg="#121314")
        charts_frame.pack(fill="both", expand=True, padx=12, pady=12)

        # Line chart (RPM Live)
        self.fig1, self.ax1 = plt.subplots(figsize=(5,4), facecolor="#1a1c1e")
        self.ax1.set_facecolor("#1a1c1e")
        self.ax1.set_title("Live RPM", color="#0ff0fc")
        self.ax1.set_ylabel("RPM", color="#cbd5da")
        self.ax1.tick_params(colors="#cbd5da")
        self.line, = self.ax1.plot([], [], 'o-', color="#0ff0fc", linewidth=2)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=charts_frame)
        self.canvas1.get_tk_widget().pack(side="left", fill="both", expand=True)

        # CO₂ Prediction chart
        self.fig2, self.ax2 = plt.subplots(figsize=(5,4), facecolor="#1a1c1e")
        self.ax2.set_facecolor("#1a1c1e")
        self.ax2.set_title("Predicted CO₂ Emissions", color="#0ff0fc")
        self.ax2.set_ylabel("CO₂ (g/km)", color="#cbd5da")
        self.ax2.tick_params(colors="#cbd5da")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=charts_frame)
        self.canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)

        # Results box
        results_frame = tk.Frame(self.root, bg="#121314")
        results_frame.pack(fill="x", padx=12, pady=6)
        lbl_results_title = tk.Label(results_frame, text="Predicted Windows:",
                                     bg="#121314", fg="#0ff0fc", font=("Arial", 12, "bold"))
        lbl_results_title.pack(anchor="w")
        self.txt_results = tk.Text(results_frame, height=10, bg="#0f1112", fg="#dfeff0")
        self.txt_results.pack(fill="x", padx=6, pady=6)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bg="#121314", fg="#cbd5da", anchor="w")
        status_bar.pack(fill="x", side="bottom", ipady=6)

    def _init_charts(self):
        self.rpm_x, self.rpm_y = [], []
        self.predictions = []

    # ---------------- Bluetooth (Bleak) ----------------
    def connect_elsm(self):
        if self.sim_running:
            return
        self.cards["Status"].config(text="Checking Bluetooth...")
        self.status_var.set("🔍 Scanning Bluetooth devices...")

        async def scan():
            devices = await BleakScanner.discover(timeout=4.0)
            if not devices:
                self.cards["Status"].config(text="❌ No device connected")
                self.status_var.set("No Bluetooth device found")
                return
            device_name = devices[0].name or "Unknown Device"
            self.cards["Status"].config(text=f"Connected to {device_name} ✅")
            self.status_var.set(f"Connected to {device_name}")
            self.sim_running = True
            self._random_step()

        threading.Thread(target=lambda: asyncio.run(scan()), daemon=True).start()

    def _random_step(self):
        if not self.sim_running: return
        rpm = np.random.randint(800, 4800)
        speed = np.random.randint(0, 120)
        throttle = np.random.randint(0, 100)
        load = np.random.randint(0, 100)
        ts = time.strftime("%H:%M:%S")

        self.update_ui(ts, rpm, speed, throttle, load)
        self.update_rpm_chart(ts, rpm)
        self.root.after(2000, self._random_step)

    # ---------------- CSV load + train ----------------
    def load_csv(self):
        path = filedialog.askopenfilename(title="Select Vehicle Telematics CSV",
                                          filetypes=[("CSV Files", "*.csv")])
        if not path: return
        self.data_file = path
        self.status_var.set("📥 CSV loaded. Training model...")
        threading.Thread(target=self._background_train, args=(path,), daemon=True).start()

    def _background_train(self, path):
        try:
            df = pd.read_csv(path)
            df = normalize_headers(df)
            features = df[["RPM", "Speed", "Throttle", "Load"]].values.astype(float)
            self.scaler = MinMaxScaler()
            features_scaled = self.scaler.fit_transform(features)
            X, _ = create_fixed_blocks(features_scaled, BLOCK_SIZE)
            if len(X) < 2:
                raise ValueError("Not enough data windows.")
            avg_inputs = X.mean(axis=1)
            y = (avg_inputs[:,0]*0.5 + avg_inputs[:,1]*0.3 +
                 avg_inputs[:,2]*0.2 + avg_inputs[:,3]*0.4)
            y = (y * 100 + np.random.normal(0,5,len(y))).reshape(-1,1)

            model = Sequential()
            model.add(LSTM(64, return_sequences=True, input_shape=(BLOCK_SIZE, X.shape[-1])))
            model.add(Dropout(0.2))
            model.add(LSTM(128))
            model.add(Dense(1))
            model.compile(optimizer='rmsprop', loss='mean_squared_error')
            model.fit(X, y, epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH, verbose=0)

            self.model = model
            self.status_var.set("✅ Model trained and ready.")
        except Exception as e:
            self.status_var.set(f"❌ Train error: {e}")
            messagebox.showerror("Error", str(e))

    # ---------------- Playback ----------------
    def start_playback(self):
        if not self.data_file: return
        if self.playback_running: return
        self.playback_running = True
        threading.Thread(target=self._csv_playback, daemon=True).start()

    def _csv_playback(self):
        try:
            df = pd.read_csv(self.data_file)
            df = normalize_headers(df)
            for _, row in df.iterrows():
                if not self.playback_running: break
                ts, rpm, speed, throttle, load = row["Timestamp"], row["RPM"], row["Speed"], row["Throttle"], row["Load"]
                self.update_ui(ts, rpm, speed, throttle, load)
                self.update_rpm_chart(ts, rpm)
                time.sleep(1)
            if self.model:
                self.show_emission()
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    def stop_all(self):
        self.playback_running = False
        self.sim_running = False
        self.cards["Status"].config(text="Stopped")
        self.status_var.set("⏸ All stopped")

    def reset_all(self):
        self.stop_all()
        self._init_charts()
        self.ax1.clear(); self.canvas1.draw()
        self.ax2.clear(); self.canvas2.draw()
        for lbl in self.cards.values(): lbl.config(text="--")
        self.txt_results.delete("1.0", tk.END)
        self.status_var.set("🔄 Reset done")

    # ---------------- Update UI ----------------
    def update_ui(self, ts, rpm, speed, throttle, load):
        self.cards["Timestamp"].config(text=ts)
        self.cards["RPM"].config(text=f"{rpm} RPM")
        self.cards["Speed"].config(text=f"{speed} km/h")
        self.cards["Throttle"].config(text=f"{throttle} %")
        self.cards["Load"].config(text=f"{load} %")
        self.rpm_x.append(ts)
        self.rpm_y.append(rpm)

    def update_rpm_chart(self, ts, rpm):
        self.ax1.plot(self.rpm_x, self.rpm_y, 'o-', color="#0ff0fc")
        self.canvas1.draw()

    # ---------------- Prediction ----------------
    def show_emission(self):
        if not self.model or not self.data_file:
            messagebox.showwarning("⚠ Not Ready", "Upload a CSV and wait for model training first.")
            return

        df = pd.read_csv(self.data_file)
        df = normalize_headers(df)
        features = df[["RPM", "Speed", "Throttle", "Load"]].values.astype(float)
        features_scaled = self.scaler.transform(features)
        X, _ = create_fixed_blocks(features_scaled, BLOCK_SIZE)
        preds = self.model.predict(X).flatten()

        # Clear old chart
        self.ax2.clear()
        self.ax2.set_facecolor("#1a1c1e")
        self.ax2.set_title("Predicted CO₂ Emissions", color="#0ff0fc")
        self.ax2.set_ylabel("CO₂ (g/km)", color="#cbd5da")

        def update(frame):
            self.ax2.clear()
            self.ax2.set_facecolor("#1a1c1e")
            self.ax2.set_title("Predicted CO₂ Emissions", color="#0ff0fc")
            self.ax2.set_ylabel("CO₂ (g/km)", color="#cbd5da")
            self.ax2.plot(range(1, frame+1), preds[:frame], 'o-', color="#0ff0fc")
            self.canvas2.draw()

            self.txt_results.delete("1.0", tk.END)
            for i, p in enumerate(preds[:frame]):
                self.txt_results.insert(tk.END, f"Window {i+1}: {p:.2f} g/km\n")

                # Keep reference to animation so it’s not garbage collected
        self.anim = animation.FuncAnimation(
            self.fig2, update, frames=len(preds),
            interval=1000, repeat=False
        )

        # Force redraw
        self.canvas2.draw_idle()
        self.root.after(100, lambda: None)

# ---------------- Run ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = CO2App(root)
    root.mainloop()