import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import timm
import os
import threading
import time
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = datasets.ImageFolder("test").classes

model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.head = torch.nn.Linear(model.head.in_features, len(class_names))
model.load_state_dict(torch.load("vit_model_best.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(path):
    image = Image.open(path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return class_names[predicted.item()], confidence.item()

def animate_loading():
    for frame in itertools.cycle(["⏳", "⌛", "🔄", "🔁"]):
        if stop_animation_flag:
            break
        label.config(text=f"İnceleniyor... {frame}", font=("Segoe UI", 13, "italic"), fg="#eeeeee", bg="#1e1e1e")
        time.sleep(0.4)

def run_prediction(file_path):
    global stop_animation_flag
    stop_animation_flag = False
    threading.Thread(target=animate_loading).start()

    tahmin, guven = predict_image(file_path)

    stop_animation_flag = True
    time.sleep(0.5)

    label.config(
        text=f"🔍 Tahmin: {tahmin.upper()}\n🎯 Güven: %{guven*100:.2f}",
        font=("Segoe UI Bold", 14),
        bg="#2a2a2a",
        fg="#ffffff"
    )

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Görseller", "*.jpg *.png *.jpeg")])
    if file_path:
        img = Image.open(file_path).resize((320, 320))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        label.config(text="", bg="#1e1e1e")

        threading.Thread(target=lambda: run_prediction(file_path)).start()

root = tk.Tk()
root.title("Hayvan Tanıma Sistemi")
root.geometry("500x620")
root.configure(bg="#1e1e1e")  # Antrasit/Siyah arka plan

title = tk.Label(
    root,
    text="🐾 Hayvan Tanıma Sistemi",
    font=("Segoe UI", 20, "bold"),
    bg="#1e1e1e",
    fg="#ffffff"
)
title.pack(pady=20)

btn = tk.Button(
    root,
    text="📁 Görsel Yükleyin",
    font=("Segoe UI Semibold", 13),
    bg="#4CAF50",
    fg="white",
    padx=15,
    pady=8,
    relief="raised",
    command=open_image
)
btn.pack(pady=10)


panel_frame = tk.Frame(root, bg="#2a2a2a", bd=2, relief="groove")
panel_frame.pack(pady=15)
panel = tk.Label(panel_frame, bg="#2a2a2a")
panel.pack()


label = tk.Label(
    root,
    text="",
    font=("Segoe UI", 13),
    bg="#1e1e1e",
    fg="#eeeeee",
    wraplength=460,
    justify="center"
)
label.pack(pady=30)


root.mainloop()
