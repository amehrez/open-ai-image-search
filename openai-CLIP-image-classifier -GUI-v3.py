import os
import subprocess
import sys
import threading
import torch
import clip
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
import time

# Add the global flag
stop_thread = False

# Create the main window
app = tk.Tk()
app.state('zoomed')  # Start the window maximized
app.title("Image Classifier")

# Create the GUI widgets
frame = tk.Frame(app)
frame.pack(padx=10, pady=10)

labels_label = tk.Label(frame, text="Labels (comma separated):")
labels_label.grid(row=0, column=0, sticky=tk.W, pady=5)

progress_percent_label = tk.Label(app, text="")
progress_percent_label.pack(side=tk.BOTTOM)


def stop_process_images():
    global stop_thread
    stop_thread = True


def open_image(image_path):
    if sys.platform == "win32":
        os.startfile(image_path)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, image_path])


def update_progressbar(processed_images, total_images):
    progress_percentage = f"{processed_images / total_images * 100:.2f}%"
    progress_var.set(processed_images)
    progress_bar.configure(value=processed_images,
                           mode='determinate',
                           orient='horizontal',
                           length=300,
                           variable=progress_var,
                           maximum=total_images,
                           style='green.Horizontal.TProgressbar')
    progress_bar.update()
    app.update_idletasks()
    progress_percent_label.config(text=progress_percentage)
    result_canvas.bind_all("<MouseWheel>", lambda event: result_canvas.yview_scroll(
        int(-1 * (event.delta / 120)), "units"))


def process_images_thread():
    global stop_thread
    # Reset the stop_thread flag
    stop_thread = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    labels = [label.strip() for label in labels_entry.get().split(',')]
    text = clip.tokenize(labels).to(device)

    image_folder = filedialog.askdirectory(title="Select Folder with Images")
    image_files = [
        f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
    ]

    min_probability = float(min_probability_entry.get()) / 100

    total_images = len(image_files)
    processed_images = 0

    for widget in result_canvas_frame.winfo_children():
        widget.destroy()

    progress_var.set(0)
    progress_bar.configure(value=0, maximum=total_images)

    column = 0
    row = 0

    for image_file in image_files:
        # Check for the stop_thread flag
        if stop_thread:
            break
        image_path = os.path.join(image_folder, image_file)
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        except OSError:
            print(
                f"Error processing {image_file}: broken data stream when reading image file")
            continue
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        max_index = probs.argmax()
        if probs[0, max_index] < min_probability:
            continue

        image_thumbnail = Image.open(image_path)
        image_thumbnail.thumbnail((100, 100))
        image_tk = ImageTk.PhotoImage(image_thumbnail)

        result_subframe = ttk.Frame(result_canvas_frame, padding=5)

        image_label = tk.Label(result_subframe, image=image_tk)
        image_label.image = image_tk
        image_label.pack(side=tk.LEFT)
        image_label.bind("<Button-1>", lambda e,
                         path=image_path: open_image(path))

        text_label = tk.Label(
            result_subframe,
            text=f"{image_file}\nLabel: {labels[max_index]}\nProbability: {probs[0, max_index]:.4f}",
            justify=tk.LEFT,
        )
        text_label.pack(side=tk.LEFT, padx=5)

        result_subframe.grid(row=row, column=column, padx=5, pady=5)

        column += 1
        if column >= 3:
            column = 0
            row += 1

        processed_images += 1
        update_progressbar(processed_images, total_images)

########################################################################

    progress_bar.configure(mode='determinate', value=total_images)
    progress_bar.update()
    app.update_idletasks()

    # Update the progress bar to show the finished state
    if processed_images == 0:
        progress_bar.configure(mode='determinate', value=total_images)
    else:
        progress_bar.configure(mode='determinate', value=processed_images)

    # Add MouseWheel event to the canvas
    result_canvas.bind_all("<MouseWheel>", lambda event: result_canvas.yview_scroll(
        int(-1 * (event.delta / 120)), "units"))

    # Update the status label to "Finished"
    status_label.config(text="Finished")
    app.update_idletasks()


def process_images():
    # Update the status label to "Running"
    status_label.config(text="Running")
    app.update_idletasks()

    threading.Thread(target=process_images_thread).start()


# Create the status label widget
status_label = tk.Label(app, text="")

# Add the widgets to the GUI
status_label.pack(side=tk.BOTTOM)

progress_label = tk.Label(frame, text="")
progress_label.grid(row=4, column=0, columnspan=2, pady=5)

status_label.config(text="Finished")
labels_entry = tk.Entry(frame, width=40)
labels_entry.insert(0, "food, dead clown")

labels_entry.grid(row=0, column=1, sticky=tk.W, pady=5)

min_probability_label = tk.Label(frame, text="Min probability (%):")
min_probability_label.grid(row=1, column=0, sticky=tk.W, pady=5)

min_probability_entry = tk.Entry(frame, width=10)
min_probability_entry.insert(0, "99")
min_probability_entry.grid(row=1, column=1, sticky=tk.W, pady=5)

start_button = ttk.Button(
    frame, text="Select Folder and Process Images", command=process_images)
start_button.grid(row=2, column=0, columnspan=2, pady=10)

# Add the stop button in the GUI
stop_button = ttk.Button(frame, text="Stop", command=stop_process_images)
stop_button.grid(row=2, column=2, pady=10)


progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(
    frame, variable=progress_var, maximum=100, length=300)
progress_bar.grid(row=3, column=0, columnspan=2, pady=10)

result_frame = tk.Frame(app, bd=2, relief=tk.SUNKEN)
result_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

result_canvas = tk.Canvas(result_frame, width=400, height=400)
result_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(
    result_frame, orient="vertical", command=result_canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

result_canvas.config(yscrollcommand=scrollbar.set)

result_canvas_frame = tk.Frame(result_canvas)
result_canvas.create_window((0, 0), window=result_canvas_frame, anchor=tk.NW)

app.mainloop()
