import cv2
import torch
import numpy as np
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import beepy
from facenet_pytorch import MTCNN, InceptionResnetV1
import threading
import uuid
from fpdf import FPDF

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ====== Load known faces from image folder and subdirectories ======
def get_embedding(pil_img):
    try:
        face = mtcnn(pil_img)
        if face is None:
            return None
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = resnet(face).cpu().numpy()[0]
        return emb
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None

def load_known_faces():
    known_faces = {}
    image_dir = Path("image")
    image_dir.mkdir(exist_ok=True)
    
    # Recursively search for image files in subdirectories
    for img_file in image_dir.rglob("*.*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        # The parent folder name is the name_class (e.g., BILL_GATE_DHCNTT)
        parent_folder = img_file.parent.name
        if not parent_folder:
            continue
        name_class = parent_folder  # e.g., BILL_GATE_DHCNTT
        if "_" in name_class:
            name, class_ = name_class.split("_", 1)
        else:
            name, class_ = name_class, "Unknown"
        img = Image.open(img_file).convert('RGB')
        emb = get_embedding(img)
        if emb is not None:
            # Use the parent folder name as the key
            if name_class not in known_faces:
                known_faces[name_class] = emb
            # If multiple images exist for the same name_class, we keep the first embedding
    return known_faces

known_faces = load_known_faces()
known_embs = torch.tensor(np.stack(list(known_faces.values()))).to(device) if known_faces else torch.tensor([]).to(device)
known_names = list(known_faces.keys())

# ====== Frame navigation ======
def show_welcome():
    add_member_frame.pack_forget()
    attendance_frame.pack_forget()
    statistics_frame.pack_forget()
    welcome_frame.pack(fill=tk.BOTH, expand=True)

def show_add_member():
    welcome_frame.pack_forget()
    attendance_frame.pack_forget()
    statistics_frame.pack_forget()
    add_member_frame.pack(fill=tk.BOTH, expand=True)

def show_attendance():
    welcome_frame.pack_forget()
    add_member_frame.pack_forget()
    statistics_frame.pack_forget()
    attendance_frame.pack(fill=tk.BOTH, expand=True)

def show_statistics_frame():
    welcome_frame.pack_forget()
    add_member_frame.pack_forget()
    attendance_frame.pack_forget()
    statistics_frame.pack(fill=tk.BOTH, expand=True)
    update_stats_comboboxes()

# ====== Add new face functions ======
def add_new_face(frame, name, class_):
    if not name or not class_:
        messagebox.showerror("Error", "Please enter both name and class")
        return
    
    name = name.strip().replace(" ", "_")
    class_ = class_.strip().replace(" ", "_")
    name_class = f"{name}_{class_}"
    save_dir = Path("image") / name_class
    save_dir.mkdir(exist_ok=True)
    # Use a unique filename (e.g., 0.jpg, 1.jpg, etc.)
    existing_files = list(save_dir.glob("*.jpg"))
    new_index = len(existing_files)
    filename = f"{new_index}.jpg"
    save_path = save_dir / filename
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    cv2.imwrite(str(save_path), frame)
    
    global known_faces, known_embs, known_names
    emb = get_embedding(img_pil)
    if emb is not None:
        known_faces[name_class] = emb
        known_embs = torch.tensor(np.stack(list(known_faces.values()))).to(device)
        known_names = list(known_faces.keys())
        messagebox.showinfo("Success", f"Added {name} ({class_}) successfully")
    else:
        messagebox.showerror("Error", "No face detected in the image")
        # Remove the file if no face is detected
        if save_path.exists():
            os.remove(save_path)
        # Remove the directory if it's empty
        if not list(save_dir.glob("*")):
            save_dir.rmdir()

def capture_face():
    ret, frame = cap.read()
    if ret:
        add_new_face(frame, name_entry.get(), class_entry.get())
    else:
        messagebox.showerror("Error", "Failed to capture image from webcam")

def load_image():
    # Validate Name and Class before opening file dialog
    name = name_entry.get().strip()
    class_ = class_entry.get().strip()
    if not name or not class_:
        messagebox.showerror("Error", "Please enter both name and class before loading an image")
        return
    
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    
    # Debug: Print the file path to verify
    print("Attempting to load image from (raw):", file_path)
    
    # Normalize the path
    file_path = os.path.normpath(file_path)
    print("Attempting to load image from (normalized):", file_path)
    
    # Check if file exists
    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"File does not exist at:\n{file_path}\nCheck the file path.")
        return
    
    # First attempt: Use cv2.imread
    img = None
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        print("cv2.imread successful:", img is not None)
    except Exception as e:
        print("cv2.imread failed:", str(e))
    
    # Fallback: Use cv2.imdecode if cv2.imread fails
    if img is None:
        print("Falling back to cv2.imdecode...")
        try:
            with open(file_path, 'rb') as f:
                file_data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
            print("cv2.imdecode successful:", img is not None)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image using fallback method from:\n{file_path}\nError: {str(e)}")
            return
    
    # Final check: Ensure image loaded successfully
    if img is None:
        messagebox.showerror("Error", f"Failed to load the image from:\n{file_path}\nCheck file path/integrity and ensure it contains a valid image.")
        return
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    emb = get_embedding(img_pil)
    if emb is None:
        messagebox.showerror("Error", "No face detected in the image")
        return
    
    # Use the validated name and class
    name = name.replace(" ", "_")
    class_ = class_.replace(" ", "_")
    name_class = f"{name}_{class_}"
    save_dir = Path("image") / name_class
    save_dir.mkdir(exist_ok=True)
    existing_files = list(save_dir.glob("*.jpg"))
    new_index = len(existing_files)
    filename = f"{new_index}.jpg"
    save_path = save_dir / filename
    cv2.imwrite(str(save_path), img)
    
    global known_faces, known_embs, known_names
    known_faces[name_class] = emb
    known_embs = torch.tensor(np.stack(list(known_faces.values()))).to(device)
    known_names = list(known_faces.keys())
    messagebox.showinfo("Success", f"Loaded and added {name} ({class_}) successfully")

# ====== GUI setup ======
root = tk.Tk()
root.title("Attendance System")
# Set initial size and make it responsive
root.geometry("900x700")
root.minsize(600, 400)  # Minimum size to prevent widgets from breaking
root.state('zoomed')  # Start maximized (optional, can be removed if not desired)

# Styling for ttk widgets
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.map("TButton",
          background=[("active", "#4CAF50")],
          foreground=[("active", "white")])
style.configure("TLabel", background="#f0f4f8", font=("Helvetica", 12))
style.configure("Treeview", rowheight=25, font=("Helvetica", 10))
style.configure("Treeview.Heading", font=("Helvetica", 11, "bold"))
style.map("Treeview", background=[("selected", "#4CAF50")])

# Menu bar setup (always visible)
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Empty CSV", command=lambda: empty_csv())
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
navmenu = tk.Menu(menubar, tearoff=0)
navmenu.add_command(label="Thêm Ảnh Người Dùng", command=lambda: show_add_member())
navmenu.add_command(label="Điểm Danh", command=lambda: show_attendance())
navmenu.add_command(label="Thống Kê", command=lambda: show_statistics_frame())
menubar.add_cascade(label="Chức Năng", menu=navmenu)
root.config(menu=menubar)

# Welcome frame with face_detection.jpg background
welcome_frame = tk.Frame(root, bg="#f0f4f8")
welcome_frame.pack(fill=tk.BOTH, expand=True)
welcome_canvas = tk.Canvas(welcome_frame, bg="#f0f4f8", highlightthickness=0)
welcome_canvas.pack(fill=tk.BOTH, expand=True, pady=10)

# Load and display face_detection.jpg, resize dynamically
original_main_img = None
try:
    original_main_img = Image.open("face_detection.jpg")
except FileNotFoundError:
    welcome_canvas.create_text(450, 325, text="face_detection.jpg not found", font=("Helvetica", 20), fill="red", tags="error_text")

def resize_welcome_image(event):
    if original_main_img is None:
        return
    canvas_width = event.width
    canvas_height = event.height
    welcome_canvas.delete("image")
    # Maintain aspect ratio
    img_aspect = original_main_img.width / original_main_img.height
    canvas_aspect = canvas_width / canvas_height
    if img_aspect > canvas_aspect:
        new_width = canvas_width
        new_height = int(canvas_width / img_aspect)
    else:
        new_height = canvas_height
        new_width = int(canvas_height * img_aspect)
    resized_img = original_main_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    main_photo = ImageTk.PhotoImage(resized_img)
    welcome_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=main_photo, tags="image")
    welcome_canvas.image = main_photo
    # Update error text position if it exists
    welcome_canvas.delete("error_text")
    if original_main_img is None:
        welcome_canvas.create_text(canvas_width // 2, canvas_height // 2, text="face_detection.jpg not found", font=("Helvetica", 20), fill="red", tags="error_text")

welcome_canvas.bind("<Configure>", resize_welcome_image)

# Add member frame
add_member_frame = tk.Frame(root, bg="#f0f4f8")
tk.Label(add_member_frame, text="Thêm Ảnh Người Dùng", font=("Helvetica", 24, "bold"), bg="#f0f4f8", fg="#333").pack(pady=20)
input_frame = tk.Frame(add_member_frame, bg="#f0f4f8")
input_frame.pack(pady=10)
tk.Label(input_frame, text="Name:", font=("Helvetica", 12), bg="#f0f4f8").pack(side=tk.LEFT, padx=5)
name_entry = ttk.Entry(input_frame, font=("Helvetica", 12), width=20)
name_entry.pack(side=tk.LEFT, padx=5)
tk.Label(input_frame, text="Class:", font=("Helvetica", 12), bg="#f0f4f8").pack(side=tk.LEFT, padx=5)
class_entry = ttk.Entry(input_frame, font=("Helvetica", 12), width=20)
class_entry.pack(side=tk.LEFT, padx=5)
load_image_button = ttk.Button(input_frame, text="Load Image", style="TButton", command=load_image)
load_image_button.pack(side=tk.LEFT, padx=5)
capture_button = ttk.Button(input_frame, text="Capture Face", style="TButton", command=capture_face)
capture_button.pack(side=tk.LEFT, padx=5)
add_canvas = tk.Canvas(add_member_frame, bg="#e0e0e0", highlightthickness=1, highlightbackground="#ccc")
add_canvas.pack(fill=tk.BOTH, expand=True, pady=10)

# Resize handler for add member canvas
def resize_add_canvas(event):
    canvas_width = event.width
    canvas_height = event.height
    # Maintain aspect ratio (e.g., 4:3 for camera feed)
    aspect_ratio = 4 / 3
    if canvas_width / canvas_height > aspect_ratio:
        new_width = int(canvas_height * aspect_ratio)
        new_height = canvas_height
    else:
        new_width = canvas_width
        new_height = int(canvas_width / aspect_ratio)
    add_canvas.config(width=new_width, height=new_height)

add_canvas.bind("<Configure>", resize_add_canvas)

# Attendance frame with canvas on top and tree below
attendance_frame = tk.Frame(root, bg="#f0f4f8")
tk.Label(attendance_frame, text="Điểm Danh", font=("Helvetica", 24, "bold"), bg="#f0f4f8", fg="#333").pack(pady=20)
canvas_frame = tk.Frame(attendance_frame, bg="#f0f4f8")
canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
attendance_canvas = tk.Canvas(canvas_frame, bg="#e0e0e0", highlightthickness=1, highlightbackground="#ccc")
attendance_canvas.pack(fill=tk.BOTH, expand=True)
attendance_label = tk.Label(canvas_frame, text="Camera Feed", font=("Helvetica", 14), bg="#f0f4f8", fg="#333")
attendance_label.pack(pady=5)
tree_frame = tk.Frame(attendance_frame, bg="#f0f4f8")
tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
tree_scroll = ttk.Scrollbar(tree_frame)
tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
tree = ttk.Treeview(tree_frame, columns=("Name", "Class", "Time", "Date"), yscrollcommand=tree_scroll.set)
tree.column("#0", width=50, anchor="center")
tree.column("Name", width=150, anchor="center")
tree.column("Class", width=150, anchor="center")
tree.column("Time", width=120, anchor="center")
tree.column("Date", width=120, anchor="center")
tree.heading("#0", text="Index")
tree.heading("Name", text="Name")
tree.heading("Class", text="Class")
tree.heading("Time", text="Time")
tree.heading("Date", text="Date")
tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
tree_scroll.config(command=tree.yview)
back_button_attendance = ttk.Button(attendance_frame, text="Back to Menu", style="TButton", command=show_welcome)
back_button_attendance.pack(pady=10)

# Resize handler for attendance canvas
def resize_attendance_canvas(event):
    canvas_width = event.width
    canvas_height = event.height
    # Maintain aspect ratio (e.g., 16:9 for camera feed)
    aspect_ratio = 16 / 9
    if canvas_width / canvas_height > aspect_ratio:
        new_width = int(canvas_height * aspect_ratio)
        new_height = canvas_height
    else:
        new_width = canvas_width
        new_height = int(canvas_width / aspect_ratio)
    attendance_canvas.config(width=new_width, height=new_height)

attendance_canvas.bind("<Configure>", resize_attendance_canvas)

# Statistics frame
statistics_frame = tk.Frame(root, bg="#f0f4f8")
tk.Label(statistics_frame, text="Thống Kê", font=("Helvetica", 24, "bold"), bg="#f0f4f8", fg="#333").pack(pady=20)

# Stats selection frame
stats_selection_frame = tk.Frame(statistics_frame, bg="#f0f4f8")
stats_selection_frame.pack(fill=tk.X, pady=10, padx=20)
tk.Label(stats_selection_frame, text="Class:", font=("Helvetica", 12), bg="#f0f4f8").pack(side=tk.LEFT, padx=5)
class_combobox = ttk.Combobox(stats_selection_frame, state="readonly", font=("Helvetica", 12), width=20)
class_combobox.pack(side=tk.LEFT, padx=5)
tk.Label(stats_selection_frame, text="Date:", font=("Helvetica", 12), bg="#f0f4f8").pack(side=tk.LEFT, padx=5)
date_combobox = ttk.Combobox(stats_selection_frame, state="readonly", font=("Helvetica", 12), width=20)
date_combobox.pack(side=tk.LEFT, padx=5)
stats_button = ttk.Button(stats_selection_frame, text="Show Statistics", style="TButton")
stats_button.pack(side=tk.LEFT, padx=5)

# PDF export frame
pdf_frame = tk.Frame(statistics_frame, bg="#f0f4f8")
pdf_frame.pack(fill=tk.X, pady=5, padx=20)
export_pdf_btn = ttk.Button(pdf_frame, text="Export to PDF", style="TButton")
export_pdf_btn.pack(side=tk.LEFT, padx=5)

stats_tree_frame = tk.Frame(statistics_frame, bg="#f0f4f8")
stats_tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
stats_tree_scroll = ttk.Scrollbar(stats_tree_frame)
stats_tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
stats_tree = ttk.Treeview(stats_tree_frame, columns=("Name", "Class", "Time", "Date"), yscrollcommand=stats_tree_scroll.set)
stats_tree.column("#0", width=50, anchor="center")
stats_tree.column("Name", width=150, anchor="center")
stats_tree.column("Class", width=150, anchor="center")
stats_tree.column("Time", width=120, anchor="center")
stats_tree.column("Date", width=120, anchor="center")
stats_tree.heading("#0", text="Index")
stats_tree.heading("Name", text="Name")
stats_tree.heading("Class", text="Class")
stats_tree.heading("Time", text="Time")
stats_tree.heading("Date", text="Date")
stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
stats_tree_scroll.config(command=stats_tree.yview)

# ====== Clear CSV ======
def empty_csv():
    csv_path = Path(__file__).with_name('Attendance.csv')
    with open(csv_path, mode='w') as file:
        file.write('Name,Class,Time,Date\n')
    for row in tree.get_children():
        tree.delete(row)

# ====== Attendance logic ======
csv_path = Path(__file__).with_name('Attendance.csv')
if not csv_path.exists() or csv_path.stat().st_size == 0:
    empty_csv()

def load_csv_df():
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame(columns=["Name", "Class", "Time", "Date"])

df = load_csv_df()

for i, row in df.iterrows():
    tree.insert('', 'end', text=str(i + 1), values=(row['Name'], row['Class'], row['Time'], row['Date']))

def mark_attendance(name, class_, img, bbox):
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%Y-%m-%d')
    with open(csv_path, mode='a') as f:
        f.write(f'{name},{class_},{time_str},{date_str}\n')
    
    def update_treeview():
        index = len(tree.get_children()) + 1
        tree.insert('', 'end', text=str(index), values=(name, class_, time_str, date_str))
        beepy.beep(sound=2)
    
    root.after(0, update_treeview)

# ====== Face detection for Add Member frame (only draw green bounding box) ======
def detect_face_for_add_member(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    boxes, _ = mtcnn.detect(img_pil)
    if boxes is None:
        return img_bgr

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

    return img_bgr

# ====== Face recognition logic for Attendance frame ======
def recognize_face(img_bgr):
    global df
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    boxes, _ = mtcnn.detect(img_pil)
    if boxes is None:
        return img_bgr

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Ensure the bounding box coordinates are valid
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > img_rgb.shape[1] or y2 > img_rgb.shape[0]:
            continue
        
        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            continue
        
        pil_face = Image.fromarray(face)
        emb = get_embedding(pil_face)
        if emb is None:
            continue
        
        emb_tensor = torch.tensor(emb).to(device)
        
        # Check if known_embs is empty
        if known_embs.size(0) == 0:
            name_class = "Unknown"
            color = (0, 0, 255)  # Red for unknown
        else:
            dists = torch.norm(known_embs - emb_tensor, dim=1)
            min_dist, min_idx = torch.min(dists, 0)
            
            if min_dist < 0.8:
                name_class = known_names[min_idx]
                color = (0, 255, 0)  # Green for recognized
            else:
                name_class = "Unknown"
                color = (0, 0, 255)  # Red for unknown
        
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = name_class
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = x1
        text_y = y2 + text_height + 5
        
        cv2.rectangle(img_bgr, (text_x, y2), (text_x + text_width, text_y), color, -1)
        cv2.putText(img_bgr, text, (text_x, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if name_class != "Unknown":
            if "_" in name_class:
                name, class_ = name_class.split("_", 1)
            else:
                name, class_ = name_class, "Unknown"
            
            df = load_csv_df()
            # Check if already marked today
            today = datetime.now().strftime('%Y-%m-%d')
            already_marked = ((df['Name'] == name) & (df['Class'] == class_) & (df['Date'] == today)).any()
            if not already_marked:
                mark_attendance(name, class_, img_bgr, (x1, y1, x2, y2))
                df = load_csv_df()

    return img_bgr

# ====== Statistics logic ======
def update_stats_comboboxes():
    df = load_csv_df()
    classes = sorted(df['Class'].unique())
    class_combobox['values'] = ['All'] + classes if classes else ['No classes available']
    class_combobox.set('All' if classes else 'No classes available')
    dates = sorted(df['Date'].unique())
    date_combobox['values'] = ['All'] + dates if dates else ['No dates available']
    date_combobox.set('All' if dates else 'No dates available')

def show_statistics():
    df = load_csv_df()
    for row in stats_tree.get_children():
        stats_tree.delete(row)

    selected_class = class_combobox.get()
    selected_date = date_combobox.get()

    stats_data = df
    if selected_class and selected_class != "No classes available" and selected_class != "All":
        stats_data = stats_data[stats_data['Class'] == selected_class]
    if selected_date and selected_date != "No dates available" and selected_date != "All":
        stats_data = stats_data[stats_data['Date'] == selected_date]

    display_index = 1
    for _, row in stats_data.iterrows():
        tag = "evenrow" if display_index % 2 == 0 else "oddrow"
        stats_tree.insert('', 'end', text=str(display_index), values=(row['Name'], row['Class'], row['Time'], row['Date']), tags=(tag,))
        display_index += 1
    stats_tree.tag_configure("evenrow", background="#f9f9f9")
    stats_tree.tag_configure("oddrow", background="#e8ecef")

stats_button.config(command=show_statistics)
class_combobox.bind('<<ComboboxSelected>>', lambda event: show_statistics())
date_combobox.bind('<<ComboboxSelected>>', lambda event: show_statistics())

# ====== PDF Export Functionality ======
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Attendance Report', 0, 1, 'C')
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf():
    selected_class = class_combobox.get()
    selected_date = date_combobox.get()
    
    df = load_csv_df()
    if selected_class and selected_class != "No classes available" and selected_class != "All":
        df = df[df['Class'] == selected_class]
    if selected_date and selected_date != "No dates available" and selected_date != "All":
        df = df[df['Date'] == selected_date]
    
    if df.empty:
        messagebox.showwarning("Warning", "No data to export")
        return
    
    default_filename = "Attendance_Report"
    if selected_class != "All":
        default_filename += f"_{selected_class}"
    if selected_date != "All":
        default_filename += f"_{selected_date}"
    default_filename += ".pdf"
    
    save_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")],
        initialfile=default_filename
    )
    
    if not save_path:
        return
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    filter_text = ""
    if selected_class != "All":
        filter_text += f"Class: {selected_class}"
    if selected_date != "All":
        if selected_class != "All":
            filter_text += ", "
        filter_text += f"Date: {selected_date}"
    if filter_text == "":
        filter_text = "All Records"
    
    pdf.cell(0, 10, filter_text, ln=1)
    pdf.ln(5)
    
    if selected_class != "All" and selected_date == "All":
        grouped = df.groupby('Date')
        for date, group in grouped:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 10, f"Date: {date}", ln=1)
            pdf.ln(2)
            
            col_widths = [60, 40, 30]
            headers = ["Name", "Class", "Time"]
            pdf.set_font("Arial", 'B', 10)
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, border=1)
            pdf.ln()
            
            pdf.set_font("Arial", size=10)
            for _, row in group.iterrows():
                for i, col in enumerate(headers):
                    pdf.cell(col_widths[i], 10, str(row[col]), border=1)
                pdf.ln()
            
            pdf.ln(5)
            
    elif selected_date != "All" and selected_class == "All":
        grouped = df.groupby('Class')
        for class_, group in grouped:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 10, f"Class: {class_}", ln=1)
            pdf.ln(2)
            
            col_widths = [60, 40, 30]
            headers = ["Name", "Class", "Time"]
            pdf.set_font("Arial", 'B', 10)
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, border=1)
            pdf.ln()
            
            pdf.set_font("Arial", size=10)
            for _, row in group.iterrows():
                for i, col in enumerate(headers):
                    pdf.cell(col_widths[i], 10, str(row[col]), border=1)
                pdf.ln()
            
            pdf.ln(5)
            
    else:
        col_widths = [60, 40, 30, 30]
        headers = ["Name", "Class", "Time", "Date"]
        pdf.set_font("Arial", 'B', 10)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, border=1)
        pdf.ln()
        
        pdf.set_font("Arial", size=10)
        for _, row in df.iterrows():
            for i, col in enumerate(headers):
                pdf.cell(col_widths[i], 10, str(row[col]), border=1)
            pdf.ln()
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 10, f"Total Records: {len(df)}", ln=1)
    
    pdf.output(save_path)
    messagebox.showinfo("Success", f"PDF exported successfully to:\n{save_path}")

export_pdf_btn.config(command=generate_pdf)

# ====== Webcam capture loop ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    messagebox.showerror("Error", "Cannot open webcam. Please check if it is connected and accessible.")
    root.destroy()
    exit()

current_frame = None

def loop():
    global current_frame
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam")
                continue

            current_frame = frame.copy()
            if attendance_frame.winfo_ismapped():
                try:
                    annotated = recognize_face(frame.copy())
                except Exception as e:
                    print(f"Error in recognize_face: {e}")
                    annotated = frame.copy()  # Continue with the original frame if recognition fails
            elif add_member_frame.winfo_ismapped():
                try:
                    annotated = detect_face_for_add_member(frame.copy())
                except Exception as e:
                    print(f"Error in detect_face_for_add_member: {e}")
                    annotated = frame.copy()
            else:
                annotated = frame.copy()
            
            img_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            if attendance_frame.winfo_ismapped():
                canvas_width = attendance_canvas.winfo_width()
                canvas_height = attendance_canvas.winfo_height()
            elif add_member_frame.winfo_ismapped():
                canvas_width = add_canvas.winfo_width()
                canvas_height = add_canvas.winfo_height()
            else:
                continue
            img_pil = img_pil.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            
            if attendance_frame.winfo_ismapped():
                attendance_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                attendance_canvas.imgtk = imgtk
            elif add_member_frame.winfo_ismapped():
                add_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                add_canvas.imgtk = imgtk
            
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                print("Application window closed. Stopping webcam loop.")
                break
    except Exception as e:
        print(f"Unexpected error in webcam loop: {e}")
    finally:
        # Ensure cleanup on thread exit
        cap.release()
        cv2.destroyAllWindows()

# Start the webcam loop in a separate thread
threading.Thread(target=loop, daemon=True).start()

# Ensure cleanup when the window is closed
def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()