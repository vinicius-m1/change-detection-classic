import tkinter as tk
from tkinter import messagebox, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
from skimage.exposure import match_histograms

class ImagePanel(tk.Frame):
    def __init__(self, master, title, **kwargs):
        super().__init__(master, width=350, height=350, relief="sunken", borderwidth=2, **kwargs)
        self.pack_propagate(False)
        
        self.title = title
        self.image_path = None
        self.cv_image = None
        
        self.label = tk.Label(self, text=title, wraplength=300)
        self.label.pack(expand=True, fill=tk.BOTH)
        self.label.bind("<Button-1>", self.open_full_size)
        
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_drop)
        
    def open_full_size(self, event):
        if self.cv_image is None: return
        top = tk.Toplevel(self)
        top.title(f"{self.title}")
        
        rgb_img = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.full_pil_img = Image.fromarray(rgb_img)
        self.zoom_level = 1.0
        
        frame = tk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.zoom_canvas = tk.Canvas(frame, width=min(1200, self.full_pil_img.width), 
                                     height=min(800, self.full_pil_img.height))
        self.zoom_canvas.grid(row=0, column=0, sticky="nsew")
        
        vbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=self.zoom_canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.zoom_canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")
        
        self.zoom_canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self.update_zoom()
        
        top.bind("<Control-MouseWheel>", self.on_zoom)

    def on_zoom(self, event):
        if hasattr(event, 'delta') and event.delta > 0: self.zoom_level *= 1.1
        else: self.zoom_level /= 1.1
        self.zoom_level = max(0.1, min(self.zoom_level, 10.0))
        self.update_zoom()

    def update_zoom(self):
        w, h = self.full_pil_img.size
        new_size = (int(w * self.zoom_level), int(h * self.zoom_level))
        resized_pil = self.full_pil_img.resize(new_size, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(image=resized_pil)
        self.zoom_canvas.delete("all")
        self.zoom_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        self.zoom_canvas.image = tk_img 
        self.zoom_canvas.config(scrollregion=self.zoom_canvas.bbox(tk.ALL))

    def on_drop(self, event):
        file_path = event.data.strip('{}')
        self.load_image(file_path)

    def load_image(self, path):
        try:
            img = cv2.imread(path)
            if img is None: raise ValueError("Could not load image")
            self.image_path = path
            self.cv_image = img
            self.display_image(img)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def display_image(self, cv_img):
        self.cv_image = cv_img
        h, w = cv_img.shape[:2]
        scale = min(340/w, 340/h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cv_img, (new_w, new_h))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(rgb_img))
        self.label.config(image=tk_img, text="")
        self.label.image = tk_img

class ChangeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Change Detection")
        self.root.geometry("1000x950")
        
        self.current_mask = None # Store binary mask

        # Menu Tkinter
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(pady=10, fill=tk.X)
        self.panel1 = ImagePanel(self.top_frame, "Image 1 (Reference)")
        self.panel1.pack(side=tk.LEFT, padx=20, expand=True)
        self.panel2 = ImagePanel(self.top_frame, "Image 2 (Target)")
        self.panel2.pack(side=tk.LEFT, padx=20, expand=True)

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)

        self.btn_row1 = tk.Frame(self.control_frame)
        self.btn_row1.pack()
        tk.Button(self.btn_row1, text="Align ECC", command=self.apply_alignment).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_row1, text="Match Histograms", command=self.apply_hist_matching).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_row1, text="Clean Noise", command=self.apply_cleaning).pack(side=tk.LEFT, padx=5)      
        self.btn_row2 = tk.Frame(self.control_frame)
        self.btn_row2.pack(pady=5)
        tk.Button(self.btn_row2, text="Run Subtraction", command=self.apply_subtraction, font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_row2, text="Otsu Threshold", command=self.apply_otsu).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_row2, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_row2, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)

        # Sliders & Toggles
        self.opt_frame = tk.Frame(self.control_frame)
        self.opt_frame.pack(pady=5)
        tk.Label(self.opt_frame, text="Threshold:").pack(side=tk.LEFT)
        self.thresh_slider = tk.Scale(self.opt_frame, from_=0, to=255, orient=tk.HORIZONTAL)
        self.thresh_slider.set(30)
        self.thresh_slider.pack(side=tk.LEFT, padx=10)
        
        self.ghost_var = tk.BooleanVar(value=False)
        self.ghost_check = tk.Checkbutton(self.opt_frame, text="Ghost Overlay", variable=self.ghost_var, command=self.refresh_display)
        self.ghost_check.pack(side=tk.LEFT, padx=20)

        self.result_panel = ImagePanel(root, "Final Result")
        self.result_panel.pack(pady=10)

    def get_images(self):
        if self.panel1.cv_image is None or self.panel2.cv_image is None:
            messagebox.showwarning("Warning", "Load two images first.")
            return None, None
        img1, img2 = self.panel1.cv_image, self.panel2.cv_image
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2

    def apply_hist_matching(self):
        img1, img2 = self.get_images()
        if img1 is None: return
        # Match img2 to img1 color profile
        matched = match_histograms(img2, img1, channel_axis=-1)
        self.panel2.display_image(matched.astype(np.uint8))

    def apply_cleaning(self):
        if self.current_mask is None: return
        kernel = np.ones((3,3), np.uint8)
        # Remover noise
        cleaned = cv2.morphologyEx(self.current_mask, cv2.MORPH_OPEN, kernel)
        self.current_mask = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        self.refresh_display()

    def apply_subtraction(self):
        img1, img2 = self.get_images()
        if img1 is None: return
        g1, g2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g1, g2)
        _, self.current_mask = cv2.threshold(diff, self.thresh_slider.get(), 255, cv2.THRESH_BINARY)
        self.refresh_display()

    def refresh_display(self):
        if self.current_mask is None: return
        img1 = self.panel1.cv_image
        
        if self.ghost_var.get():
            # Create red overlay
            ghost = img1.copy()
            red_mask = np.zeros_like(img1)
            red_mask[:, :] = [0, 0, 255] # Red BGR
            # where mask 255, ghost red into the original
            mask_bool = self.current_mask == 255
            ghost[mask_bool] = cv2.addWeighted(img1[mask_bool], 0.5, red_mask[mask_bool], 0.5, 0)
            self.result_panel.display_image(ghost)
        else:
            self.result_panel.display_image(cv2.cvtColor(self.current_mask, cv2.COLOR_GRAY2BGR))

    def apply_alignment(self):
        img1, img2 = self.get_images()
        if img1 is None: return
        g1, g2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_matrix = cv2.findTransformECC(g1, g2, warp_matrix, cv2.MOTION_AFFINE)
            aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), 
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            self.panel2.display_image(aligned)
        except Exception as e:
            messagebox.showerror("Error", "Alignment failed.")

    def apply_otsu(self):
        img1, img2 = self.get_images()
        if img1 is None: return
        diff = cv2.absdiff(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
        val, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.thresh_slider.set(int(val))
        self.apply_subtraction()

    def save_result(self):
        if self.result_panel.cv_image is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path: cv2.imwrite(path, self.result_panel.cv_image)

    def clear_all(self):
        for p in [self.panel1, self.panel2, self.result_panel]:
            p.label.config(image='', text=p.title)
            p.cv_image = None
        self.current_mask = None

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ChangeDetectionApp(root)
    root.mainloop()
