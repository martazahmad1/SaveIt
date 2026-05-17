"""
SaveIt - Desktop GUI
Clean interface for Lossless Image Compression.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os

from image_compressor import (compress_image, lossless_compress, lossless_decompress,
                              smart_decompress, detect_saveit_version)

COLORS = {
    'bg_dark': '#0f172a', 'bg_panel': '#1e293b', 'bg_card': '#334155',
    'primary': '#3b82f6', 'primary_h': '#2563eb', 'accent': '#8b5cf6',
    'success': '#22c55e', 'text': '#f8fafc', 'text_dim': '#94a3b8',
    'border': '#475569', 'warning': '#f59e0b',
}


class SaveItApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SaveIt - Lossless Image Compression")
        self.root.geometry("1100x700")
        self.root.minsize(800, 500)
        self.root.configure(bg=COLORS['bg_dark'])

        self.selected_file = None
        self.last_output_path = None
        self._orig_pil = None
        self._res_pil = None

        # Mode: compress or decompress
        self.mode_var = tk.StringVar(value="compress")
        self.compress_type_var = tk.StringVar(value="ai")

        self._build_ui()
        self.root.bind("<Configure>", self._on_resize)

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=COLORS['bg_dark'], pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="SaveIt", font=("Segoe UI", 26, "bold"),
                 fg=COLORS['primary'], bg=COLORS['bg_dark']).pack()
        tk.Label(hdr, text="Lossless Image Compression  |  Pixel-Perfect Recovery",
                 font=("Segoe UI", 10), fg=COLORS['text_dim'], bg=COLORS['bg_dark']).pack()

        # Main layout
        self.pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                   bg=COLORS['bg_dark'], sashwidth=6, sashrelief=tk.FLAT)
        self.pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        left = tk.Frame(self.pane, bg=COLORS['bg_panel'],
                        highlightbackground=COLORS['border'], highlightthickness=1)
        right = tk.Frame(self.pane, bg=COLORS['bg_panel'],
                         highlightbackground=COLORS['border'], highlightthickness=1)
        self.pane.add(left, minsize=280, width=320)
        self.pane.add(right, minsize=400)

        self._build_sidebar(left)
        self._build_preview(right)

    def _build_sidebar(self, parent):
        p = tk.Frame(parent, bg=COLORS['bg_panel'])
        p.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # ── FILE SELECTION ──
        self._section_label(p, "SELECT FILE")

        self.lbl_file = tk.Label(p, text="No file selected", font=("Segoe UI", 10),
                                 fg=COLORS['text'], bg=COLORS['bg_card'],
                                 padx=10, pady=8, wraplength=260, anchor=tk.W)
        self.lbl_file.pack(fill=tk.X, pady=(4, 6))

        bf = tk.Frame(p, bg=COLORS['bg_panel'])
        bf.pack(fill=tk.X)
        self._make_btn(bf, "Browse Image", self._browse,
                       COLORS['primary']).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self._make_btn(bf, "Reset", self._reset,
                       COLORS['bg_card']).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        # ── MODE ──
        self._section_label(p, "MODE")

        for txt, val in [("Compress Image", "compress"),
                         ("Decompress .saveit", "decompress")]:
            rb = tk.Radiobutton(p, text=txt, variable=self.mode_var, value=val,
                                font=("Segoe UI", 11), fg=COLORS['text'],
                                bg=COLORS['bg_panel'], selectcolor=COLORS['bg_card'],
                                activebackground=COLORS['bg_panel'],
                                activeforeground=COLORS['text'], cursor="hand2")
            rb.pack(anchor=tk.W, pady=2)

        # ── RUN ──
        tk.Frame(p, bg=COLORS['bg_panel'], height=14).pack()
        self.btn_run = self._make_btn(p, "RUN", self._run, COLORS['accent'])
        self.btn_run.pack(fill=tk.X, pady=4)
        self.btn_run.config(font=("Segoe UI", 12, "bold"), pady=10)

        self.btn_save = self._make_btn(p, "Save Output As...", self._save_output, COLORS['primary'])
        self.btn_save.pack(fill=tk.X, pady=4)
        self.btn_save.config(state=tk.DISABLED)

        # ── LOG ──
        self._section_label(p, "LOG")
        self.log_text = tk.Text(p, height=10, wrap=tk.WORD, font=("Consolas", 9),
                                bg=COLORS['bg_card'], fg=COLORS['success'], bd=0,
                                highlightthickness=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def _build_preview(self, parent):
        top = tk.Frame(parent, bg=COLORS['bg_panel'])
        top.pack(fill=tk.X, padx=8, pady=(8, 0))
        tk.Label(top, text="Original", font=("Segoe UI", 12, "bold"),
                 fg=COLORS['text'], bg=COLORS['bg_panel']).pack(side=tk.LEFT, expand=True)
        tk.Label(top, text="Result", font=("Segoe UI", 12, "bold"),
                 fg=COLORS['accent'], bg=COLORS['bg_panel']).pack(side=tk.RIGHT, expand=True)

        cf = tk.Frame(parent, bg=COLORS['bg_dark'])
        cf.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        cf.columnconfigure(0, weight=1)
        cf.columnconfigure(1, weight=1)
        cf.rowconfigure(0, weight=1)

        self.canvas_orig = tk.Canvas(cf, bg=COLORS['bg_card'], highlightthickness=0)
        self.canvas_orig.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        self.canvas_result = tk.Canvas(cf, bg=COLORS['bg_card'], highlightthickness=0)
        self.canvas_result.grid(row=0, column=1, sticky="nsew", padx=(3, 0))

        self.info_bar = tk.Label(parent, text="Ready", font=("Segoe UI", 10),
                                  fg=COLORS['text_dim'], bg=COLORS['bg_panel'],
                                  anchor=tk.W, padx=10)
        self.info_bar.pack(fill=tk.X, pady=(0, 8))

    # ── helpers ──
    def _section_label(self, p, text):
        tk.Label(p, text=text, font=("Segoe UI", 9, "bold"),
                 fg=COLORS['text_dim'], bg=COLORS['bg_panel']).pack(anchor=tk.W, pady=(14, 2))

    def _make_btn(self, parent, text, cmd, bg_color):
        return tk.Button(parent, text=text, command=cmd, font=("Segoe UI", 10, "bold"),
                         fg="white", bg=bg_color, activebackground=COLORS['primary_h'],
                         activeforeground="white", bd=0, padx=12, pady=7, cursor="hand2")

    def _log(self, m):
        self.root.after(0, lambda: (self.log_text.insert(tk.END, m + "\n"),
                                     self.log_text.see(tk.END)))

    def _set_info(self, t):
        self.root.after(0, lambda: self.info_bar.config(text=t))

    def _on_resize(self, e):
        if self._orig_pil:
            self.root.after(100, lambda: self._draw(self.canvas_orig, self._orig_pil))
        if self._res_pil:
            self.root.after(100, lambda: self._draw(self.canvas_result, self._res_pil))

    def _draw(self, canvas, pil_img):
        canvas.delete("all")
        canvas.update_idletasks()
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        img = pil_img.copy()
        img.thumbnail((cw - 10, ch - 10), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=tk_img)
        canvas._tk_img = tk_img

    # ── file actions ──
    def _browse(self):
        ft = [("All Supported", "*.jpg *.jpeg *.png *.bmp *.webp *.saveit"),
              ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
              ("SaveIt", "*.saveit")]
        path = filedialog.askopenfilename(filetypes=ft)
        if not path:
            return
        self.selected_file = path
        self.lbl_file.config(text=os.path.basename(path))
        ext = os.path.splitext(path)[1].lower()

        if ext == '.saveit':
            self.mode_var.set("decompress")
            try:
                version = detect_saveit_version(path)
                ver_label = "Lossless v2" if version == 2 else "Autoencoder v1"
            except:
                ver_label = "Unknown"
            comp_kb = os.path.getsize(path) / 1024
            self.canvas_orig.delete("all")
            self.canvas_orig.update_idletasks()
            cw = max(self.canvas_orig.winfo_width(), 100)
            ch = max(self.canvas_orig.winfo_height(), 100)
            self.canvas_orig.create_text(cw // 2, ch // 2 - 20,
                                         text="Compressed .saveit File",
                                         fill=COLORS['primary'],
                                         font=("Segoe UI", 14, "bold"))
            self.canvas_orig.create_text(cw // 2, ch // 2 + 15,
                                         text=f"{os.path.basename(path)}",
                                         fill=COLORS['text'], font=("Segoe UI", 11))
            self.canvas_orig.create_text(cw // 2, ch // 2 + 45,
                                         text=f"{comp_kb:.0f} KB  |  {ver_label}",
                                         fill=COLORS['success'],
                                         font=("Segoe UI", 12, "bold"))
            self._set_info(f"{os.path.basename(path)}  |  {comp_kb:.0f} KB  |  {ver_label}")
        else:
            self.mode_var.set("compress")
            try:
                img = Image.open(path)
                self._orig_pil = img
                self._draw(self.canvas_orig, img)
                w, h = img.size
                kb = os.path.getsize(path) / 1024
                self._set_info(f"{os.path.basename(path)}  |  {w}x{h}  |  {kb:.0f} KB")
            except:
                pass

    def _reset(self):
        self.selected_file = None
        self.last_output_path = None
        self._orig_pil = None
        self._res_pil = None
        self.lbl_file.config(text="No file selected")
        self.canvas_orig.delete("all")
        self.canvas_result.delete("all")
        self.log_text.delete("1.0", tk.END)
        self._set_info("Ready")
        self.btn_save.config(state=tk.DISABLED)
        self.btn_run.config(state=tk.NORMAL, text="RUN")

    def _save_output(self):
        if not self.last_output_path or not os.path.exists(self.last_output_path):
            messagebox.showinfo("No output", "Run an operation first.")
            return
        ext = os.path.splitext(self.last_output_path)[1]
        dest = filedialog.asksaveasfilename(defaultextension=ext,
                                            initialfile=f"saveit_output{ext}")
        if dest:
            import shutil
            shutil.copy2(self.last_output_path, dest)
            self._log(f"Saved: {dest}")

    # ── run ──
    def _run(self):
        if not self.selected_file:
            messagebox.showwarning("No File", "Select a file first.")
            return
        self.log_text.delete("1.0", tk.END)
        self.btn_run.config(state=tk.DISABLED, text="Processing...")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        mode = self.mode_var.get()
        try:
            if mode == "compress":
                self._do_compress()
            elif mode == "decompress":
                self._do_decompress()
        except Exception as e:
            self._log(f"ERROR: {e}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL, text="RUN"))

    def _do_compress(self):
        d = os.path.dirname(self.selected_file)
        out = os.path.join(d, "compressed.saveit")

        comp_type = self.compress_type_var.get()
        
        if comp_type == "lossless":
            self._log("LOSSLESS COMPRESSION")
            self._log("=" * 40)
            self._log("Your image will be compressed without losing any quality.")
            self._log("Decompression restores exact original pixels.\n")

            result = lossless_compress(self.selected_file, output_saveit_path=out, callback=self._log)
        else:
            self._log("AI NEURAL COMPRESSION")
            self._log("=" * 40)
            self._log("Using pre-trained Convolutional Autoencoder.")
            self._log("Fast inference and massive file size reduction.\n")

            result = compress_image(self.selected_file, output_saveit_path=out, callback=self._log)

        self.last_output_path = result
        self.root.after(0, lambda: self.btn_save.config(state=tk.NORMAL))

        # Show original image on result side too
        if self._orig_pil:
            self._res_pil = self._orig_pil
            self.root.after(0, lambda: self._draw(self.canvas_result, self._orig_pil))

        # Calculate stats
        orig_file = os.path.getsize(self.selected_file)
        comp_file = os.path.getsize(result)
        img = Image.open(self.selected_file)
        w, h = img.size
        raw_size = w * h * 3  # Raw pixel size

        saved_pct = (1 - comp_file / raw_size) * 100
        mode_str = "LOSSLESS" if comp_type == "lossless" else "AI NEURAL"
        self._set_info(
            f"Raw: {raw_size/1024:.0f} KB  |  "
            f"Compressed: {comp_file/1024:.0f} KB  |  "
            f"{saved_pct:.0f}% reduced  |  {mode_str}"
        )
        self._log(f"\nDone! Use 'Save Output As...' to save the .saveit file.")

    def _do_decompress(self):
        d = os.path.dirname(self.selected_file)
        comp_kb = os.path.getsize(self.selected_file) / 1024

        self._log("LOSSLESS DECOMPRESSION")
        self._log("=" * 40)
        self._log("Restoring original image quality...\n")

        # Auto-detect version
        try:
            version = detect_saveit_version(self.selected_file)
            if version == 2:
                self._log("Format: Lossless v2 -- pixel-perfect recovery\n")
                out = os.path.join(d, "restored.png")
            else:
                self._log("Format: Autoencoder v1 -- neural reconstruction\n")
                out = os.path.join(d, "restored.jpg")
            result = smart_decompress(self.selected_file, output_path=out,
                                       callback=self._log)
        except Exception as e:
            self._log(f"Error: {e}")
            return

        self.last_output_path = result
        self.root.after(0, lambda: self.btn_save.config(state=tk.NORMAL))
        self._show_result(result)

        rest_kb = os.path.getsize(result) / 1024
        self._set_info(
            f"Compressed: {comp_kb:.0f} KB  |  "
            f"Restored: {rest_kb:.0f} KB  |  "
            f"Original quality recovered"
        )
        self._log(f"\nDone! Use 'Save Output As...' to save the restored image.")

    def _show_result(self, path):
        try:
            img = Image.open(path)
            self._res_pil = img
            self.root.after(0, lambda: self._draw(self.canvas_result, img))
        except Exception as e:
            self._log(f"Preview error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    SaveItApp(root)
    root.mainloop()
