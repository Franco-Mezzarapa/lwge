import threading
import customtkinter
from tkinter import filedialog
import queue
import os

class ControlPanel:
    """CustomTkinter UI with modern tabbed interface"""
    def __init__(self):
        self.app = None
        self.running = True
        self.ui_started = False
        
        # Store imported models
        self.imported_models = {}
        self.model_combo = None
        self.scene_listbox = None
        
        # Skybox images
        self.skybox_images = {'right': None, 'left': None, 'top': None, 'bottom': None, 'front': None, 'back': None}
        self.skybox_equirect = None
        self.skybox_type = "equirectangular"
        
        # Queues for thread-safe communication
        self.model_load_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Engine reference
        self.engine = None
    
    def update_scene_list(self, model_names=None):
        """Update the scene model list"""
        if self.scene_listbox:
            if model_names is None and self.engine:
                model_names = [getattr(m, 'name', f'Model {i+1}') for i, m in enumerate(self.engine.models)]
            
            if not model_names:
                model_names = ["No models in scene"]
            
            current_selection = self.scene_listbox.get()
            self.scene_listbox.configure(values=model_names)
            
            if current_selection in model_names:
                self.scene_listbox.set(current_selection)
            elif model_names and model_names[0] != "No models in scene":
                self.scene_listbox.set(model_names[0])
            else:
                self.scene_listbox.set("No models in scene")
    
    def browse_file(self):
        """Browse for 3D model file"""
        filetypes = (
            ('3D Model files', '*.gltf *.glb *.obj *.fbx'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(title='Select a 3D model file', filetypes=filetypes)
        
        if filename:
            display_name = os.path.basename(filename)
            self.imported_models[display_name] = filename
            
            if self.model_combo:
                self.model_combo.configure(values=list(self.imported_models.keys()))
                self.model_combo.set(display_name)
            
            print(f"Imported model: {filename}")
    
    def add_model_to_scene(self):
        """Add selected model to scene"""
        if self.model_combo:
            selected = self.model_combo.get()
            if selected and selected in self.imported_models:
                self.model_load_queue.put(self.imported_models[selected])
                print(f"Adding model to scene: {selected}")
    
    def set_transform_mode(self, mode):
        """Transform mode - now handled by keyboard"""
        pass
    
    def delete_selected(self):
        """Delete selected model"""
        self.command_queue.put(('delete_selected', None))
    
    def duplicate_selected(self):
        """Duplicate selected model"""
        self.command_queue.put(('duplicate_selected', None))
    
    def browse_skybox_equirect(self):
        """Browse for equirectangular skybox"""
        filetypes = (('Image files', '*.hdr *.png *.jpg *.jpeg *.exr'), ('All files', '*.*'))
        filename = filedialog.askopenfilename(title='Select equirectangular skybox', filetypes=filetypes)
        
        if filename:
            self.skybox_equirect = filename
            print(f"Selected equirectangular skybox: {filename}")
    
    def load_skybox(self):
        """Load the selected skybox"""
        if self.skybox_type == "cubemap":
            if all(self.skybox_images.values()):
                self.command_queue.put(('load_skybox_cubemap', self.skybox_images.copy()))
            else:
                print("ERROR: All 6 cubemap faces must be selected")
        else:
            if self.skybox_equirect:
                self.command_queue.put(('load_skybox_equirect', self.skybox_equirect))
            else:
                print("ERROR: No equirectangular image selected")
    
    def clear_skybox(self):
        """Clear current skybox"""
        self.command_queue.put(('clear_skybox', None))
        self.skybox_images = {face: None for face in self.skybox_images}
        self.skybox_equirect = None
    
    def show_rename_dialog(self, model):
        """Show rename dialog"""
        if not self.app:
            return
        
        dialog = customtkinter.CTkToplevel(self.app)
        dialog.title("Rename Model")
        dialog.geometry("300x150")
        dialog.attributes('-topmost', True)
        dialog.transient(self.app)
        dialog.grab_set()
        
        label = customtkinter.CTkLabel(dialog, text=f"Rename: {getattr(model, 'name', 'Model')}", font=("Arial", 12, "bold"))
        label.pack(pady=15)
        
        entry = customtkinter.CTkEntry(dialog, width=250)
        entry.pack(pady=10)
        entry.insert(0, getattr(model, 'name', 'Model'))
        entry.focus()
        
        def on_confirm():
            new_name = entry.get().strip()
            if new_name:
                self.command_queue.put(('rename_model', (model, new_name)))
            dialog.destroy()
        
        button_frame = customtkinter.CTkFrame(dialog)
        button_frame.pack(pady=10)
        
        customtkinter.CTkButton(button_frame, text="Confirm", command=on_confirm, width=100, fg_color="green").pack(side="left", padx=5)
        customtkinter.CTkButton(button_frame, text="Cancel", command=dialog.destroy, width=100, fg_color="gray").pack(side="left", padx=5)
        
        entry.bind('<Return>', lambda e: on_confirm())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
    
    def _create_models_tab(self, parent):
        """Create Models tab"""
        customtkinter.CTkLabel(parent, text="Import and manage 3D models", font=("Arial", 12), text_color="gray").pack(pady=(10, 15))
        
        customtkinter.CTkButton(parent, text="📁 Browse Model Files", command=self.browse_file, height=40, 
                               font=("Arial", 13, "bold"), fg_color="#2B7A0B", hover_color="#1F5808").pack(pady=10, padx=20, fill="x")
        
        imported_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        imported_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        customtkinter.CTkLabel(imported_frame, text="Imported Models:", font=("Arial", 12, "bold"), anchor="w").pack(pady=(5, 5), fill="x")
        
        self.model_combo = customtkinter.CTkComboBox(imported_frame, values=["No models imported"], state="readonly", height=35)
        self.model_combo.pack(pady=5, fill="x")
        self.model_combo.set("No models imported")
        
        customtkinter.CTkButton(imported_frame, text="➕ Add to Scene", command=self.add_model_to_scene, height=35, 
                               font=("Arial", 12, "bold"), fg_color="#1F6AA5", hover_color="#144870").pack(pady=15, fill="x")
        
        customtkinter.CTkLabel(parent, text="Supported: .gltf, .glb, .obj, .fbx", font=("Arial", 10), text_color="gray").pack(pady=(0, 10))
    
    def _create_transform_tab(self, parent):
        """Create Transform tab"""
        customtkinter.CTkLabel(parent, text="Transform selected model", font=("Arial", 12), text_color="gray").pack(pady=(10, 15))
        
        tools_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        tools_frame.pack(pady=10, padx=20, fill="x")
        
        for text, mode, color, hover in [
            ("✥ Move (G)", 'move', "#1F6AA5", "#144870"),
            ("↻ Rotate (R)", 'rotate', "#7E3794", "#5A2769"),
            ("⇲ Scale (S)", 'scale', "#D97706", "#A85A05")
        ]:
            customtkinter.CTkButton(tools_frame, text=text, command=lambda m=mode: self.set_transform_mode(m), 
                                   height=50, font=("Arial", 13, "bold"), fg_color=color, hover_color=hover).pack(pady=5, fill="x")
        
        customtkinter.CTkFrame(parent, height=2, fg_color="gray30").pack(pady=15, padx=20, fill="x")
        
        actions_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        actions_frame.pack(pady=10, padx=20, fill="x")
        
        customtkinter.CTkButton(actions_frame, text="⎘ Duplicate (Ctrl+D)", command=self.duplicate_selected, height=40, 
                               font=("Arial", 12, "bold"), fg_color="#0D9488", hover_color="#0A6D65").pack(pady=5, fill="x")
        customtkinter.CTkButton(actions_frame, text="🗑 Delete (X)", command=self.delete_selected, height=40, 
                               font=("Arial", 12, "bold"), fg_color="#DC2626", hover_color="#991B1B").pack(pady=5, fill="x")
        
        instructions = customtkinter.CTkTextbox(parent, height=120, font=("Arial", 10), fg_color="gray20", wrap="word")
        instructions.pack(pady=15, padx=20, fill="x")
        instructions.insert("1.0", "KEYBOARD SHORTCUTS:\n\n• G - Move\n• R - Rotate\n• S - Scale\n• X/Y/Z - Axis constraint\n• Enter - Confirm\n• ESC - Cancel\n• F2 - Rename")
        instructions.configure(state="disabled")
    
    def _create_scene_tab(self, parent):
        """Create Scene tab"""
        customtkinter.CTkLabel(parent, text="Manage scene models", font=("Arial", 12), text_color="gray").pack(pady=(10, 15))
        
        scene_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        scene_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        customtkinter.CTkLabel(scene_frame, text="Models in Scene:", font=("Arial", 12, "bold"), anchor="w").pack(pady=(5, 5), fill="x")
        
        self.scene_listbox = customtkinter.CTkComboBox(scene_frame, values=["No models in scene"], state="readonly", height=35)
        self.scene_listbox.pack(pady=5, fill="x")
        self.scene_listbox.set("No models in scene")
        
        camera_frame = customtkinter.CTkFrame(parent)
        camera_frame.pack(pady=15, padx=20, fill="x")
        
        customtkinter.CTkLabel(camera_frame, text="Camera Controls", font=("Arial", 13, "bold")).pack(pady=10)
        
        camera_info = customtkinter.CTkTextbox(camera_frame, height=100, font=("Arial", 10), fg_color="gray20", wrap="word")
        camera_info.pack(pady=5, padx=10, fill="x")
        camera_info.insert("1.0", "CAMERA NAVIGATION:\n\n• MMB Drag - Orbit\n• Shift+MMB - Pan\n• Ctrl+MMB - Zoom\n• Mouse Wheel - Zoom\n• Arrow Keys - Pan")
        camera_info.configure(state="disabled")
    
    def _create_skybox_tab(self, parent):
        """Create Skybox tab"""
        customtkinter.CTkLabel(parent, text="Add immersive environment backgrounds", font=("Arial", 12), text_color="gray").pack(pady=(10, 15))
        
        type_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        type_frame.pack(pady=10, padx=20, fill="x")
        
        customtkinter.CTkLabel(type_frame, text="Skybox Type:", font=("Arial", 12, "bold"), anchor="w").pack(pady=(5, 5), fill="x")
        
        def on_type_change(choice):
            self.skybox_type = "equirectangular" if "Equirect" in choice else "cubemap"
        
        type_dropdown = customtkinter.CTkSegmentedButton(type_frame, values=["Equirectangular", "Cubemap"], 
                                                         command=lambda v: on_type_change(v), height=35)
        type_dropdown.pack(pady=5, fill="x")
        type_dropdown.set("Equirectangular")
        
        customtkinter.CTkButton(parent, text="📁 Browse Skybox Image", command=self.browse_skybox_equirect, height=40, 
                               font=("Arial", 13, "bold"), fg_color="#7C3AED", hover_color="#5B21B6").pack(pady=15, padx=20, fill="x")
        
        info_box = customtkinter.CTkTextbox(parent, height=100, font=("Arial", 10), fg_color="gray20", wrap="word")
        info_box.pack(pady=10, padx=20, fill="x")
        info_box.insert("1.0", "SKYBOX FORMATS:\n\nEquirectangular:\n• Single panoramic (2:1)\n• .hdr, .jpg, .png\n\nCubemap:\n• 6 separate images")
        info_box.configure(state="disabled")
        
        action_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        action_frame.pack(pady=15, padx=20, fill="x")
        
        customtkinter.CTkButton(action_frame, text="✓ Load Skybox", command=self.load_skybox, height=40, 
                               font=("Arial", 12, "bold"), fg_color="#059669", hover_color="#047857").pack(side="left", expand=True, fill="x", padx=(0, 5))
        customtkinter.CTkButton(action_frame, text="✗ Clear", command=self.clear_skybox, height=40, 
                               font=("Arial", 12, "bold"), fg_color="gray40", hover_color="gray30").pack(side="left", expand=True, fill="x", padx=(5, 0))
    
    def create_ui(self):
        """Create the UI window"""
        try:
            customtkinter.set_appearance_mode("dark")
            customtkinter.set_default_color_theme("blue")
            
            self.app = customtkinter.CTk()
            self.app.title("LWGE Control Panel")
            self.app.geometry("400x750")
            self.app.attributes('-topmost', True)
            
            customtkinter.CTkLabel(self.app, text="🎮 LWGE Engine", font=("Arial", 20, "bold")).pack(pady=(15, 5))
            customtkinter.CTkLabel(self.app, text="Lightweight Game Engine", font=("Arial", 11), text_color="gray").pack(pady=(0, 15))
            
            tabview = customtkinter.CTkTabview(self.app, width=360, height=600)
            tabview.pack(pady=10, padx=20, fill="both", expand=True)
            
            self._create_models_tab(tabview.add("📦 Models"))
            self._create_transform_tab(tabview.add("🔧 Transform"))
            self._create_scene_tab(tabview.add("🌍 Scene"))
            self._create_skybox_tab(tabview.add("🌌 Skybox"))
            
            self.app.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            def periodic_update():
                if self.running:
                    self.update_scene_list()
                    self.app.after(500, periodic_update)
            
            periodic_update()
            
            self.ui_started = True
            self.app.mainloop()

        except Exception as e:
            print(f"Error creating UI: {e}")
            self.running = False
    
    def on_closing(self):
        """Handle window close"""
        self.running = False
        if self.app:
            self.app.destroy()
    
    def start(self):
        """Start UI in separate thread"""
        ui_thread = threading.Thread(target=self.create_ui, daemon=True)
        ui_thread.start()
        return ui_thread
