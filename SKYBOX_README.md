# Skybox System Documentation

## Overview
LWGE now supports skybox rendering for immersive 3D environments. You can use either **equirectangular** images (single panoramic image) or **cubemap** format (6 separate images).

## Supported Formats

### 1. Equirectangular (Recommended)
- **Format**: Single panoramic image
- **File types**: `.hdr`, `.jpg`, `.png`, `.exr`
- **Aspect ratio**: 2:1 (e.g., 4096x2048, 2048x1024)
- **Source**: HDRIs from sites like Poly Haven, HDRI Haven

### 2. Cubemap
- **Format**: 6 separate images (one for each cube face)
- **Required faces**: right, left, top, bottom, front, back
- **File types**: `.png`, `.jpg`, `.bmp`, `.tga`
- **All faces must**: have same dimensions and be square (e.g., 1024x1024)

## How to Use

### Via UI:
1. Open the Control Panel
2. Scroll to the **Skybox** section
3. Select skybox type:
   - **Equirectangular (Single Image)** - default, easiest option
   - **Cubemap (6 Images)** - for pre-separated cubemap textures
4. Click **"📁 Browse Equirect Image"** and select your HDR/panoramic image
5. Click **"✓ Load Skybox"** to apply
6. Click **"✗ Clear"** to remove the skybox

### Programmatically:
```python
# Load equirectangular skybox
engine.load_skybox_equirectangular("path/to/skybox.hdr")

# Load cubemap skybox
engine.load_skybox_cubemap({
    'right': 'path/to/right.png',
    'left': 'path/to/left.png',
    'top': 'path/to/top.png',
    'bottom': 'path/to/bottom.png',
    'front': 'path/to/front.png',
    'back': 'path/to/back.png'
})

# Or with a list (order: right, left, top, bottom, front, back)
engine.load_skybox_cubemap([
    'right.png', 'left.png', 'top.png', 
    'bottom.png', 'front.png', 'back.png'
])

# Clear skybox
engine.clear_skybox()
```

## Where to Get Skybox Images

### Free HDR Skyboxes:
- **Poly Haven** (https://polyhaven.com/hdris) - Highest quality, CC0 license
- **HDRI Haven** (https://hdrihaven.com) - Free 8K HDRIs
- **HDR Labs** (http://www.hdrlabs.com/sibl/archive.html)

### Tips:
- Higher resolution = better quality (but larger file size)
- HDR format (.hdr) provides better lighting than JPG/PNG
- Outdoor scenes work great for most 3D environments
- Indoor HDRIs good for architectural visualization

## Technical Details

### Rendering
- Skybox renders first (after clear, before models)
- Uses depth trick: `gl_Position.z = gl_Position.w` (always at far plane)
- View matrix translation stripped to keep skybox centered on camera
- Depth test set to `LEQUAL` during skybox render

### Performance
- Very efficient - single draw call with 36 vertices
- Mipmapping enabled for smooth appearance
- No impact on model rendering performance

### Shader
- Cubemap: Direct texture sampling with direction vector
- Equirectangular: Direction vector converted to UV coordinates using spherical mapping

## Troubleshooting

**Skybox not appearing:**
- Check console for error messages
- Verify file paths are correct
- Ensure image files are valid and readable

**Skybox looks wrong:**
- For cubemap: verify all 6 faces are correct orientation
- For equirectangular: ensure 2:1 aspect ratio
- Check if HDR vs LDR format is appropriate

**Performance issues:**
- Reduce skybox resolution (e.g., 2K instead of 8K)
- Use JPG instead of HDR for lower quality/faster loading
