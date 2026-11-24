# Hat Try-On

Real-time virtual hat try-on in the browser. It tracks your head (position, scale, roll, yaw, pitch) and puts a hat on your live camera feed or an uploaded photo. Everything runs on your device; nothing is uploaded.

## Run

```bash
python3 -m http.server 8765
```

Open http://localhost:8765, then click **Start camera** or **Upload photo**. It must be served over `localhost` or HTTPS (the camera and the models do not load from `file://`).

The MediaPipe runtime and models are bundled in `vendor/` and `models/`, so it works offline. If they are missing it falls back to the CDN.

## Features

- 6 built-in 3D hats: beanie, cap, fedora, sun hat, cowboy, top hat, plus a photo-scanned boonie. Each can be recoloured.
- Upload your own hat image (PNG/JPG product photo). A plain background is removed automatically; a transparent PNG is used as is.
- Fit controls: size, height on head, left/right, tilt, crown height, tip forward/back. You can also drag the hat and scroll to resize.
- The hat is lit like the picture: brightness, colour tone and light side are read from the face, and the hat shades the forehead.
- The hat wraps around an invisible 3D head, so its far side is hidden. Brims are also hidden behind hair and face using a segmentation mask, and the hat is sized to go round the hair: width and crown height follow the measured hair.
- Picture grain is measured on the face and added to the hat so it matches the photo or camera.
- Save snapshot, hold to compare, tracking overlay.

## How it works

| Step | What happens |
|---|---|
| Tracking | MediaPipe Face Landmarker gives 478 landmarks and a head transformation matrix. Head centre and width come from the ear landmarks, the hat line from the top-of-forehead landmark, and roll/yaw/pitch from the matrix. |
| Stability | One-Euro filters smooth position, scale and angles: steady when you are still, responsive when you move. If tracking drops for a moment (blink, blur, a hand), the hat holds its place for 250 ms instead of vanishing. |
| Who wears it | Up to 3 faces are tracked. The hat stays with the person already wearing it; at the start, or when that person leaves, it goes to the biggest face nearest the middle, so people walking behind you don't steal it. |
| Camera angle | Laptop webcams sit below the eyes and head pitch from one camera is noisy, so the hat's view is limited to 12° from below and 28° from above. Past that, the hat turns about the front of its band: it stays on the forehead and the far side of the brim can't swing down past the face. |
| Placement | Hats are 3D models (Three.js) in head-width units. They are attached to a head group that follows the tracked position, scale and rotation, so they turn and tilt in true perspective. |
| Occlusion | An invisible head shape writes depth only, hiding the parts of the hat behind the head. The MediaPipe selfie segmenter adds a hair/face mask for brims and measures the hair width to size the hat. |
| Rendering | Three.js draws the frame, the hat with fabric textures and estimated lighting, and a soft shade band under the hat edge. |

## Custom hats

Best results come from a front-on product photo on a plain background. Use **Tolerance** if parts of the background remain or parts of the hat vanish, and **Tuck the back behind the head** to hide the lower part of the hat (the far side of the brim) behind the forehead and hair. A custom hat is a flat picture wrapped over the head, so it looks less real than the built-in 3D hats.

## Files

- `index.html` — UI
- `app.js` — tracking, placement, rendering, controls
- `hats.js` — built-in 3D hats and fabric textures, generated procedurally
- `vendor/`, `models/` — Three.js, MediaPipe tasks-vision runtime, face landmarker and selfie segmentation models

## 3D hat models

The "Boonie" hat is a photo-scanned 3D model (Poly Haven "Fisherman's Hat", CC0) with real fabric textures, and is the most realistic hat here.
Use "Upload a 3D hat model (.glb)" to try your own: the model is scaled to the head automatically. It should be y-up with the front
towards +z, a single self-contained .glb, not Draco/Meshopt compressed.
