# Lens Try-On

Real-time virtual contact lens try-on in the browser, similar to Banuba's lens try-on.
It tracks the iris with MediaPipe Face Landmarker, fits the lens to the iris ellipse,
hides it under the eyelids, and blends it so natural shading and catchlights show through.

Everything runs locally. The MediaPipe runtime and the face model are bundled, so it works offline.

## Run

The app is a static folder. Serve it with any web server and open it in a browser:

```
cd lens-tryon-deploy
python3 -m http.server 8765
```

Then open `http://localhost:8765` and click **Start camera** or **Upload photo** (you can also drop a photo onto the preview).

Don't open `index.html` by double-clicking it. Browsers won't load the bundled model or the camera from a `file://` page.
No Python? Run `npx serve -l 8765` in this folder and open `http://localhost:8765`.

To try it on a phone, deploy the folder to an HTTPS host (see Deploying). The camera doesn't work over plain `http://` on another device.

## Folder layout

All files sit side by side in this folder:

```
index.html                         app (UI, tracking, lens rendering)
vision_bundle.mjs                  @mediapipe/tasks-vision 0.10.14 (Apache-2.0)
vision_wasm_internal.js / .wasm    MediaPipe WASM runtime (SIMD)
vision_wasm_nosimd_internal.js / .wasm   MediaPipe WASM runtime (no SIMD)
face_landmarker.task               MediaPipe Face Landmarker model (478 landmarks incl. iris)
```

If a bundled file is missing, the app falls back to jsDelivr, unpkg and Google's model host.
If all of those fail, the error appears under the video, and a **Load model file** button lets you pick a `.task` file by hand.

## Performance

Face tracking runs on the GPU when the browser has a real one. On machines where WebGL is software-rendered
(VMs, remote desktops, blocked drivers) the app switches to the CPU path, which is several times faster there.
Add `?delegate=GPU` or `?delegate=CPU` to the URL to force one.

## Custom lenses

Click **Upload your own lens image** and pick a product photo of a single lens, ideally front-on and centred.
White backgrounds and a white pupil area are removed automatically. Turn off
**Remove white background** if the lens itself is white.

## Recording issues

Open **Report a tracking issue** at the bottom of the side panel. With the camera on, type what you're testing in the note box (for example, "lens shakes when I tilt my head").
Then click **Record** and reproduce the problem. Press **M** (or click **⚑ Mark moment**) each time the issue appears, then click **Stop**.
Recording stops on its own after 60 seconds. The note shows on the video, goes in the log and in the file name. Each mark flashes on the video and is logged with its time.
Two files land in your Downloads folder:

- `lens-recording-<time>.webm`: the raw camera (left) and the result (right), side by side. The corner shows the frame number, time, fps and processing time per frame.
- `lens-recording-<time>.json`: a per-frame tracking log with eye centre, pupil-locked centre, iris radius, eye openness and pupil size, plus the lens and slider settings.

Send both files to Claude for analysis. The left half is the raw camera, so the issue can be traced to tracking or to rendering.

## Deploying

The folder is a static site. Upload it as-is to any static host (Netlify, Vercel, GitHub Pages, S3).
The camera needs HTTPS when the page isn't on localhost.
