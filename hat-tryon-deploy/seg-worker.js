/* Hair / face / ear segmentation off the main thread (used on phones).
   A classic worker on purpose: MediaPipe loads its wasm with importScripts, which module workers don't have. */
let seg = null;
self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    let lastErr = "no source";
    for (const bundle of m.bundles) {
      try {
        const { ImageSegmenter, FilesetResolver } = await import(bundle);
        for (const wasm of m.wasm) {
          let fs;
          try { fs = await FilesetResolver.forVisionTasks(wasm); } catch (err) { lastErr = "wasm: " + err; continue; }
          for (const delegate of ["CPU", "GPU"]) {   // CPU first: the GPU is busy with the face tracker and the hat
            try {
              seg = await ImageSegmenter.createFromOptions(fs, {
                baseOptions: { modelAssetBuffer: m.model, delegate }, runningMode: "VIDEO",
                outputCategoryMask: false, outputConfidenceMasks: true });
              self.postMessage({ type: "ready", delegate }); return;
            } catch (err) { lastErr = delegate + ": " + (err && err.message || err); }
          }
        }
      } catch (err) { lastErr = "import: " + (err && err.message || err); }
    }
    self.postMessage({ type: "fail", msg: lastErr });
    return;
  }
  if (m.type === "frame") {
    let out = null;
    try {
      seg.segmentForVideo(m.bitmap, m.t, (r) => {
        const k = r.confidenceMasks; if (!k || k.length < 4) return;
        out = { hair: k[1].getAsFloat32Array().slice(), face: k[3].getAsFloat32Array().slice(), skin: k[2].getAsFloat32Array().slice(), w: k[1].width, h: k[1].height };
      });
    } catch (err) { self.postMessage({ type: "error", msg: String(err && err.message || err) }); }
    m.bitmap.close();
    if (out) self.postMessage({ type: "mask", id: m.id, seg: out }, [out.hair.buffer, out.face.buffer, out.skin.buffer]);
    else self.postMessage({ type: "mask", id: m.id, seg: null });
  }
};
