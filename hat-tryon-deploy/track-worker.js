/* Face tracking off the main thread (used on phones): the camera picture is drawn at full rate
   while this works on the latest frame. Classic worker: MediaPipe loads its wasm with importScripts. */
let lmk = null;
self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    let lastErr = "no source";
    for (const bundle of m.bundles) {
      try {
        const { FaceLandmarker, FilesetResolver } = await import(bundle);
        for (const wasm of m.wasm) {
          let fs;
          try { fs = await FilesetResolver.forVisionTasks(wasm); } catch (err) { lastErr = "wasm: " + err; continue; }
          for (const delegate of ["GPU", "CPU"]) {
            try {
              lmk = await FaceLandmarker.createFromOptions(fs, {
                baseOptions: { modelAssetBuffer: m.model, delegate }, runningMode: "VIDEO", numFaces: m.numFaces,
                outputFacialTransformationMatrixes: true,
                minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5 });
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
    const t0 = performance.now();
    let res = null;
    try {
      const r = lmk.detectForVideo(m.bitmap, m.t);
      res = { faceLandmarks: r.faceLandmarks,
        facialTransformationMatrixes: (r.facialTransformationMatrixes || []).map(x => ({ data: Array.from(x.data) })) };
    } catch (err) { self.postMessage({ type: "error", msg: String(err && err.message || err) }); }
    m.bitmap.close();
    self.postMessage({ type: "res", id: m.id, key: m.key, t: m.t, res, ms: performance.now() - t0 });
  }
};
