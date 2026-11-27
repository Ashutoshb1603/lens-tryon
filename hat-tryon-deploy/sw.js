/* Offline-first cache for the big, rarely changing files: after the first visit, the models, the MediaPipe engine,
   three.js and the hats load from the device instead of the network. They are refreshed in the background
   (stale-while-revalidate), so an updated file arrives on the next visit. The page and the app's code are
   fetched from the network first (they carry ?v= version stamps), with the cache only as an offline fallback. */
const CACHE = "hat-tryon-v1";
const BIG = /\/(models|vendor|icons)\//;

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", e => e.waitUntil((async () => {
  for (const k of await caches.keys()) if (k !== CACHE && k.startsWith("hat-tryon-")) await caches.delete(k);
  await self.clients.claim();
})()));

self.addEventListener("fetch", e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== location.origin || req.headers.has("range")) return;
  if (BIG.test(url.pathname)) e.respondWith(staleWhileRevalidate(e, req));
  else e.respondWith(networkFirst(req));
});

async function staleWhileRevalidate(e, req) {
  const cache = await caches.open(CACHE), hit = await cache.match(req, { ignoreSearch: true });
  const fresh = fetch(req).then(res => { if (res.ok && res.status === 200) cache.put(req, res.clone()); return res; }).catch(() => null);
  if (hit) { e.waitUntil(fresh); return hit; }
  return (await fresh) || new Response("offline", { status: 503 });
}
async function networkFirst(req) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(req);
    if (res.ok && res.status === 200) cache.put(req, res.clone());
    return res;
  } catch (err) {
    return (await cache.match(req)) || (await cache.match(req, { ignoreSearch: true })) || new Response("offline", { status: 503 });
  }
}
