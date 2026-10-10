// FoodLens-AI - Progressive Web App Service Worker
const CACHE_NAME = 'foodlens-v2.0.0';
const STATIC_ASSETS = [
  '/',
  '/scan',
  '/features',
  '/about',
  '/contact',
  '/chatbot',
  '/static/css/style.css',
  '/static/js/main.js',
  '/static/js/scan.js',
  '/static/image/logo-symbol.svg',
  '/static/image/favicon.svg',
  '/static/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS).catch((err) => {
        console.warn('Pre-cache warning for offline assets:', err);
      });
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Pass dynamic API and upload requests directly to network
  if (
    event.request.url.includes('/api/') ||
    event.request.url.includes('/predict') ||
    event.request.url.includes('/feedback') ||
    event.request.url.includes('/chat') ||
    event.request.method !== 'GET'
  ) {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cached) => {
      return (
        cached ||
        fetch(event.request).catch(() => {
          if (event.request.destination === 'document') {
            return caches.match('/');
          }
        })
      );
    })
  );
});
