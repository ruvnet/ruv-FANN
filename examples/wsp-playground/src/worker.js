import { evaluateRequest } from './api.js';
import assets from '../dist/site-assets.js';
import coreBase64 from '../dist/core-base64.txt';
import simdBase64 from '../dist/simd-base64.txt';

const binaryCache = new Map();
const headers = {
  'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
  'Content-Security-Policy': "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
};
export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (url.search) return new Response('Query parameters are not supported', { status: 400, headers });
    if (url.pathname === '/api/evaluate') {
      const origin = request.headers.get('origin');
      if (origin && origin !== url.origin) return new Response('Origin rejected', { status: 403, headers });
      const result = await evaluateRequest(request);
      const response = new Response(result.body, result); for (const [k, v] of Object.entries(headers)) response.headers.set(k, v);
      return response;
    }
    if (!['GET', 'HEAD'].includes(request.method)) return new Response('Method not allowed', { status: 405, headers });
    if (url.pathname === '/health') return Response.json({ ok: true, service: 'wsp-live-lab', runtime: 'static-wasm', state: 'stateless' }, { headers });
    if (['/runtime/core.wasm', '/runtime/simd.wasm'].includes(url.pathname)) {
      if (!binaryCache.has(url.pathname)) {
        const text = atob(url.pathname.includes('simd') ? simdBase64 : coreBase64);
        binaryCache.set(url.pathname, Uint8Array.from(text, ch => ch.charCodeAt(0)));
      }
      return new Response(request.method === 'HEAD' ? null : binaryCache.get(url.pathname), {
        headers: { ...headers, 'Content-Type': 'application/wasm', 'Cache-Control': 'public, max-age=3600' },
      });
    }
    const asset = assets[url.pathname === '/' ? '/index.html' : url.pathname];
    if (!asset) return new Response('Not found', { status: 404, headers });
    return new Response(request.method === 'HEAD' ? null : asset.body, { headers: { ...headers,
      'Content-Type': asset.type, 'Cache-Control': 'no-cache' } });
  },
};
