import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { evaluateRequest } from './src/api.js';
const files = { '/': ['index.html', 'text/html'], '/app.js': ['app.js', 'application/javascript'],
  '/style.css': ['style.css', 'text/css'], '/runtime/core.wasm': ['core.wasm', 'application/wasm'], '/runtime/simd.wasm': ['simd.wasm', 'application/wasm'] };
export function createAppServer() {
  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      if (url.pathname === '/api/evaluate') {
        const body = []; let size = 0;
        for await (const chunk of req) { size += chunk.length; if (size > 1024) { res.writeHead(413).end(); return; } body.push(chunk); }
        const response = await evaluateRequest(new Request(url, { method: req.method, headers: req.headers,
          ...(req.method === 'POST' ? { body: Buffer.concat(body) } : {}) }));
        res.writeHead(response.status, Object.fromEntries(response.headers)); res.end(Buffer.from(await response.arrayBuffer())); return;
      }
      const file = files[url.pathname]; if (!file) { res.writeHead(404).end(); return; }
      res.writeHead(200, { 'content-type': file[1] }); res.end(await readFile(new URL('./dist/' + file[0], import.meta.url)));
    } catch { res.writeHead(500).end(); }
  });
}
if (process.argv[1] === new URL(import.meta.url).pathname) {
  createAppServer().listen(Number(process.env.PORT || 4173), '0.0.0.0', () => console.log('Playground listening on http://localhost:' + (process.env.PORT || 4173)));
}
