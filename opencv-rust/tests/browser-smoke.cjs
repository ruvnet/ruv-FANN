'use strict';
// Build opencv-wasm with wasm-pack --target web --out-name opencv_wasm first.
// Then: node tests/browser-smoke.cjs /path/to/generated/pkg
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {chromium} = require('playwright');

(async () => {
  if (!process.argv[2]) throw new Error('Pass the generated wasm-pack package directory');
  const directory = path.resolve(process.argv[2]);
  const assets = new Map([
    ['/opencv_wasm.js', ['application/javascript', fs.readFileSync(path.join(directory, 'opencv_wasm.js'))]],
    ['/opencv_wasm_bg.wasm', ['application/wasm', fs.readFileSync(path.join(directory, 'opencv_wasm_bg.wasm'))]],
  ]);
  const server = http.createServer((request, response) => {
    if (request.url === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><title>OpenCV WASM browser tests</title>');
    } else if (assets.has(request.url)) {
      const [type, bytes] = assets.get(request.url);
      response.setHeader('Content-Type', type);
      response.end(bytes);
    } else {
      response.writeHead(404).end();
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    const launchOptions = {headless: true};
    if (process.env.CHROMIUM_BIN) launchOptions.executablePath = process.env.CHROMIUM_BIN;
    browser = await chromium.launch(launchOptions);
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result = await page.evaluate(async () => {
      const cv = await import('/opencv_wasm.js');
      await cv.default();
      const ensure = (condition, message) => { if (!condition) throw new Error(message); };
      const checks = [];
      const rgba = new Uint8ClampedArray([255,0,0,255, 0,255,0,255, 0,0,255,255, 255,255,255,255]);
      const source = cv.WasmMat.from_image_data(new ImageData(rgba, 2, 2));
      const retainedPixels = source.to_image_data();
      const gray = Array.from(retainedPixels.data).filter((_, i) => i % 4 === 0);
      ensure(JSON.stringify(gray) === JSON.stringify([77,149,29,255]), `RGBA conversion: ${gray}`);
      checks.push('RGBA to grayscale');
      const resized = cv.resize(source, 4, 2);
      const pixels = Array.from(resized.to_image_data().data).filter((_, i) => i % 4 === 0);
      ensure(JSON.stringify(pixels) === JSON.stringify([77,77,149,149,29,29,255,255]), `Resize: ${pixels}`);
      checks.push('nearest-neighbor pixels');
      const roi = source.roi(1, 0, 1, 2);
      ensure(roi.to_image_data().data[0] === 149 && roi.to_image_data().data[4] === 255, 'ROI pixels');
      checks.push('copied ROI');
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = 2;
      canvas.getContext('2d').putImageData(new ImageData(rgba, 2, 2), 0, 0);
      const canvasMat = cv.mat_from_canvas(canvas);
      const output = document.createElement('canvas');
      output.width = output.height = 2;
      cv.mat_to_canvas(canvasMat, output);
      ensure(output.getContext('2d').getImageData(0,0,2,2).data[0] === 77, 'Canvas roundtrip');
      checks.push('real canvas import/export');
      let rejected = false;
      try { new cv.WasmMat(-1, 2); } catch (error) { rejected = String(error).length > 0; }
      ensure(rejected, 'Invalid matrix dimensions must throw');
      rejected = false;
      try { source.roi(99,0,1,1); } catch (error) { rejected = String(error).length > 0; }
      ensure(rejected, 'Invalid ROI must throw');
      checks.push('invalid dimensions/ROI');
      const tallRgba = new Uint8ClampedArray(65536 * 4);
      tallRgba.set([233,233,233,255], tallRgba.length-4);
      const tall = cv.WasmMat.from_image_data(new ImageData(tallRgba,1,65536));
      const taller = cv.resize(tall,1,65537);
      const last = taller.to_image_data().data.at(-4);
      ensure(last === 233, `WASM32 resize overflow regression: ${last}`);
      checks.push('WASM32 tall-image overflow');
      const capabilities = cv.check_capabilities();
      ensure(typeof capabilities.simd === 'boolean' && capabilities.threads === false, 'Capability flags');
      checks.push('compiled capability flags');
      const version = cv.get_version();
      for (const image of [source,resized,roi,canvasMat,tall,taller]) image.free();
      ensure(retainedPixels.data.length === 16 && retainedPixels.data[0] === 77 && retainedPixels.data[4] === 149,
        'Exported ImageData must own its pixels after Rust buffers drop or WASM memory grows');
      checks.push('exported pixel lifetime');
      return {checks, gray, resized: pixels, tallLastPixel: last, version};
    });
    assert.equal(result.checks.length, 8);
    console.log(JSON.stringify(result));
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
