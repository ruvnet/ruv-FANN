import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, realpath } from 'node:fs/promises';
import { resolve, sep, extname } from 'node:path';
import { chromium } from 'playwright';
const root=resolve(new URL('..',import.meta.url).pathname);
const importmap={imports:{'ruv-fann-wsp':'/node_modules/ruv-fann-wsp/src/index.js','ruv-fann':'/node_modules/ruv-fann/src/index.js','ruv-fann-wasm':'/node_modules/ruv-fann-wasm/src/index.js',
  ...Object.fromEntries(['lifecycle','search','huggingface','ruos'].map(p=>[`ruv-fann-wsp-${p}`,`/packages/${p}/src/index.js`]))}};

test('real Chromium executes surgical repair, diagnostics, search, Hub fixture and rollback',async()=>{
  const server=createServer(async(req,res)=>{
    try {
      const path=new URL(req.url,'http://localhost').pathname;
      if (path==='/') {res.setHeader('Content-Type','text/html');res.end(`<script type="importmap">${JSON.stringify(importmap)}</script>`);return;}
      const file=await realpath(resolve(root,`.${decodeURIComponent(path)}`));
      if(!file.startsWith(root+sep)) {res.writeHead(403).end();return;}
      res.setHeader('Content-Type',extname(file)==='.wasm'?'application/wasm':'text/javascript');
      res.end(await readFile(file));
    }catch{res.writeHead(404).end();}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try {
    browser=await chromium.launch({headless:true});
    const page=await browser.newPage();
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const results=await page.evaluate(async()=>{
      const {runScenario}=await import('/test/scenario.mjs');
      return [await runScenario(false),await runScenario(true)];
    });
    assert.deepEqual(errors,[]);
    assert.deepEqual(results.map(r=>r.backend),['wasm','wasm-simd']);
    assert.ok(results.every(r=>r.candidateMSE===0 && r.regressionRejected && r.compiledXor && r.rollbackRevision===3));
  }finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
});
