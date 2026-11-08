'use strict';
// Drive real Rust IndexedDB calls compiled from indexeddb_browser_smoke.rs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {chromium} = require('playwright');
(async () => {
  if (!process.argv[2]) throw new Error('Pass the wasm-bindgen output directory');
  const directory = path.resolve(process.argv[2]);
  const assets = new Map([
    ['/persistence_smoke.js',['application/javascript',fs.readFileSync(path.join(directory,'persistence_smoke.js'))]],
    ['/persistence_smoke_bg.wasm',['application/wasm',fs.readFileSync(path.join(directory,'persistence_smoke_bg.wasm'))]],
  ]);
  const server=http.createServer((request,response)=>{
    if(request.url==='/') {response.setHeader('Content-Type','text/html');response.end('<!doctype html><title>IndexedDB integration</title>');}
    else if(assets.has(request.url)) {const [type,data]=assets.get(request.url);response.setHeader('Content-Type',type);response.end(data);}
    else response.writeHead(404).end();
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try {
    const options={headless:true};if(process.env.CHROMIUM_BIN) options.executablePath=process.env.CHROMIUM_BIN;
    browser=await chromium.launch(options);
    const page=await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const result=await page.evaluate(async()=>{
      const module=await import('/persistence_smoke.js');await module.default();
      return JSON.parse(await module.run_persistence_smoke());
    });
    assert.equal(result.checks.length,7);assert.equal(result.database_deleted,true);assert.equal(result.claim_winners,1);
    console.log(JSON.stringify(result));
  } finally {if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
