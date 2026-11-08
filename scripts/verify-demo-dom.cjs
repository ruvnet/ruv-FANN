'use strict';
// Exercise the actual rendering methods with hostile text in a real DOM.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');

(async () => {
  const read = file => fs.readFileSync(path.join(__dirname, '..', file), 'utf8');
  const cuda = read('cuda-wasm/demo/demo.js');
  const swarm = read('ruv-swarm/examples/web_demo/demo.js');
  const declarations = cuda.slice(cuda.indexOf('class CudaRustWasmDemo'), cuda.lastIndexOf('// Initialize demo')) +
    '\n' + swarm.slice(swarm.indexOf('class SwarmDemo'), swarm.lastIndexOf('// Initialize demo')) +
    '\nglobalThis.renderClasses = {CudaRustWasmDemo, SwarmDemo};';
  const browser = await chromium.launch({headless: true,
    ...(process.env.CHROMIUM_BIN ? {executablePath: process.env.CHROMIUM_BIN} : {})});
  try {
    const page = await browser.newPage();
    await page.setContent('<!doctype html><div id="output-area"></div><div id="agentList"></div>');
    await page.addScriptTag({content: declarations});
    const result = await page.evaluate(() => {
      const payload = '<img src=x onerror="globalThis.injected=true"><svg onload="globalThis.injected=true">';
      const cuda = Object.create(renderClasses.CudaRustWasmDemo.prototype);
      const output = document.getElementById('output-area');
      output.innerHTML = cuda.formatAnalysis({memoryPattern: payload, threadUtilization: payload,
        sharedMemoryUsage: payload, registerUsage: payload, suggestions: [payload], occupancy: 0.5});
      if (output.querySelector('img,svg,script') || !output.textContent.includes(payload)) throw new Error('Analysis text became markup');
      cuda.displayError(payload);
      if (output.querySelector('img,svg,script') || !output.textContent.includes(payload)) throw new Error('Error text became markup');
      const swarm = Object.create(renderClasses.SwarmDemo.prototype);
      swarm.agents = new Map([[payload, {id: payload, type: payload}]]);
      swarm.updateAgentList();
      const agents = document.getElementById('agentList');
      if (agents.querySelector('img,svg,script') || !agents.textContent.includes(payload)) throw new Error('Agent text became markup');
      swarm.agents.clear(); swarm.updateAgentList();
      if (agents.textContent !== 'No agents spawned' || globalThis.injected) throw new Error('Invalid empty list or executed payload');
      return {checks: 4, injected: !!globalThis.injected};
    });
    assert.deepEqual(result, {checks: 4, injected: false});
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
