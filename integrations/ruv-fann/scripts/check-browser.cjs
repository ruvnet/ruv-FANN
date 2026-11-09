/** Real browser acceptance; Playwright is supplied by the development host. */
const { chromium } = require('playwright');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');

(async () => {
  if (process.argv.length !== 4 || process.argv[2] !== '--output') throw new Error('Use --output NEW_EXTERNAL_JSON');
  const helper = await import('../test/e2e/fixture.mjs');
  const output = await helper.externalEvidencePath(process.argv[3]);
  const before = await helper.sourceSnapshot();
  const pageHtml = `<!doctype html><meta charset="utf-8"><title>ruv-FANN integration acceptance</title>
    <script type="importmap">{"imports":{"@ruvector/wasm":"/integrations/ruv-fann/node_modules/@ruvector/wasm/pkg/ruvector_wasm.js"}}</script>
    <script type="module">
    import { createEncoder, createNeuralIndex } from '/integrations/ruv-fann/src/index.mjs';
    import { createRuvectorStore } from '/integrations/ruv-fann/src/ruvector.mjs';
    window.runIntegration = async () => {
      const check = (condition, message) => { if (!condition) throw new Error(message); };
      const read = async url => { const r = await fetch(url); if (!r.ok) throw new Error(url); return r; };
      const wasmBytes = new Uint8Array(await (await read('/integrations/ruv-fann/node_modules/@ruvector/wasm/pkg/ruvector_wasm_bg.wasm')).arrayBuffer());
      const fixture = await (await read('/ruv-swarm/crates/ruv-swarm-wasm/tests/fixtures/prepared-dense.json')).json();
      const inputs = Float64Array.from({length:131*3}, (_,i) => fixture.input[i%3] + ((Math.floor(i/3)*17+i%3*11)%67-33)/16);
      const cases = [];
      for (const variant of ['core','simd']) {
        const fann = await import('/ruv-swarm/npm/wasm/ruv_swarm_wasm.js?browser-variant='+variant);
        const name = variant==='simd'?'ruv_swarm_simd.wasm':'ruv_swarm_wasm_bg.wasm';
        await fann.default({module_or_path:await (await read('/ruv-swarm/npm/wasm/'+name)).arrayBuffer()});
        check(JSON.parse(fann.get_features()).simd_support === (variant==='simd'), 'FANN variant');
        const prepared = fann.WasmPreparedNetwork.from_network_json(JSON.stringify(fixture.network));
        try {
          const encoder=createEncoder(prepared,{batchRows:7});
          try {
            const actual=encoder.encode(inputs);
            for(let row=0;row<131;row++){
              const expected=prepared.run(inputs.subarray(row*3,row*3+3));
              for(let j=0;j<2;j++)check(Object.is(actual[row*2+j],expected[j]),'FANN batch reference');
            }
          } finally {encoder.close();}
          for(const useHnsw of [false,true]){
            const store=await createRuvectorStore({wasmBytes,dimensions:prepared.output_size,metric:'euclidean',useHnsw,maxEntries:128});
            const index=createNeuralIndex({prepared,store,modelId:'native-dense-browser',batchRows:7,maxEntries:128});
            try{
              check(index.indexType===(useHnsw?'hnsw':'flat'),'RuVector backend');
              for(let row=0;row<33;row++)index.add({id:'row-'+row,input:inputs.subarray(row*3,row*3+3),metadata:{row}});
              const found=index.search(inputs.subarray(21,24),{k:5});
              check(found.length===5&&found[0].id==='row-7'&&found[0].distance===0,'Neural nearest neighbor');
              check(found[0].metadata.row===7,'Metadata');
              check(index.delete('row-7')===true,'Delete');
              check(!index.search(inputs.subarray(21,24),{k:5}).some(x=>x.id==='row-7'),'Deleted ID');
              cases.push({variant,indexType:index.indexType,records:33,outputMaximumAbsoluteError:0,retrievalActualId:found[0].id,queryDistance:found[0].distance});
            }finally{index.close();}
          }
          check(prepared.run(Float64Array.from(fixture.input)).length===2,'Prepared ownership');
        }finally{prepared.free();}
      }
      return {schema:'ruv-fann.integration-browser/1',status:'passed',actualNeuralInference:true,actualRuVector:true,coreAndSimd:true,cases};
    };
    </script>`;
  const allowed = ['/integrations/ruv-fann/src/', '/integrations/ruv-fann/node_modules/@ruvector/wasm/pkg/', '/ruv-swarm/npm/wasm/'];
  const fixturePath = '/ruv-swarm/crates/ruv-swarm-wasm/tests/fixtures/prepared-dense.json';
  const server = http.createServer(async (req, res) => {
    try {
      const name = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
      if (name === '/') { res.setHeader('Content-Type', 'text/html'); res.end(pageHtml); return; }
      if (!allowed.some(prefix => name.startsWith(prefix)) && name !== fixturePath) { res.writeHead(404); res.end(); return; }
      const target = await fs.realpath(path.join(helper.repo, name));
      if (!target.startsWith(helper.repo) || !(await fs.stat(target)).isFile()) throw new Error('Invalid static file');
      const bytes = await fs.readFile(target);
      if (bytes.length > 8388608) throw new Error('Static file exceeds limit');
      res.setHeader('Content-Type', name.endsWith('.wasm') ? 'application/wasm' : name.endsWith('.json') ? 'application/json' : 'text/javascript');
      res.end(bytes);
    } catch { res.writeHead(404); res.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium', headless: true,
      args: ['--no-sandbox', '--disable-dev-shm-usage'] });
    const page = await browser.newPage();
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    const origin = 'http://127.0.0.1:' + server.address().port;
    await page.route('**/*', route => route.request().url().startsWith(origin + '/') ? route.continue() : route.abort());
    await page.goto(origin);
    await page.waitForFunction(() => typeof window.runIntegration === 'function');
    const result = await page.evaluate(() => window.runIntegration());
    if (errors.length) throw new Error(errors.join('\n'));
    const after = await helper.sourceSnapshot();
    if (JSON.stringify(before) !== JSON.stringify(after)) throw new Error('Source changed during browser check');
    Object.assign(result, { chromium: browser.version(), inputsBefore: before, inputsAfter: after, unchanged: true,
      tuningScope: 'Darwin tuning is Node-only and is exercised by separate real Node acceptance.' });
    await fs.writeFile(output, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
    console.log(JSON.stringify({ status: result.status, chromium: result.chromium, cases: result.cases.length, output }));
  } finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error.message); process.exitCode = 1; });
