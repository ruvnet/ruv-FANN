import test from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {chromium} from 'playwright';
import {renderConstellationDashboard} from 'ruvnet-constellation-telemetry';
import {constellationFixture} from './constellation-fixture.mjs';
test('standalone dashboard filters real report shapes without external requests or executable metadata',async()=>{
  const fixture=await constellationFixture({adoption:true,hostile:true});
  const unsampled=await constellationFixture();
  const pages=new Map([['/',renderConstellationDashboard(fixture.report)],['/unsampled',renderConstellationDashboard(unsampled.report)]]);
  const server=createServer((req,res)=>{if(pages.has(req.url)){res.writeHead(200,{'Content-Type':'text/html'});res.end(pages.get(req.url));}else res.writeHead(404).end();});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
  try {
    browser=await chromium.launch({headless:true});const page=await browser.newPage(),errors=[],external=[];
    page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(!r.url().startsWith('http://127.0.0.1:'))external.push(r.url());});
    const origin=`http://127.0.0.1:${server.address().port}`;await page.goto(origin);
    await page.waitForSelector('#package-table tr[data-package]');
    assert.equal(await page.locator('#package-table tr[data-package]').count(),24);
    assert.match(await page.locator('#attach-rate').innerText(),/66\.7|66\.67|67/);
    assert.equal(await page.evaluate(()=>window.constellationPwned),undefined);
    assert.equal(await page.locator('img').count(),0);
    const monthly=await page.locator('#download-total').innerText();await page.selectOption('#period-filter','7');
    assert.notEqual(await page.locator('#download-total').innerText(),monthly);
    await page.selectOption('#family-filter','ruv-fann');await page.selectOption('#role-filter','satellite');
    assert.equal(await page.locator('#package-table tr[data-package]').count(),9);
    await page.fill('#package-search','loop');assert.equal(await page.locator('#package-table tr[data-package]').count(),1);
    assert.equal(await page.locator('#package-table tr[data-package]').getAttribute('data-package'),'ruv-fann-wsp-loop');
    assert.match(await page.locator('#package-table').innerText(),/Local candidate/);
    await page.fill('#package-search','no-such-package');assert.match(await page.locator('#package-table').innerText(),/No packages match/);
    await page.goto(origin+'/unsampled');assert.match(await page.locator('#attach-rate').innerText(),/Not measured/);
    assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  } finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
});
