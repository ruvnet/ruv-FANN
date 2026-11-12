import test from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
import { createAppServer } from '../server.mjs';
const external = process.env.PLAYGROUND_URL;

test('real Chromium: desktop/mobile, rejection, edge agreement, promotion, rollback, export and benchmark', { timeout: 120000 }, async () => {
  let server;
  if (!external) { server = createAppServer(); await new Promise(resolve => server.listen(0, '127.0.0.1', resolve)); }
  const url = external || `http://127.0.0.1:${server.address().port}`;
  let browser;
  try { browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] }); } catch (error) { if (server) await new Promise(resolve => server.close(resolve)); throw error; }
  await mkdir('.artifacts', { recursive: true });
  const observations = [];
  try {
    for (const viewport of [{ width: 1440, height: 1100 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport, deviceScaleFactor: 1 }); const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(url); await page.waitForSelector('html[data-ready="true"]');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      assert.equal(await page.locator('#promote').isDisabled(), true);
      const hash = await page.locator('#checkpoint').innerText();
      await page.locator('#shortcut').click(); await page.locator('#evaluate').click();
      await page.waitForFunction(() => document.getElementById('gate-status').textContent === 'REJECTED');
      assert.equal(await page.locator('#promote').isDisabled(), true);
      assert.equal(await page.locator('#checkpoint').innerText(), hash);
      await page.locator('#repair').click(); await page.locator('#evaluate').click();
      await page.waitForFunction(() => document.getElementById('gate-status').textContent === 'READY TO PROMOTE');
      assert.match(await page.locator('#receipt-summary').innerText(), /edge gates agree/);
      await page.screenshot({ path: `.artifacts/${viewport.width}-robot.png`, fullPage: true });
      await page.locator('#promote').click(); await page.waitForFunction(() => document.getElementById('revision').textContent === 'REV 02');
      assert.notEqual(await page.locator('#checkpoint').innerText(), hash);
      await page.locator('#rollback').click(); await page.waitForFunction(() => document.getElementById('revision').textContent === 'REV 03');
      assert.equal(await page.locator('#checkpoint').innerText(), hash);
      const downloadEvent = page.waitForEvent('download'); await page.locator('#export').click(); const download = await downloadEvent;
      assert.equal(download.suggestedFilename(), 'wsp-robot-experiment.json');
      await download.saveAs(`.artifacts/${viewport.width}-receipt.json`);
      await page.locator('[data-tab="room"]').click(); await page.waitForFunction(() => document.getElementById('scene-label').textContent.includes('RUVIEW'));
      await page.locator('#shortcut').click(); await page.locator('#evaluate').click();
      await page.waitForFunction(() => document.getElementById('gate-status').textContent === 'REJECTED');
      await page.locator('#repair').click(); await page.locator('#evaluate').click();
      await page.waitForFunction(() => document.getElementById('gate-status').textContent === 'READY TO PROMOTE');
      await page.screenshot({ path: `.artifacts/${viewport.width}-room.png`, fullPage: true });
      // Any edit invalidates previously passing promotion evidence.
      await page.locator('#weight-a').evaluate(e => { e.value = '0.5'; e.dispatchEvent(new Event('input', { bubbles: true })); });
      assert.equal(await page.locator('#promote').isDisabled(), true);
      await page.locator('[data-tab="bench"]').click(); await page.locator('#run-benchmark').click();
      await page.waitForFunction(() => document.querySelectorAll('#benchmark-results dl').length === 2);
      await page.screenshot({ path: `.artifacts/${viewport.width}-benchmark.png`, fullPage: true });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      assert.deepEqual(errors, []); observations.push({ viewport, errors, flowsPassed: true }); await page.close();
    }
    const page = await browser.newPage(); await page.goto(url); await page.waitForSelector('html[data-ready="true"]');
    await page.locator('#search').click(); await page.waitForFunction(() => document.getElementById('search-note').textContent.includes('candidate screens'));
    await page.locator('#evaluate').click(); await page.waitForFunction(() => document.getElementById('gate-status').textContent === 'REJECTED');
    await page.route('**/api/evaluate', route => route.abort());
    await page.locator('#repair').click(); await page.locator('#evaluate').click();
    await page.waitForFunction(() => document.getElementById('notice').textContent.includes('fetch') || document.getElementById('notice').textContent.includes('unavailable'));
    assert.equal(await page.locator('#promote').isDisabled(), true);
    observations.push({ failedEdgeBlocksPromotion: true, boundedSearchShortcutRejected: true });
    await page.close();
    await writeFile('.artifacts/browser-results.json', JSON.stringify({ url, observations }, null, 2));
  } finally { await browser.close(); if (server) await new Promise(resolve => server.close(resolve)); }
});
