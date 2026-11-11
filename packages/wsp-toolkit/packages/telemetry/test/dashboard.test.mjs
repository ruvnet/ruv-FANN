import test from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { analyzeConstellation,renderConstellationDashboard } from '../src/index.js';
import { catalog,observations } from './fixtures.mjs';

test('real Chromium presents explicitly labelled observed subtotals while cohort totals stay unknown',async()=>{
  const input=observations();input.packages[1].status='unavailable';input.packages[1].days=[];
  const report=analyzeConstellation(catalog(),input);assert.equal(report.totals.last30Downloads,null);assert.equal(report.totals.observedLast30Downloads,825);
  const browser=await chromium.launch({headless:true});
  try{
    const page=await browser.newPage(),errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.setContent(renderConstellationDashboard(report));
    assert.equal(await page.locator('#download-label').textContent(),'30-day observed subtotal');assert.equal(await page.locator('#download-total').textContent(),'825');
    assert.match(await page.locator('#download-detail').textContent(),/1\/2.*Full-cohort total is unknown/);
    assert.match(await page.locator('#trend-caption').textContent(),/Gold dashed.*1\/2/);assert.equal(await page.locator('#trend-chart path.partial').count(),1);
    assert.equal(await page.locator('#trend-legend-dot.partial').count(),1);
    await page.selectOption('#period-filter','7');assert.equal(await page.locator('#download-total').textContent(),'273');assert.equal(await page.locator('#download-label').textContent(),'7-day observed subtotal');
    await page.selectOption('#period-filter','1');assert.equal(await page.locator('#download-total').textContent(),'42');assert.equal(await page.locator('#download-label').textContent(),'Final-day observed subtotal');
    assert.equal(await page.locator('#trend-chart circle').count(),1);assert.match(await page.locator('#trend-caption').textContent(),/1\/2/);
    await page.selectOption('#trend-package','core');assert.equal(await page.locator('#trend-chart circle').count(),0);assert.match(await page.locator('#trend-caption').textContent(),/No observed daily series/);
    assert.deepEqual(errors,[]);assert.equal(report.totals.last30Downloads,null);
  }finally{await browser.close();}
});
test('real Chromium keeps no-observation windows unknown instead of displaying a zero subtotal',async()=>{
  const input=observations();for(const item of input.packages.filter(p=>p.status!=='candidate')){item.status='unavailable';item.days=[];}
  const report=analyzeConstellation(catalog(),input),browser=await chromium.launch({headless:true});
  try{
    const page=await browser.newPage();await page.setContent(renderConstellationDashboard(report));
    assert.equal(await page.locator('#download-total').textContent(),'Unknown');assert.equal(await page.locator('#download-label').textContent(),'30-day downloads');
    assert.match(await page.locator('#download-detail').textContent(),/0\/2.*Full-cohort total is unknown/);assert.equal(await page.locator('#trend-chart path').count(),0);assert.equal(await page.locator('#trend-chart circle').count(),0);
    assert.match(await page.locator('#trend-caption').textContent(),/No observed daily series/);
    await page.selectOption('#period-filter','1');assert.equal(await page.locator('#download-total').textContent(),'Unknown');assert.equal(await page.locator('#download-label').textContent(),'Final-day downloads');
  }finally{await browser.close();}
});
