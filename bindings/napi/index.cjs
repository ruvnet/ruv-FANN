'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const binary = path.join(__dirname, 'ruv_fann.node');
const receipt = path.join(__dirname, 'build-info.json');
if (!fs.existsSync(binary) || !fs.existsSync(receipt)) {
  throw new Error('@ruv-fann/napi requires an explicit local native build; run its build script in the repository. No fallback or download is performed.');
}
const build = JSON.parse(fs.readFileSync(receipt, 'utf8'));
if (build.schema !== 'ruv-fann.napi-build/1' || build.platform !== process.platform || build.arch !== process.arch || build.nodeApi !== 8) {
  throw new Error('@ruv-fann/napi native artifact does not match this platform/architecture/API');
}
const bytes = fs.readFileSync(binary);
if (createHash('sha256').update(bytes).digest('hex') !== build.binarySha256) {
  throw new Error('@ruv-fann/napi native artifact differs from its build receipt');
}
module.exports = require(binary);
