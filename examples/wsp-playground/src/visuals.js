const LIME = '#bcf169', MUTED = '#a3b7ba', CORAL = '#ff9c88';
function surface(canvas) {
  const r = canvas.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(r.width * d) || canvas.height !== Math.round(r.height * d)) {
    canvas.width = Math.round(r.width * d); canvas.height = Math.round(r.height * d);
  }
  const c = canvas.getContext('2d'); c.setTransform(d, 0, 0, d, 0, 0);
  c.clearRect(0, 0, r.width, r.height);
  return { c, w: r.width, h: r.height };
}
function label(c, text, x, y, color = '#829a9b', size = 9) {
  c.font = `${size}px ui-monospace,monospace`; c.fillStyle = color; c.fillText(text, x, y);
}
function robot(c, x, y, color, angle, time) {
  c.save(); c.translate(x, y); c.rotate(angle);
  c.shadowColor = color; c.shadowBlur = 15;
  c.fillStyle = '#142226'; c.strokeStyle = color; c.lineWidth = 1.5;
  c.beginPath(); c.roundRect(-14, -18, 28, 36, 9); c.fill(); c.stroke(); c.shadowBlur = 0;
  c.fillStyle = color; c.fillRect(-18, -8, 4, 17); c.fillRect(14, -8, 4, 17);
  c.beginPath(); c.arc(0, -3, 6, 0, Math.PI * 2); c.fill();
  c.fillStyle = '#142226'; c.fillRect(-3, -5, 2, 2); c.fillRect(2, -5, 2, 2);
  c.strokeStyle = color; c.beginPath(); c.moveTo(0, -18); c.lineTo(Math.sin(time * 3) * 8, -25); c.stroke();
  c.restore();
}

export function drawScene(canvas, state, time) {
  const { c, w, h } = surface(canvas);
  c.fillStyle = '#101a1e'; c.fillRect(0, 0, w, h);
  c.strokeStyle = '#213238'; c.lineWidth = 0.5;
  for (let x = 0; x < w; x += 27) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke(); }
  for (let y = 0; y < h; y += 27) { c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke(); }
  if (state.name === 'room') { drawRoom(c, w, h, state, time); return; }
  const center = w / 2;
  c.strokeStyle = '#405052'; c.setLineDash([3, 6]); c.beginPath(); c.moveTo(center, 17); c.lineTo(center, h - 40); c.stroke(); c.setLineDash([]);
  label(c, 'ACTIVE', 17, 22, MUTED); label(c, 'CANDIDATE', center + 17, 22, LIME);
  for (const [cx, value, color] of [[w * 0.25, state.active, MUTED], [w * 0.75, state.candidate, LIME]]) {
    const startY = h - 70, targetY = 67, shift = Math.max(-1, Math.min(1, value)) * w * 0.125;
    const a = state.inputs[0], b = state.inputs[1];
    for (const [x, strength] of [[cx - w * 0.17, a], [cx + w * 0.10, b]]) {
      c.fillStyle = `rgba(255,156,136,${0.08 + strength * 0.32})`; c.strokeStyle = `rgba(255,156,136,${0.15 + strength * 0.7})`;
      c.beginPath(); c.roundRect(x, 65, w * 0.07, 70 * strength + 8, 5); c.fill(); c.stroke();
    }
    c.strokeStyle = color + '80'; c.lineWidth = 1.8; c.setLineDash([4, 5]);
    c.beginPath(); c.moveTo(cx, startY); c.bezierCurveTo(cx, startY - 60, cx + shift, targetY + 70, cx + shift, targetY); c.stroke(); c.setLineDash([]);
    const p = (time * 0.18) % 1, ease = p * p * (3 - 2 * p);
    const ry = startY + (targetY - startY) * p, rx = cx + shift * ease;
    c.fillStyle = color + '0d'; c.strokeStyle = color + '25';
    c.beginPath(); c.moveTo(rx, ry); c.arc(rx, ry, 55, -Math.PI * .83, -Math.PI * .17); c.closePath(); c.fill(); c.stroke();
    robot(c, rx, ry, color, Math.atan2(shift * 1.2, startY - targetY), time);
    label(c, value > .05 ? 'TURN RIGHT' : value < -.05 ? 'TURN LEFT' : 'STRAIGHT', cx - 34, h - 37, color, 8);
  }
  label(c, 'OBSTACLE AVOIDANCE · SENSOR RESPONSE ONLY', 15, 41, '#657e80', 7);
}

function drawRoom(c, w, h, state, time) {
  const cx = w * .42, cy = h * .45;
  c.save(); c.translate(cx, cy);
  c.strokeStyle = '#577071'; c.lineWidth = 1;
  const points = [[-w * .26, -20], [0, -85], [w * .27, -15], [10, 60]];
  c.beginPath(); points.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); c.closePath(); c.fillStyle = '#1c2e31'; c.fill(); c.stroke();
  for (const [x, y] of points.slice(0, 3)) { c.beginPath(); c.moveTo(x, y); c.lineTo(x, y - 35); c.stroke(); }
  c.strokeStyle = '#718d8a'; c.beginPath(); c.moveTo(points[0][0], -55); c.lineTo(0, -120); c.lineTo(points[2][0], -50); c.stroke();
  c.fillStyle = '#304446'; c.beginPath(); c.roundRect(-w * .14, -23, w * .12, 35, 5); c.fill();
  const presence = Math.max(0, Math.min(1, state.candidate));
  const px = 22 + Math.sin(time * .7) * 18 * state.inputs[0], py = 0;
  for (let r = 0; r < 3; r++) {
    const radius = 20 + ((time * 22 + r * 24) % 72);
    c.strokeStyle = `rgba(188,241,105,${(1 - (radius - 20) / 72) * (.08 + presence * .3)})`;
    c.beginPath(); c.ellipse(px, 15, radius, radius * .37, 0, 0, Math.PI * 2); c.stroke();
  }
  c.strokeStyle = LIME; c.lineWidth = 3; c.lineCap = 'round';
  c.beginPath(); c.moveTo(px, py - 37); c.lineTo(px, py - 10); c.moveTo(px - 13, py - 25); c.lineTo(px, py - 32); c.lineTo(px + 13, py - 25); c.moveTo(px - 10, py + 10); c.lineTo(px, py - 10); c.lineTo(px + 10, py + 10); c.stroke();
  c.fillStyle = LIME; c.beginPath(); c.arc(px, py - 46, 7, 0, Math.PI * 2); c.fill(); c.restore();
  label(c, 'SIMULATED AMBIENT INTELLIGENCE', 17, 22, '#98ad9d', 8);
  const x = w * .77, y = 72;
  for (const [i, output, color, title] of [[0, state.active, MUTED, 'ACTIVE'], [1, state.candidate, LIME, 'CANDIDATE']]) {
    label(c, title, x, y + i * 80, color, 8);
    c.fillStyle = '#2c3c3b'; c.fillRect(x, y + 13 + i * 80, w * .16, 5);
    c.fillStyle = color; c.fillRect(x, y + 13 + i * 80, w * .16 * Math.max(0, Math.min(1, output)), 5);
    label(c, output.toFixed(2), x, y + 40 + i * 80, color, 14);
  }
  label(c, 'SYNTHETIC EVIDENCE · NO LIVE SENSOR DATA', 17, h - 38, '#657e80', 7);
}

export function drawHeatmap(canvas, predictions, expected) {
  const { c, w, h } = surface(canvas), size = (Math.min(w, h) - 16) / 11;
  for (let a = 0; a < 11; a++) for (let b = 0; b < 11; b++) {
    const error = Math.min(1, Math.abs(predictions[a * 11 + b] - expected[a * 11 + b]));
    const low = [188, 241, 105], high = [255, 130, 115];
    c.fillStyle = `rgb(${low.map((v, i) => Math.round(v + (high[i] - v) * error)).join(',')})`;
    c.globalAlpha = .32 + error * .60;
    c.beginPath(); c.roundRect(14 + b * size, (10 - a) * size, size - 2, size - 2, 2); c.fill();
  }
  c.globalAlpha = 1;
  label(c, '0', 3, h - 15, '#829a9b', 7); label(c, '1', 3, 7, '#829a9b', 7);
  label(c, '0', 15, h - 1, '#829a9b', 7); label(c, '1', w - 10, h - 1, '#829a9b', 7);
}
