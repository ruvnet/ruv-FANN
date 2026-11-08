import { createBoundedLineDecoder } from './bounded-lines.js';

/** A bounded JSON-RPC pipe that drains replies before completing stdin EOF. */
export function createStdioRpcServer({ input, output, handle, maxFrameBytes = 1048576,
  maxPending = 32, maxQueuedBytes = 4194304 }) {
  let pending = 0;
  let ended = false;
  let failed = false;
  let writing = false;
  let queuedBytes = 0;
  const queue = [];
  let resolveDone, rejectDone;
  const done = new Promise((resolve, reject) => {
    resolveDone = resolve; rejectDone = reject;
  });
  // Callers attach their awaited lifetime after writing the readiness message.
  done.catch(() => undefined);
  function finish() {
    if (ended && pending === 0 && queue.length === 0 && !writing && !failed) {
      resolveDone();
    }
  }
  function fail(error) {
    if (failed) {
      return;
    }
    failed = true;
    input.pause();
    for (const item of queue.splice(0)) {
      item.reject(error);
    }
    rejectDone(error);
  }
  function updateInput() {
    if (failed || ended || pending >= maxPending || queuedBytes >= maxFrameBytes) {
      input.pause();
    } else {
      input.resume();
    }
  }
  function pump() {
    if (writing || failed || queue.length === 0) {
      finish(); return;
    }
    writing = true;
    const item = queue.shift();
    output.write(item.text, error => {
      writing = false;
      queuedBytes -= item.bytes;
      if (error) {
        item.reject(error); fail(error); return;
      }
      item.resolve();
      updateInput();
      pump();
    });
  }
  function send(response) {
    if (failed) {
      return Promise.reject(new Error('JSON-RPC transport closed'));
    }
    let text = `${JSON.stringify(response)}\n`;
    if (Buffer.byteLength(text, 'utf8') > maxFrameBytes) {
      const limited = { jsonrpc: '2.0', id: response.id ?? null,
        error: { code: -32001, message: 'Response exceeds the configured frame limit' } };
      text = `${JSON.stringify(limited)}\n`;
      if (Buffer.byteLength(text, 'utf8') > maxFrameBytes) {
        limited.id = null;
        text = `${JSON.stringify(limited)}\n`;
      }
    }
    const bytes = Buffer.byteLength(text, 'utf8');
    if (bytes > maxFrameBytes || queuedBytes + bytes > maxQueuedBytes || queue.length >= maxPending) {
      const error = new Error('JSON-RPC output queue limit exceeded');
      fail(error);
      return Promise.reject(error);
    }
    const sent = new Promise((resolve, reject) => queue.push({ text, bytes, resolve, reject }));
    queuedBytes += bytes;
    updateInput();
    pump();
    return sent;
  }
  function error(code, message, id = null) {
    send({ jsonrpc: '2.0', id, error: { code, message } }).catch(fail);
  }
  const decoder = createBoundedLineDecoder({ maxBytes: maxFrameBytes,
    onError: value => error(-32600, value.message),
    onLine(line) {
      if (failed || !line.trim()) {
        return;
      }
      let request;
      try {
        request = JSON.parse(line);
      } catch {
        error(-32700, 'Parse error'); return;
      }
      const id = request?.id;
      const validId = id === undefined || id === null || (typeof id === 'number' && Number.isFinite(id)) ||
        (typeof id === 'string' && Buffer.byteLength(id, 'utf8') <= 256);
      if (!request || typeof request !== 'object' || Array.isArray(request) || request.jsonrpc !== '2.0' ||
          typeof request.method !== 'string' || Buffer.byteLength(request.method, 'utf8') > 128 || !validId) {
        error(-32600, 'Invalid Request'); return;
      }
      const notification = !Object.hasOwn(request, 'id');
      if (request.method.startsWith('notifications/')) {
        return;
      }
      if (pending >= maxPending) {
        if (!notification) {
          error(-32000, 'Too many pending requests', id);
        } return;
      }
      pending++;
      updateInput();
      Promise.resolve().then(() => handle(request)).then(response => {
        if (!notification) {
          return send(response);
        }
      }).catch(value => {
        if (!notification && !failed) {
          return send({ jsonrpc: '2.0', id: id ?? null,
            error: { code: -32603, message: value.message || String(value) } });
        }
      }).catch(fail).finally(() => {
        pending--; updateInput(); finish();
      });
    },
  });
  input.setEncoding('utf8');
  input.on('data', chunk => decoder.write(chunk));
  input.once('end', () => {
    ended = true; decoder.end(); finish();
  });
  input.once('error', fail);
  output.once('error', fail);
  return { send, done };
}
