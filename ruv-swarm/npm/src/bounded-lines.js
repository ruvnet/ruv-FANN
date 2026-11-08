/** Bounded newline framing for stdio JSON-RPC. Oversized frames are discarded. */
export function createBoundedLineDecoder({ onLine, onError, maxBytes = 1024 * 1024 }) {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) {
    throw new RangeError('maxBytes must be a positive safe integer');
  }
  let parts = [];
  let bytes = 0;
  let discarding = false;

  return {
    write(chunk) {
      let start = 0;
      while (start < chunk.length) {
        const newline = chunk.indexOf('\n', start);
        const end = newline === -1 ? chunk.length : newline;
        const part = chunk.slice(start, end);
        if (!discarding) {
          const partBytes = Buffer.byteLength(part, 'utf8');
          if (partBytes > maxBytes - bytes) {
            parts = [];
            bytes = 0;
            discarding = true;
            onError(new RangeError(`JSON-RPC frame exceeds ${maxBytes} bytes`));
          } else {
            parts.push(part);
            bytes += partBytes;
          }
        }
        if (newline === -1) {
          break;
        }
        if (!discarding) {
          onLine(parts.join(''));
        }
        parts = [];
        bytes = 0;
        discarding = false;
        start = end + 1;
      }
    },
    end() {
      // A frame is complete only when newline-terminated.
      parts = [];
      bytes = 0;
      discarding = false;
    },
  };
}
