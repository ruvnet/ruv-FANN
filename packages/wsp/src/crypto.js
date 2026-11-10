let fallback;
/** Browsers require secure-context WebCrypto. Node's built-in provider is the only fallback. */
export async function cryptoProvider() {
  if (globalThis.crypto?.subtle && typeof globalThis.crypto.getRandomValues === 'function') return globalThis.crypto;
  if (typeof process === 'object' && process.versions?.node) {
    const moduleName = 'node:crypto';
    fallback ??= import(moduleName).then(module => module.webcrypto);
    return fallback;
  }
  throw new Error('WebCrypto is required; use a secure browser context (HTTPS or localhost)');
}
export async function hashJSON(json) {
  const provider = await cryptoProvider();
  const digest = await provider.subtle.digest('SHA-256', new TextEncoder().encode(json));
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}
export function randomId(provider) {
  const bytes = provider.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
}
