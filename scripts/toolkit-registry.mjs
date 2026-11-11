import {randomUUID} from 'node:crypto';
import {tarballURL} from './toolkit-release-lib.mjs';

// npm's CDN caches a missing tarball for five minutes. Preflight and propagation
// probes must not prime that negative cache immediately before an upload.
export async function readPublicArchive(row,{canonical=false,fetchImpl=fetch}={}) {
  const url=new URL(tarballURL(row));
  if(!canonical)url.searchParams.set('release-probe',randomUUID());
  const response=await fetchImpl(url.href,{redirect:'error',signal:AbortSignal.timeout(30_000)});
  if(response.status===404){await response.body?.cancel();return null;}
  if(!response.ok)throw Error(`Registry HTTP${response.status}`);
  const maximum=20*1024*1024;
  if(Number(response.headers.get('content-length'))>maximum){await response.body?.cancel();throw Error('Oversized registry tarball');}
  const chunks=[];let size=0;
  for await(const chunk of response.body){size+=chunk.length;if(size>maximum)throw Error('Oversized registry tarball');chunks.push(Buffer.from(chunk));}
  return Buffer.concat(chunks);
}
