// Standalone equivalent of toolkit-registry.mjs: do not prime npm CDN negative caches.
import {randomUUID} from 'node:crypto';
import {tarballURL,assert} from './metadata-release-lib.mjs';
export async function readPublicArchive(row,{canonical=false,fetchImpl=fetch}={}){
  const url=new URL(tarballURL(row));if(!canonical)url.searchParams.set('release-probe',randomUUID());
  const response=await fetchImpl(url.href,{redirect:'error',signal:AbortSignal.timeout(30_000)});
  if(response.status===404){await response.body?.cancel();return null;}
  assert(response.ok,`Registry HTTP ${response.status}`);const maximum=40*1024*1024;
  assert(Number(response.headers.get('content-length'))<=maximum,'Oversized registry tarball');
  const chunks=[];let size=0;for await(const chunk of response.body){size+=chunk.length;assert(size<=maximum,'Oversized registry tarball');chunks.push(Buffer.from(chunk));}return Buffer.concat(chunks);
}
