import http from 'node:http';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
export async function serveSite(root, port=0) {
  root=path.resolve(root);
  const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.svg':'image/svg+xml','.json':'application/json','.wasm':'application/wasm','.png':'image/png','.ico':'image/x-icon','.woff2':'font/woff2'};
  const server=http.createServer(async(req,res)=>{
    try {
      const url=new URL(req.url,'http://localhost');
      if(!url.pathname.startsWith('/ruv-FANN/')) {res.writeHead(404).end();return;}
      const relative=decodeURIComponent(url.pathname.slice('/ruv-FANN/'.length))||'index.html';
      const target=path.resolve(root,relative.endsWith('/')?relative+'index.html':relative);
      if(!target.startsWith(root+path.sep)||relative.includes('\\')) {res.writeHead(400).end();return;}
      const stat=await fs.lstat(target); if(!stat.isFile()||stat.isSymbolicLink()) throw Error('Not file');
      res.writeHead(200,{'content-type':types[path.extname(target)]??'application/octet-stream','x-content-type-options':'nosniff','cache-control':'no-store'});res.end(await fs.readFile(target));
    } catch {res.writeHead(404).end('Not found');}
  });
  await new Promise(resolve=>server.listen(port,'127.0.0.1',resolve));
  return {server,url:`http://127.0.0.1:${server.address().port}/ruv-FANN/`};
}
if(process.argv[1]===fileURLToPath(import.meta.url)) {
  const repo=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
  const {url}=await serveSite(path.resolve(repo,process.argv[2]??'_site'),Number(process.env.PORT??4173));console.log(url);
}
