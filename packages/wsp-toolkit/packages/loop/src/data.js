export function record(value, allowed, required, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || ![Object.prototype,null].includes(Object.getPrototypeOf(value)))
    throw new TypeError(`${name} must be a plain data object`);
  const out=Object.create(null);
  for(const key of Reflect.ownKeys(value)) {
    const descriptor=Object.getOwnPropertyDescriptor(value,key);
    if(typeof key!=='string'||!allowed.includes(key)||!descriptor||!Object.hasOwn(descriptor,'value'))throw new TypeError(`unknown field or accessor in ${name}`);
    out[key]=descriptor.value;
  }
  if(required.some(key=>!Object.hasOwn(out,key)))throw new TypeError(`missing ${name} field`);
  return out;
}
export function integer(value,min,max,name) {
  if(!Number.isSafeInteger(value)||value<min||value>max)throw new RangeError(`${name} must be an integer in ${min}..${max}`);
  return value;
}
export function finite(value,min,name) {
  if(typeof value!=='number'||!Number.isFinite(value)||value<min)throw new RangeError(`${name} must be a finite number >= ${min}`);
  return value===0?0:value;
}
export function identifier(value,name) {
  if(typeof value!=='string'||!/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,47}$/.test(value))throw new TypeError(`${name} must be a 1..48 character identifier`);
  return value;
}
export function snapshot(value) {
  const parents=new Set();let nodes=0,chars=0;
  function copy(item,depth) {
    if(++nodes>524288||depth>20)throw new RangeError('loop input tree exceeds limits');
    if(item===null||typeof item==='boolean')return item;
    if(typeof item==='number')return finite(item,-Number.MAX_VALUE,'input');
    if(typeof item==='string') {
      chars+=item.length;if(chars>10*1024*1024)throw new RangeError('loop input text exceeds limits');
      return item;
    }
    if(!item||typeof item!=='object'||parents.has(item))throw new TypeError('loop inputs must be acyclic JSON data');
    const array=Array.isArray(item);
    if(!(array?Object.getPrototypeOf(item)===Array.prototype:[Object.prototype,null].includes(Object.getPrototypeOf(item))))throw new TypeError('loop inputs must be plain data');
    if(array&&item.length>262144)throw new RangeError('loop input array exceeds limits');
    const out=array?[]:Object.create(null);parents.add(item);
    for(const key of Reflect.ownKeys(item)) {
      if(array&&key==='length')continue;
      if(typeof key!=='string'||['__proto__','prototype','constructor'].includes(key)||(array&&!/^(0|[1-9][0-9]*)$/.test(key)))throw new TypeError('unsafe loop input property');
      chars+=key.length;if(chars>10*1024*1024)throw new RangeError('loop input text exceeds limits');
      const descriptor=Object.getOwnPropertyDescriptor(item,key);
      if(!descriptor||!Object.hasOwn(descriptor,'value'))throw new TypeError('loop input accessors are forbidden');
      out[key]=copy(descriptor.value,depth+1);
    }
    parents.delete(item);
    if(array&&(out.length!==item.length||Object.keys(out).length!==item.length))throw new TypeError('loop arrays must be dense');
    return out;
  }
  const out=copy(value,0);
  if(new TextEncoder().encode(JSON.stringify(out)).length>10*1024*1024)throw new RangeError('loop inputs exceed byte limit');
  return out;
}
export function dataset(value,name) {
  const out=record(value,['inputs','targets','rows'],['inputs','targets','rows'],name);
  integer(out.rows,1,1024,`${name} rows`);
  for(const key of ['inputs','targets'])if(!Array.isArray(out[key])||out[key].length<1||out[key].length>65536)throw new RangeError(`${name} ${key} size invalid`);
  return out;
}
export function policy(value) {
  const out=record(value,['maxMSE','maxRegressionMSE','maxAbsoluteError'],['maxMSE','maxRegressionMSE'],'policy');
  finite(out.maxMSE,0,'maxMSE');finite(out.maxRegressionMSE,0,'maxRegressionMSE');
  if(out.maxAbsoluteError!==undefined)finite(out.maxAbsoluteError,0,'maxAbsoluteError');
  return out;
}
export function limits(value={}) {
  const out=record(value,['maxEdits','maxAbsValue','maxAbsDelta'],[],'limits');
  if(out.maxEdits!==undefined)integer(out.maxEdits,1,128,'maxEdits');
  for(const [key,max]of [['maxAbsValue',1e6],['maxAbsDelta',1000]])if(out[key]!==undefined&&(finite(out[key],0,key)>max))throw new RangeError(`${key} cannot raise core limits`);
  return out;
}
