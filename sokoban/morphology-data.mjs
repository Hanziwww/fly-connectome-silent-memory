// Source-space coordinates are converted by the same rigid rotation as the atlas.
export function decodeMorphology(meta,routeBuffer,arborBuffer,bodyIds){
  if(meta.schema_version!==1||meta.units!=='um'||meta.body_ids.length!==bodyIds.length||
    meta.body_ids.some((id,i)=>id!==bodyIds[i]))throw Error('Morphology body-ID mapping does not match the activity sample.');
  const positions=buffer=>{const p=new Float32Array(buffer);if(p.length%3)throw Error('Invalid morphology coordinates.');
    for(let i=0;i<p.length;i+=3){if(!Number.isFinite(p[i]+p[i+1]+p[i+2]))throw Error('Non-finite morphology coordinate.');p[i+1]*=-1;p[i+2]*=-1;}return p;};
  const routePoints=positions(routeBuffer),arborPoints=positions(arborBuffer);
  const routes=meta.routes.map(r=>{if(!r)return null;if(r.count<2||(r.offset+r.count)*3>routePoints.length)throw Error('Invalid anatomical route.');
    const points=routePoints.subarray(r.offset*3,(r.offset+r.count)*3),distances=new Float32Array(r.count);
    for(let i=1;i<r.count;i++){const k=i*3;distances[i]=distances[i-1]+Math.hypot(points[k]-points[k-3],points[k+1]-points[k-2],points[k+2]-points[k-1]);}
    return {...r,points,distances,length:distances[distances.length-1]};
  });
  const neuronIds=new Float32Array(arborPoints.length/3);
  meta.arbors.forEach((a,i)=>{if(!a)return;if(a.count%2||(a.offset+a.count)>neuronIds.length)throw Error('Invalid arbor geometry.');neuronIds.fill(i,a.offset,a.offset+a.count);});
  return {meta,routes,arborPoints,neuronIds};
}

export function pointAlongRoute(route,progress,out,offset=0){
  const target=Math.max(0,Math.min(1,progress))*route.length,d=route.distances,p=route.points;
  let lo=0,hi=d.length-1;
  while(hi-lo>1){const mid=(lo+hi)>>1;if(d[mid]<target)lo=mid;else hi=mid;}
  const f=d[hi]>d[lo]?(target-d[lo])/(d[hi]-d[lo]):0;
  for(let axis=0;axis<3;axis++)out[offset+axis]=p[lo*3+axis]+f*(p[hi*3+axis]-p[lo*3+axis]);
}
