import * as THREE from 'three';
import {decodeMorphology,pointAlongRoute} from './morphology-data.mjs';

const positive=new THREE.Color('#adb8be'),negative=new THREE.Color('#b58d80');
const read=async(url,binary=false)=>{const r=await fetch(url);if(!r.ok)throw Error(`Morphology data unavailable (${r.status})`);return binary?r.arrayBuffer():r.json();};

function particles(capacity,size,opacity){
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(capacity*3),3));
  geometry.setAttribute('color',new THREE.BufferAttribute(new Float32Array(capacity*3),3));geometry.setDrawRange(0,0);
  const material=new THREE.PointsMaterial({size,sizeAttenuation:false,vertexColors:true,transparent:true,opacity,depthWrite:false,depthTest:false,blending:THREE.AdditiveBlending});
  material.onBeforeCompile=s=>{s.fragmentShader=s.fragmentShader.replace('#include <clipping_planes_fragment>','#include <clipping_planes_fragment>\nif(length(gl_PointCoord-vec2(.5))>.5)discard;');};
  const result=new THREE.Points(geometry,material);result.frustumCulled=false;result.renderOrder=5;return result;
}

export async function createMorphology(scene,graph){
  const base='artifacts/morphology/';
  const [meta,routeBuffer,arborBuffer]=await Promise.all([read(base+'metadata.json'),read(base+'routes.bin',true),read(base+'arbors.bin',true)]);
  const anatomy=decodeMorphology(meta,routeBuffer,arborBuffer,graph.nodes.map(n=>n.id));
  const textureWidth=8192,states=new Float32Array(textureWidth*4);
  const texture=new THREE.DataTexture(states,textureWidth,1,THREE.RGBAFormat,THREE.FloatType);texture.needsUpdate=true;
  const arborGeometry=new THREE.BufferGeometry();arborGeometry.setAttribute('position',new THREE.BufferAttribute(anatomy.arborPoints,3));arborGeometry.setAttribute('neuron',new THREE.BufferAttribute(anatomy.neuronIds,1));
  const scopeUniform={value:1};
  const arborMaterial=new THREE.ShaderMaterial({transparent:true,depthWrite:false,depthTest:false,blending:THREE.NormalBlending,
    uniforms:{states:{value:texture},scope:scopeUniform},
    vertexShader:`attribute float neuron;uniform sampler2D states;varying vec4 state;varying float z;
      void main(){state=texture2D(states,vec2((neuron+.5)/8192.,.5));z=position.z;
      gl_Position=state.g>.5?projectionMatrix*modelViewMatrix*vec4(position,1.):vec4(2.,2.,2.,1.);}`,
    fragmentShader:`uniform float scope;varying vec4 state;varying float z;
      void main(){if(state.g<.5||(scope>.5&&z< -400.)||(scope< -.5&&z> -400.))discard;
      vec3 c=mix(vec3(.29,.30,.30),vec3(.70,.68,.62),state.b);
      gl_FragColor=vec4(c,.025+state.r*.04+state.b*.34+state.a*.04);
      #include <colorspace_fragment>
      }`});
  // Draw only the currently inspected arbors; keep the complete geometry resident.
  const arbors=new THREE.LineSegments(arborGeometry,[arborMaterial]);arbors.frustumCulled=false;arbors.renderOrder=3;scene.add(arbors);
  const lineMaterial=new THREE.ShaderMaterial({transparent:true,depthWrite:false,depthTest:false,blending:THREE.NormalBlending,
    uniforms:{scope:scopeUniform},vertexShader:`attribute vec3 color;attribute float strength;varying vec3 c;varying float a;varying float z;
      void main(){c=color;a=strength;z=position.z;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader:`uniform float scope;varying vec3 c;varying float a;varying float z;
      void main(){if((scope>.5&&z< -400.)||(scope< -.5&&z> -400.))discard;gl_FragColor=vec4(c,a);
      #include <colorspace_fragment>
      }`});
  const lines=new THREE.LineSegments(new THREE.BufferGeometry(),lineMaterial);lines.frustumCulled=false;lines.renderOrder=4;scene.add(lines);
  let capacity=0,positions,colors,strengths;
  function allocate(needed){if(needed<=capacity)return;capacity=Math.max(needed,capacity*2,20000);
    lines.geometry.dispose();const geo=new THREE.BufferGeometry();
    positions=new Float32Array(capacity*3);colors=new Float32Array(capacity*3);strengths=new Float32Array(capacity);
    geo.setAttribute('position',new THREE.BufferAttribute(positions,3).setUsage(THREE.DynamicDrawUsage));geo.setAttribute('color',new THREE.BufferAttribute(colors,3).setUsage(THREE.DynamicDrawUsage));geo.setAttribute('strength',new THREE.BufferAttribute(strengths,1).setUsage(THREE.DynamicDrawUsage));lines.geometry=geo;
  }
  allocate(20000);lines.geometry.setDrawRange(0,0);
  const packets=particles(graph.edges.length,7,.92),sites=particles(graph.edges.length,3,.5);scene.add(packets,sites);
  const inScope=(z,scope)=>scope==='all'||(scope==='brain'?z>-400:z<=-400);
  function update(visible,data,time,rates,{scope,gain,selected,focus,neighbors,branches,previous=null}){
    scopeUniform.value=scope==='brain'?1:scope==='vnc'?-1:0;
    const needed=visible.reduce((total,item)=>total+(anatomy.routes[item.e].count-1)*2,0);allocate(needed);
    let vertex=0,packetCount=0,siteCount=0;
    for(const item of visible){const route=anatomy.routes[item.e],p=route.points,c=data.trace.edge_weights_mv[item.e]>=0?positive:negative;
      const alpha=Math.min(.52,.05+Math.log1p(item.score)/Math.log(18)*.26*gain);
      for(let j=0;j<route.count-1;j++)for(let endpoint=0;endpoint<2;endpoint++){
        const k=j+endpoint;positions.set(p.subarray(k*3,k*3+3),vertex*3);c.toArray(colors,vertex*3);strengths[vertex]=alpha*(.65+.35*k/(route.count-1));vertex++;
      }
      const s=route.synapse_vertex*3,siteZ=(p[s+2]+p[s+5])*.5;
      if(inScope(siteZ,scope)){
        const array=sites.geometry.attributes.position.array;for(let axis=0;axis<3;axis++)array[siteCount*3+axis]=(p[s+axis]+p[s+3+axis])*.5;
        c.toArray(sites.geometry.attributes.color.array,siteCount*3);siteCount++;
      }
      if(item.inFlight){const array=packets.geometry.attributes.position.array;pointAlongRoute(route,item.progress,array,packetCount*3);
        if(inScope(array[packetCount*3+2],scope)){c.toArray(packets.geometry.attributes.color.array,packetCount*3);packetCount++;}}
    }
    lines.geometry.setDrawRange(0,vertex);for(const a of Object.values(lines.geometry.attributes)){a.clearUpdateRanges();a.addUpdateRange(0,vertex*a.itemSize);a.needsUpdate=true;}
    for(const [object,count] of [[packets,packetCount],[sites,siteCount]]){object.geometry.setDrawRange(0,count);for(const a of Object.values(object.geometry.attributes)){a.clearUpdateRanges();a.addUpdateRange(0,count*a.itemSize);a.needsUpdate=true;}}
    states.fill(0);arbors.visible=branches;arborGeometry.clearGroups();
    if(branches){
      const chosen=focus?[...neighbors]:meta.arbors.map((a,i)=>i).filter(i=>{const a=meta.arbors[i];return a&&(rates?.[i]||0)>1&&(scope==='all'||(scope==='brain'?a.bounds_um[0][2]<400:a.bounds_um[1][2]>=400));}).sort((a,b)=>rates[b]-rates[a]).slice(0,64);
      if(selected>=0&&!chosen.includes(selected))chosen.push(selected);
      const frame=Math.max(0,Math.min(40,Math.floor(time)));
      for(const i of chosen){if(!meta.arbors[i])continue;const age=data?time-data.lastSpikes[frame][i]:100;
        arborGeometry.addGroup(meta.arbors[i].offset,meta.arbors[i].count,0);
        states[i*4]=Math.min(1,(rates?.[i]||0)/60*gain);states[i*4+1]=1;states[i*4+2]=i===selected?1:(focus ? .35 : 0);states[i*4+3]=(!previous||data.lastSpikes[frame][i]!==previous.lastSpikes[frame][i])&&age>=0&&age<2?Math.exp(-age*1.8):0;}
      texture.needsUpdate=true;
    }
  }
  return {meta,hasRoute:e=>!!anatomy.routes[e],update,resize(scale){packets.material.size=7*scale;sites.material.size=3*scale;},
    cellDetails(i){const a=meta.arbors[i];return a?`${a.source_edges.toLocaleString()} source skeleton segments`:'Skeleton unavailable';}};
}
