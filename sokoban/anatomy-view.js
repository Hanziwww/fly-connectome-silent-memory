import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {buildActivity,edgeState,compareActivity,changedEdgeState,playbackStartFrame} from './activity-data.mjs';
import {createMorphology} from './morphology-view.js';

const $=id=>document.getElementById(id);
const world=p=>new THREE.Vector3(p[0],-p[1],-p[2]);
async function read(url,binary=false){const r=await fetch(url);if(!r.ok)throw Error(`Anatomical data unavailable (${r.status}).`);return binary?r.arrayBuffer():r.json();}
function dots(positions,color,size,opacity){
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  const m=new THREE.PointsMaterial({color,size,sizeAttenuation:false,transparent:true,opacity,depthWrite:false});
  m.onBeforeCompile=s=>{s.fragmentShader=s.fragmentShader.replace('#include <clipping_planes_fragment>',
    '#include <clipping_planes_fragment>\nif(length(gl_PointCoord-vec2(0.5))>0.5) discard;');};
  return new THREE.Points(g,m);
}
function luminousCells(positions){
  const g=new THREE.BufferGeometry(),count=positions.length/3;
  g.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  for(const key of ['energy','flash','shown'])g.setAttribute(key,new THREE.BufferAttribute(new Float32Array(count),1));
  const m=new THREE.ShaderMaterial({transparent:true,depthWrite:false,depthTest:false,blending:THREE.AdditiveBlending,
    uniforms:{dpr:{value:1},sizeScale:{value:1}},
    vertexShader:`attribute float energy;attribute float flash;attribute float shown;uniform float dpr;uniform float sizeScale;
      varying float vEnergy;varying float vFlash;varying float vShown;
      void main(){vEnergy=energy;vFlash=flash;vShown=shown;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);
      gl_PointSize=(energy>0.?4.+sqrt(energy)*7.+flash*3.:1.8)*dpr*sizeScale;}`,
    fragmentShader:`varying float vEnergy;varying float vFlash;varying float vShown;
      void main(){float r=length(gl_PointCoord-.5)*2.;if(r>1.||vShown<.5)discard;
      float core=1.-smoothstep(.15,.32,r);float halo=exp(-5.*r*r)*(1.-smoothstep(.7,1.,r));
      vec3 low=vec3(.14,.14,.14),warm=vec3(.65,.64,.60),hot=vec3(.95,.94,.90);
      vec3 color=mix(mix(low,warm,smoothstep(0.,.4,vEnergy)),hot,smoothstep(.55,1.,vEnergy));color=mix(color,vec3(1.),vFlash*.8);
      float alpha=core*(.12+vEnergy*.8+vFlash*.2)+halo*vEnergy*.2;
      if(vEnergy<=0.){color=low;alpha=core*.45;}gl_FragColor=vec4(color,alpha*vShown);
      #include <colorspace_fragment>
      }`});
  const p=new THREE.Points(g,m);p.renderOrder=6;p.frustumCulled=false;return p;
}

export async function createAnatomy(canvas,graph,{onInspect=()=>{},onTime=()=>{},onComplete=()=>{},onActivity=()=>{}}={}){
  const [meta,binary,somata]=await Promise.all([read('artifacts/anatomy/metadata.json'),read('artifacts/anatomy/regions.bin',true),read('artifacts/anatomy/somata.bin',true)]);
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false,preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.setClearColor('#000000',1);
  const scene=new THREE.Scene(),camera=new THREE.OrthographicCamera(-400,400,250,-250,.1,10000);
  camera.position.set(380,-220,1500);const controls=new OrbitControls(camera,canvas);
  controls.enableDamping=false;controls.minZoom=.35;controls.maxZoom=12;controls.target.set(380,-220,-220);
  scene.add(new THREE.HemisphereLight(0xf0eee8,0x393833,1.5));
  const light=new THREE.DirectionalLight(0xf5f3ed,1.8);light.position.set(-400,700,1200);scene.add(light);
  const groups=new Map(meta.groups.map(g=>[g.id,g])),regions=[];
  for(const item of meta.regions){
    const offset=item.byte_offset,view=new DataView(binary,offset,8),nv=view.getUint32(0,true),nf=view.getUint32(4,true);
    const positions=new Float32Array(binary,offset+8,nv*3).slice();
    for(let i=0;i<nv;i++){positions[3*i+1]*=-1;positions[3*i+2]*=-1;}
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(positions,3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(binary,offset+8+nv*12,nf*3),1));g.computeVertexNormals();g.computeBoundingSphere();
    const m=new THREE.Mesh(g,new THREE.MeshPhongMaterial({color:groups.get(item.group).color,transparent:true,opacity:.08,shininess:12,side:THREE.DoubleSide,depthWrite:false}));
    m.userData=item;regions.push(m);scene.add(m);
  }
  const parts={brain:[],vnc:[]},xyz=new Float32Array(somata),backgrounds={};
  for(let i=0;i<xyz.length;i+=3)parts[xyz[i+2]<400?'brain':'vnc'].push(xyz[i],-xyz[i+1],-xyz[i+2]);
  for(const name of ['brain','vnc']){backgrounds[name]=dots(parts[name],'#73766b',.85,.085);scene.add(backgrounds[name]);}
  const positions=[],located=[],indexMap=[],worldById=graph.nodes.map(n=>n.soma_um?world(n.soma_um):null);
  worldById.forEach((p,i)=>{if(p){positions.push(...p.toArray());located.push(p);indexMap.push(i);}});
  const cells=luminousCells(positions);cells.material.uniforms.dpr.value=renderer.getPixelRatio();scene.add(cells);
  $('anatomy-loading').textContent='Loading anatomy…';
  const morphology=await createMorphology(scene,graph);
  const ring=new THREE.Mesh(new THREE.RingGeometry(5,6.5,48),new THREE.MeshBasicMaterial({color:'#ded4b8',side:THREE.DoubleSide,depthTest:false}));ring.visible=false;ring.renderOrder=7;scene.add(ring);
  let scope='brain',preset='front',filter='',selected=-1,time=40,data=null,fallbackRates=null,previous=null,voltageChange=null,finishReplay=null,replayStart=0;
  let responseObject=null,stepBaseline=null;
  let fitWidth=880,fitHeight=480,projected=[],down=null,lastHover=0,playing=false,raf=0,lastClock=0,lastRender=0,topIndices=[];
  const raycaster=new THREE.Raycaster(),mouse=new THREE.Vector2();
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  const inScope=p=>!!p&&(scope==='all'||(scope==='brain'?p.z>-400:p.z<=-400));
  const frameIndex=()=>Math.max(0,Math.min(40,Math.floor(time)));
  const ratesAt=()=>data?data.rates[frameIndex()]:fallbackRates;
  const comparing=()=>$('activity-mode').value!=='all'&&!!voltageChange;
  const referenceKind=()=>$('activity-mode').value==='learning'?(responseObject?.reference_kind||'before-learning'):'previous-input';
  const referenceLabel=()=>responseObject?.reference_label||(referenceKind()==='before-repair'?'before repair':'before learning');
  function reference(){const source=$('activity-mode').value==='learning'?responseObject?.reference:stepBaseline;previous=buildActivity(source?.trace,graph.edges,graph.nodes.length);voltageChange=compareActivity(data,previous);}
  const playbackWindow=()=>({start:replayStart,end:40});
  const timelineX=t=>replayStart>=40?400:Math.max(0,Math.min(400,(t-replayStart)/(40-replayStart)*400));
  function refreshWindow(){
    const nodes=$('cells').checked||$('branches').checked?indexMap.filter(i=>inScope(worldById[i])):[];
    const edgeIndices=$('edges').checked?graph.edges.map((edge,e)=>e).filter(e=>morphology.hasRoute(e)&&inScope(worldById[graph.edges[e][0]])&&inScope(worldById[graph.edges[e][1]])):[];
    replayStart=$('include-quiet').checked?0:playbackStartFrame(data,graph.edges,{nodes,edgeIndices,previous,voltageChange:comparing()?voltageChange:null,linkBudget:Number($('link-budget').value)})??0;
    $('snapshot').min=String(replayStart);$('snapshot').title=replayStart?`Quiet start trimmed. Playback begins at ${replayStart} ms.`:'Full 0–40 ms recording';
  }
  const edgeAt=e=>comparing()?changedEdgeState(data,previous,graph.edges,e,time):edgeState(data,graph.edges,e,time);
  const cellValue=(i,t=frameIndex())=>comparing()?`${voltageChange[t][i]>=0?'+':''}${voltageChange[t][i].toFixed(2)} mV`:`${(data?.rates[t]?.[i]||fallbackRates?.[i]||0).toFixed(1)} Hz`;

  function modeLabels(){const diff=comparing();
    $('comparison-label').textContent=diff?(referenceKind()!=='previous-input'?`vs ${referenceLabel()}`:'vs previous step'):data?($('activity-mode').value==='changes'?'First response':'Full response'):'No response';
    $('active-label').textContent='cells';$('spike-label').textContent='spikes/ms';$('link-label').textContent='routes';
    $('energy-label').textContent=diff?'|ΔVm|':'Rate';$('energy-range').textContent=diff?'0–15+ mV':'0–60+ Hz';$('tracked-label').textContent=diff?'Most changed cells':'Tracked cells';
    $('cell-chart').setAttribute('aria-label',diff?`Selected neuron membrane voltage difference from ${referenceKind()!=='previous-input'?`${referenceLabel()} on this input`:'the previous decision'}, in millivolts`:'Selected neuron firing rate over time');
    $('trace-note').textContent=diff?'Brightness = absolute voltage change; unchanged background activity is dimmed. Signed ΔVm appears below. Links show changed synaptic contributions.':data?'Full response to this input. Routes follow real neuron branches; particles replay the model’s 2 ms delay.':fallbackRates?'Disconnected control: no propagated activity.':'No neural response is recorded for this view.';
  }

  function highlights(){
    const opacity=Number($('region-opacity').value)/100;
    for(const m of regions){const r=m.userData;m.visible=$('surfaces').checked&&(scope==='all'||r.dataset===scope);
      m.material.opacity=filter?(filter===r.id||filter===`group:${r.group}`?Math.max(.16,opacity*2):opacity*.3):opacity;}
    for(const name of ['brain','vnc'])backgrounds[name].visible=$('somata').checked&&(scope==='all'||scope===name);
    const item=meta.regions.find(r=>r.id===filter),group=groups.get(filter.slice(6));
    $('region-name').textContent=item?`${item.label} · ${groups.get(item.group).label}`:group?group.label:'All anatomical regions';
    document.querySelectorAll('[data-region-group]').forEach(b=>b.setAttribute('aria-pressed',String(filter===`group:${b.dataset.regionGroup}`)));
  }
  function drawSignals(){
    const r=ratesAt(),t=frameIndex(),gain=Number($('activity-gain').value)/100,diff=comparing(),change=diff?voltageChange[t]:null;
    const focus=$('focus-cell').checked&&selected>=0,neighbors=new Set([selected]);
    let selectedLinks=0,selectedActive=0;
    if(selected>=0)for(let e=0;e<graph.edges.length;e++){
      const [pre,post]=graph.edges[e];if(!morphology.hasRoute(e)||(pre!==selected&&post!==selected)||!inScope(worldById[pre])||!inScope(worldById[post]))continue;
      selectedLinks++;neighbors.add(pre);neighbors.add(post);if(data&&edgeAt(e).score>.03)selectedActive++;
    }
    let activeCount=0,spikeCount=0;
    const energy=cells.geometry.attributes.energy.array,flash=cells.geometry.attributes.flash.array,shown=cells.geometry.attributes.shown.array;
    located.forEach((p,j)=>{const i=indexMap[j],visible=inScope(p),value=diff?Math.abs(change[i]):r?.[i]||0,active=value>(diff ? .12 : 1);
      energy[j]=visible&&active?Math.min(1,Math.pow(value/(diff?15:60),.65)*gain):0;
      const age=data?time-data.lastSpikes[t][i]:100,newSpike=!diff||data.lastSpikes[t][i]!==previous.lastSpikes[t][i];flash[j]=newSpike&&age>=0&&age<2?Math.exp(-age*1.8):0;
      if(focus&&!neighbors.has(i)){energy[j]*=.1;flash[j]*=.1;}
      shown[j]=visible?1:0;if(visible&&active)activeCount++;
      if(visible&&data?.lastSpikes[t][i]===t&&(!diff||previous.lastSpikes[t][i]!==t))spikeCount++;
    });
    for(const name of ['energy','flash','shown'])cells.geometry.attributes[name].needsUpdate=true;
    cells.visible=$('cells').checked;ring.visible=selected>=0&&inScope(worldById[selected]);
    if(ring.visible)ring.position.copy(worldById[selected]);
    const candidates=[];
    if($('edges').checked&&data)for(let e=0;e<graph.edges.length;e++){
      const [pre,post]=graph.edges[e];if(!morphology.hasRoute(e)||!inScope(worldById[pre])||!inScope(worldById[post])||(focus&&pre!==selected&&post!==selected))continue;
      const state=edgeAt(e);if(state.score>.03)candidates.push({e,...state});
    }
    candidates.sort((a,b)=>b.score-a.score);const visible=candidates.slice(0,Number($('link-budget').value));
    const branchStrength=diff?Float32Array.from(change,v=>Math.abs(v)>.12?Math.abs(v)*4:0):r;
    morphology.update(visible,data,time,branchStrength,{scope,gain,selected,focus,neighbors,branches:$('branches').checked,previous:diff?previous:null});
    $('active-count').textContent=activeCount.toLocaleString();$('spike-count').textContent=spikeCount;$('link-count').textContent=visible.length.toLocaleString();
    $('link-count').parentElement.title=`${candidates.length.toLocaleString()} ${diff?'changed':'active'} anatomical routes in this scope; strongest ${$('link-budget').value} displayed. ${morphology.meta.available_routes.toLocaleString()} of 12,000 sampled edges have mapped routes. ${diff?'Contribution change':'Contribution'} cutoff: 0.03 mV.`;
    $('snapshot').value=String(t);$('snapshot-label').textContent=`${t} ms`;
    if(selected>=0){$('cell-rate').textContent=cellValue(selected);$('cell-rate').title=diff?`Current Vm: ${data.voltage[t][selected].toFixed(2)} mV; current rate: ${r[selected].toFixed(1)} Hz`:'';$('cell-connections').textContent=`${selectedActive} ${diff?'changed':'active'} / ${selectedLinks} mapped links · ${morphology.cellDetails(selected)}`;}
    for(const i of topIndices){const el=$(`top-rate-${i}`);if(el)el.textContent=cellValue(i);}
    const line=$('population-cursor');if(line){line.setAttribute('x1',String(timelineX(time)));line.setAttribute('x2',String(timelineX(time)));}
    const cellLine=$('cell-cursor');if(cellLine){cellLine.setAttribute('x1',String(time*10));cellLine.setAttribute('x2',String(time*10));}
    onTime(time,playbackWindow());
    onActivity({data,frame:t,previous:diff?previous:null,referenceKind:referenceKind(),referenceLabel:referenceLabel()});
  }
  function overlay(){
    const w=canvas.clientWidth,h=canvas.clientHeight;
    for(const [group,label]of [['optic_r','r'],['optic_l','l']]){
      const el=$(`side-${label}`),roi=meta.regions.find(r=>r.group===group&&r.label.startsWith('ME(')),p=world(roi.center_um).project(camera);
      el.style.left=`${(p.x+1)*w/2}px`;el.style.top=`${(1-p.y)*h/2}px`;el.hidden=scope==='vnc'||preset==='lateral';
    }
    $('scale-bar').style.width=`${100/(camera.right-camera.left)*camera.zoom*w}px`;
    projected=located.map((p,j)=>{const q=p.clone().project(camera);return {i:indexMap[j],x:(q.x+1)*w/2,y:(1-q.y)*h/2,visible:inScope(p)&&Math.abs(q.x)<=1&&Math.abs(q.y)<=1};});
  }
  function render(){if(ring.visible){ring.quaternion.copy(camera.quaternion);ring.scale.setScalar(1/camera.zoom);}renderer.render(scene,camera);overlay();}
  function resize(){const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;
    const sizeScale=Math.max(.5,Math.min(1,w/680));cells.material.uniforms.sizeScale.value=sizeScale;morphology.resize(sizeScale);
    const height=Math.max(fitHeight,fitWidth*h/w);renderer.setSize(w,h,false);camera.left=-height*w/h/2;camera.right=-camera.left;camera.top=height/2;camera.bottom=-camera.top;camera.updateProjectionMatrix();onTime(time,playbackWindow());render();}
  function frame(){const bounds=new THREE.Box3();for(const m of regions)if(scope==='all'||m.userData.dataset===scope){bounds.expandByPoint(world(m.userData.bounds_um[0]));bounds.expandByPoint(world(m.userData.bounds_um[1]));}
    const center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3());let horizontal=size.x,vertical=size.y,dir=new THREE.Vector3(0,0,1);camera.up.set(0,1,0);
    if(preset==='dorsal'){dir.set(0,1,0);camera.up.set(0,0,1);vertical=size.z;}if(preset==='lateral'){dir.set(1,0,0);horizontal=size.z;}
    fitWidth=horizontal*1.22;fitHeight=vertical*1.28;controls.target.copy(center);camera.position.copy(center).addScaledVector(dir,2000);camera.zoom=1;resize();controls.update();
    $('view-label').textContent=`${scope==='all'?'Whole CNS':scope==='brain'?'Brain':'Ventral nerve cord'} · ${preset} view`;
    document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===preset)));
  }
  function setTime(value){time=Math.max(replayStart,Math.min(40,value));drawSignals();render();}
  function pause(completed=false){playing=false;cancelAnimationFrame(raf);$('play-response').textContent='Replay';const done=finishReplay;finishReplay=null;done?.(completed);}
  function animate(now){if(!playing)return;if(document.hidden){lastClock=now;raf=requestAnimationFrame(animate);return;}
    const elapsed=Math.max(0,now-lastClock);lastClock=now;time=Math.max(replayStart,Math.min(40,time+elapsed*(40-replayStart)/Number($('game-pace').value)));
    if(now-lastRender>=30||time>=40){lastRender=now;drawSignals();render();}if(time>=40){onComplete();pause(true);return;}raf=requestAnimationFrame(animate);
  }
  function play(){pause();if(!data||reduced||replayStart>=40){setTime(40);onComplete();return Promise.resolve(true);}if(time>=40)setTime(replayStart);playing=true;lastClock=performance.now();lastRender=0;$('play-response').textContent='Pause replay';raf=requestAnimationFrame(animate);return new Promise(resolve=>{finishReplay=resolve;});}
  function populationChart(){if(!data){$('population-chart').innerHTML='';return;}
    const counts=comparing()?data.releases.map((r,t)=>r.reduce((n,v,i)=>n+(v>0&&previous.releases[t][i]===0?1:0),0)):data.spikeCounts;
    const max=Math.max(1,...counts.slice(replayStart)),bars=counts.map((n,t)=>t<replayStart?'':`<rect x="${timelineX(t)-3}" y="${40-n/max*34}" width="6" height="${n/max*34}" rx="1" fill="#7f8772"/>`).join('');
    $('population-chart').innerHTML=bars+'<line id="population-cursor" x1="400" x2="400" y1="1" y2="43" stroke="#d0c09a" stroke-width="1.5"/>';
  }
  function cellChart(){if(selected<0||!data){$('cell-chart').classList.remove('has-cell');$('cell-chart').innerHTML='';return;}
    const diff=comparing(),values=(diff?voltageChange:data.rates).map(r=>r[selected]),max=Math.max(diff?1:10,...values.map(Math.abs)),points=values.map((v,t)=>`${t*10},${diff?24-v/max*19:43-v/max*36}`).join(' ');
    $('cell-chart').classList.add('has-cell');$('cell-chart').innerHTML=`${diff?'<line x1="0" x2="400" y1="24" y2="24" stroke="#4d5442"/>':''}<polyline points="${points}" fill="none" stroke="#d1cbc0" stroke-width="1.5"/><line id="cell-cursor" x1="${time*10}" x2="${time*10}" y1="1" y2="48" stroke="#878f79"/><text x="3" y="16" font-size="16" fill="#aaa7a2">${diff?'±':''}${max.toFixed(1)} ${diff?'mV':'Hz'}</text>`;
  }
  function selectCell(i){selected=i;$('cell-inspector').hidden=i<0;$('focus-cell').disabled=i<0;$('focus-cell').checked=i>=0;$('clear-cell').hidden=i<0;
    $('cell-connections').hidden=i<0;
    if(i<0){$('cell-id').textContent='Select a neuron';$('cell-type').textContent='Click a bright cell or choose one below.';$('cell-rate').textContent='— Hz';}
    else{const n=graph.nodes[i];$('cell-id').textContent=`Body ${n.id}`;$('cell-type').textContent=`${n.type} / ${n.superclass} · (${n.soma_um.map(v=>v.toFixed(1)).join(', ')}) μm`;}
    document.querySelectorAll('[data-cell]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.cell)===i)));cellChart();drawSignals();render();
  }
  function topCells(){const diff=comparing(),values=diff?voltageChange[40]:data?.rates[40]||fallbackRates;topIndices=graph.nodes.map((n,i)=>i).filter(i=>inScope(worldById[i])&&Math.abs(values?.[i]||0)>(diff ? .12 : 1)).sort((a,b)=>Math.abs(values[b])-Math.abs(values[a])).slice(0,4);
    $('top-cells').innerHTML=topIndices.map(i=>`<button data-cell="${i}" aria-pressed="${selected===i}" title="Body ${graph.nodes[i].id}"><span class="cell-name">${graph.nodes[i].type}</span><span id="top-rate-${i}" class="cell-value">${cellValue(i,40)}</span></button>`).join('')||`<span class="caption">${diff?'No voltage change above 0.12 mV':'No firing response'} in this view.</span>`;
    document.querySelectorAll('[data-cell]').forEach(b=>b.onclick=()=>selectCell(Number(b.dataset.cell)));
  }
  function setResponse(response,label,autoplay=false,baseline=null){pause();data=buildActivity(response?.trace,graph.edges,graph.nodes.length);fallbackRates=response?.rates?.[1]||null;
    responseObject=response;stepBaseline=baseline;reference();
    $('response-label').textContent=label;$('play-response').disabled=!data;$('step-response').disabled=!data;$('snapshot').disabled=!data;
    refreshWindow();modeLabels();populationChart();cellChart();topCells();setTime(autoplay?replayStart:40);if(autoplay)return play();return Promise.resolve(true);
  }
  function customView(){$('view-label').textContent=`${scope==='all'?'Whole CNS':scope==='brain'?'Brain':'Ventral nerve cord'} · custom view`;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed','false'));}
  function chooseRegion(value){filter=value;$('region-select').value=value;const item=meta.regions.find(r=>r.id===value),target=item?.dataset||(value.startsWith('group:')?(value==='group:vnc'?'vnc':'brain'):null);
    if(target&&scope!=='all'&&scope!==target){scope=target;$('anatomy-scope').value=scope;preset=scope==='brain'?'front':'dorsal';topCells();frame();}highlights();drawSignals();render();}
  function hit(e){const box=canvas.getBoundingClientRect(),x=e.clientX-box.left,y=e.clientY-box.top;let best=-1,distance=81;
    if($('cells').checked)for(const p of projected)if(p.visible){const d=(x-p.x)**2+(y-p.y)**2;if(d<distance){distance=d;best=p.i;}}
    if(best>=0)return {neuron:best,x,y};mouse.set(x/box.width*2-1,1-y/box.height*2);raycaster.setFromCamera(mouse,camera);
    const found=raycaster.intersectObjects(regions.filter(m=>m.visible),false)[0];return found?{roi:found.object.userData,x,y}:null;
  }
  canvas.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];$('tip').hidden=true;});
  canvas.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5){down=null;return;}down=null;const found=hit(e);if(found?.neuron!==undefined)selectCell(found.neuron);else if(found?.roi)chooseRegion(found.roi.id);});
  canvas.addEventListener('pointermove',e=>{if(down){if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)customView();return;}if(performance.now()-lastHover<120)return;lastHover=performance.now();const found=hit(e),tip=$('tip');if(!found){tip.hidden=true;return;}
    if(found.neuron!==undefined){const n=graph.nodes[found.neuron];tip.textContent=`${n.type} · ${cellValue(found.neuron)}${comparing()?' ΔVm':''} · ${n.id}`;}else tip.textContent=`${found.roi.label} · ${groups.get(found.roi.group).label}`;
    tip.style.left=`${Math.max(8,Math.min(found.x+10,canvas.clientWidth-230))}px`;tip.style.top=`${Math.max(8,found.y-33)}px`;tip.hidden=false;
  });canvas.addEventListener('pointerleave',()=>$('tip').hidden=true);
  canvas.addEventListener('keydown',e=>{if(e.key==='Home'){e.preventDefault();frame();return;}if(e.key==='+'||e.key==='=')camera.zoom=Math.min(12,camera.zoom*1.2);else if(e.key==='-')camera.zoom=Math.max(.35,camera.zoom/1.2);
    else if(e.key.startsWith('Arrow')){e.preventDefault();customView();const v=camera.position.clone().sub(controls.target),axis=['ArrowLeft','ArrowRight'].includes(e.key)?camera.up:new THREE.Vector3().crossVectors(camera.up,v).normalize();v.applyAxisAngle(axis,['ArrowLeft','ArrowUp'].includes(e.key)?.12:-.12);camera.position.copy(controls.target).add(v);}else return;camera.updateProjectionMatrix();controls.update();render();});
  controls.addEventListener('change',render);
  $('region-key').innerHTML=meta.groups.map(g=>`<button data-region-group="${g.id}" aria-pressed="false"><i style="background:${g.color}"></i>${g.label}</button>`).join('');
  $('region-select').innerHTML='<option value="">All regions</option>'+meta.groups.map(g=>`<option value="group:${g.id}">${g.label}</option>`).join('')+meta.regions.map(r=>`<option value="${r.id}">${r.label} / ${r.dataset==='brain'?'brain':'VNC'}</option>`).join('');
  $('region-select').onchange=()=>chooseRegion($('region-select').value);document.querySelectorAll('[data-region-group]').forEach(b=>b.onclick=()=>chooseRegion(filter===`group:${b.dataset.regionGroup}`?'':`group:${b.dataset.regionGroup}`));
  $('anatomy-scope').onchange=()=>{scope=$('anatomy-scope').value;preset=scope==='brain'?'front':'dorsal';filter='';$('region-select').value='';highlights();topCells();drawSignals();frame();};
  document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{preset=b.dataset.view;frame();});$('fit-view').onclick=frame;
  for(const id of ['surfaces','somata','cells','branches','edges','focus-cell','link-budget'])$(id).onchange=()=>{highlights();drawSignals();render();};
  $('activity-gain').oninput=()=>{$('gain-label').textContent=`${(Number($('activity-gain').value)/100).toFixed(1)}×`;drawSignals();render();};
  $('region-opacity').oninput=()=>{$('opacity-label').textContent=`${$('region-opacity').value}%`;highlights();render();};
  $('activity-mode').onchange=()=>{reference();refreshWindow();modeLabels();populationChart();cellChart();topCells();setTime(time);};
  $('include-quiet').onchange=()=>{onInspect();pause();refreshWindow();populationChart();setTime(replayStart);};
  $('snapshot').oninput=()=>{onInspect();pause();setTime(Number($('snapshot').value));};$('play-response').onclick=()=>{const wasPlaying=playing;onInspect();if(wasPlaying)pause();else play();};$('step-response').onclick=()=>{onInspect();pause();setTime(time>=40?replayStart:Math.floor(time)+1);};
  $('snapshot').onpointerdown=()=>{onInspect();pause();};$('snapshot').onkeydown=e=>{if(e.key==='Home'||e.key==='End'){e.preventDefault();onInspect();pause();setTime(e.key==='Home'?replayStart:40);}};
  $('population-chart').onclick=e=>{if(!data)return;const box=e.currentTarget.getBoundingClientRect();onInspect();pause();setTime(Math.round(replayStart+(e.clientX-box.left)/box.width*(40-replayStart)));};
  $('clear-cell').onclick=()=>selectCell(-1);
  function expand(value){$('neural-panel').classList.toggle('expanded',value);document.body.classList.toggle('brain-expanded',value);$('expand-view').textContent=value?'Close':'Expand';$('expand-view').setAttribute('aria-expanded',String(value));resize();}
  $('expand-view').onclick=()=>expand(!$('neural-panel').classList.contains('expanded'));document.addEventListener('keydown',e=>{if(e.key==='Escape')expand(false);});
  $('anatomy-caption').textContent=`${meta.soma_count.toLocaleString()} original somata, ${meta.region_count} region surfaces, ${morphology.meta.available_skeletons.toLocaleString()} reconstructed neuron skeletons. ${morphology.meta.available_routes.toLocaleString()} of 12,000 sampled edges have a measured synapse route; others are omitted. Branch view shows the 64 strongest active arbors, or the selected cell and its neighbors. Surface colors identify anatomy.`;
  $('morphology-coverage').textContent=`${morphology.meta.available_skeletons.toLocaleString()} skeletons · ${morphology.meta.available_routes.toLocaleString()} mapped synaptic routes`;
  $('anatomy-loading').hidden=true;highlights();drawSignals();frame();
  return {resize,setResponse,pause,play,seek:setTime,selectCell,clear(){pause();data=null;previous=null;voltageChange=null;fallbackRates=null;selectCell(-1);setResponse(null,'Current board');}};
}
