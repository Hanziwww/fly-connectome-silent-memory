// Pure signal reconstruction. Events and effective weights come from the simulator.
// Frame 0 is the reset state; frame t contains the spike emitted at t milliseconds.
let halfTable;
function decodeVoltage(trace,count){
  if(!trace.voltage_mv_f16)return null;
  if(trace.voltage_shape?.[0]!==trace.steps+1||trace.voltage_shape?.[1]!==count)throw Error('Invalid voltage dimensions.');
  if(!halfTable){halfTable=new Float32Array(65536);for(let i=0;i<65536;i++){const s=i&32768?-1:1,e=(i>>10)&31,f=i&1023;halfTable[i]=e===0?s*f*2**-24:e===31?(f?NaN:s*Infinity):s*(1+f/1024)*2**(e-15);}}
  const bytes=Uint8Array.from(atob(trace.voltage_mv_f16),c=>c.charCodeAt(0));
  if(bytes.length!==(trace.steps+1)*count*2)throw Error('Invalid voltage data length.');
  const values=new Float32Array(bytes.length/2);for(let i=0;i<values.length;i++){values[i]=halfTable[bytes[i*2]|(bytes[i*2+1]<<8)];if(!Number.isFinite(values[i]))throw Error('Invalid membrane voltage.');}
  return Array.from({length:trace.steps+1},(_,t)=>values.subarray(t*count,(t+1)*count));
}
export function buildActivity(trace, edges, count) {
  if (!trace) return null;
  if (trace.schema_version !== 1 || trace.events.length !== trace.steps + 1 || trace.edge_weights_mv.length !== edges.length)
    throw Error('Incompatible neural response data.');
  const rates=[],releases=[],lastSpikes=[],signals=[],spikeCounts=[];
  for(let t=0;t<=trace.steps;t++){
    const rate=new Float32Array(count),release=new Float32Array(count),last=new Int16Array(count);
    if(t){for(let i=0;i<count;i++)rate[i]=rates[t-1][i]*trace.rate_decay;last.set(lastSpikes[t-1]);}
    else last.fill(-100);
    for(const [i,amount] of trace.events[t]){
      if(!Number.isInteger(i)||i<0||i>=count||!Number.isFinite(amount)||amount<0)throw Error('Invalid spike event.');
      rate[i]+=trace.rate_gain_hz;release[i]=amount;last[i]=t;
    }
    rates.push(rate);releases.push(release);lastSpikes.push(last);spikeCounts.push(trace.events[t].length);
    const signal=new Float32Array(edges.length),delayed=t-trace.delay_steps;
    for(let e=0;e<edges.length;e++)signal[e]=(t?signals[t-1][e]*trace.synaptic_decay:0)+
      (delayed>=0?trace.edge_weights_mv[e]*releases[delayed][edges[e][0]]:0);
    signals.push(signal);
  }
  return {rates,releases,lastSpikes,signals,spikeCounts,trace,voltage:decodeVoltage(trace,count)};
}

export function edgeState(data,edges,index,time){
  const t=Math.min(data.trace.steps,Math.max(0,Math.floor(time))),pre=edges[index][0];
  const fired=data.lastSpikes[t][pre],age=time-fired,delay=data.trace.delay_steps;
  const inFlight=fired>0&&age>=0&&age<delay;
  const release=fired>0?data.releases[fired][pre]:0;
  const arriving=inFlight?data.trace.edge_weights_mv[index]*release:0;
  return {value:data.signals[t][index],pending:arriving,inFlight,progress:inFlight?age/delay:0,
    score:Math.max(Math.abs(data.signals[t][index]),Math.abs(arriving))};
}

// Compare reset simulations at the same integration time, never across frames.
export function compareActivity(current,previous){
  if(!current?.voltage||!previous?.voltage)return null;
  if(current.voltage.length!==previous.voltage.length||current.voltage[0].length!==previous.voltage[0].length)throw Error('Incompatible comparison response.');
  return current.voltage.map((frame,t)=>Float32Array.from(frame,(v,i)=>v-previous.voltage[t][i]));
}
export function changedEdgeState(current,previous,edges,index,time){
  const now=edgeState(current,edges,index,time),before=edgeState(previous,edges,index,time);
  const difference=now.value-before.value,pendingDifference=now.pending-before.pending;
  return {...now,score:Math.max(Math.abs(difference),Math.abs(pendingDifference)),
    inFlight:now.inFlight&&Math.abs(pendingDifference)>.03,difference};
}

// Skip a visually quiet lead-in, including isolated weak input responses.
// This is a presentation cutoff (5% of peak scene strength), not neural latency.
// Keep the complete recording available for inspection.
export function playbackStartFrame(data,edges,{nodes=[],edgeIndices=[],previous=null,voltageChange=null,linkBudget=120}={}){
  if(!data)return null;
  const strength=new Float32Array(data.trace.steps+1);
  for(let t=1;t<=data.trace.steps;t++){
    for(const i of nodes){const value=voltageChange?Math.abs(voltageChange[t][i]):data.rates[t][i];if(value>(voltageChange ? .12 : 1))strength[t]+=Math.min(1,Math.pow(value/(voltageChange?15:60),.65));}
    const links=[];
    for(const e of edgeIndices){const state=voltageChange&&previous?changedEdgeState(data,previous,edges,e,t):edgeState(data,edges,e,t);if(state.score>.03)links.push(state.score);}
    links.sort((a,b)=>b-a);
    for(const score of links.slice(0,linkBudget))strength[t]+=Math.min(1,Math.log1p(score)/Math.log(18));
  }
  const peak=Math.max(...strength);if(!peak)return null;
  for(let t=1;t<=data.trace.steps;t++)if(strength[t]>=peak*.05)return t;
  return null;
}
