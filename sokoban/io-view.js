import {readoutFrame} from './readout-data.mjs';

export function createReadout(graph,{onSelect=()=>{}}={}) {
  const $=id=>document.getElementById(id),canvas=$('readout-map'),ctx=canvas.getContext('2d');
  const count=graph.readouts,columns=64,rows=Math.ceil(count/columns);
  canvas.width=columns;canvas.height=rows;
  const pixels=ctx.createImageData(columns,rows),neutral=[34,34,34],negative=[133,158,171],positive=[213,198,174];
  const colorAt=value=>{const amount=Math.sqrt(Math.min(1,Math.abs(value)/15)),target=value<0?negative:positive;return neutral.map((v,c)=>Math.round(v+(target[c]-v)*amount));};
  document.querySelector('.voltage-key i').style.background=`linear-gradient(90deg,${Array.from({length:31},(_,i)=>`rgb(${colorAt(i-15).join(',')}) ${i/30*100}%`).join(',')})`;
  let values=null,lastData=null,lastPrevious=null,lastFrame=-1,lastModel='',selected=-1;
  function detail(){
    if(selected<0||!values){$('readout-detail').textContent=values?'Select a cell to inspect it.':'No neural recording';return;}
    const value=values[selected],node=graph.nodes[selected];
    $('readout-detail').textContent=`${node.type} · ${node.id} · ${value>=0?'+':''}${value.toFixed(2)} mV`;
  }
  function update({data,frame,previous=null,referenceKind='previous-input',referenceLabel:savedLabel=null}) {
    const model=$('model').value;
    if(data===lastData&&previous===lastPrevious&&frame===lastFrame&&model===lastModel)return;
    lastData=data;lastPrevious=previous;lastFrame=frame;lastModel=model;
    values=readoutFrame(data,frame,count,previous);
    const available=!!values;
    $('readout-empty').hidden=available;
    $('readout-empty').textContent=$('model').value==='cut_wire'?'Disconnected':$('model').value==='board'?'Direct-board control':'No recording';
    $('readout-time').textContent=available?`${frame} ms`:'— ms';
    const referenceLabel=referenceKind==='previous-input'?'previous input':savedLabel||(referenceKind==='before-repair'?'before repair':'before learning');
    $('readout-reference').textContent=previous?`ΔVm vs ${referenceLabel}`:'ΔVm from rest';
    canvas.dataset.frame=available?String(frame):'';canvas.dataset.cells=available?String(count):'0';
    canvas.dataset.reference=previous?referenceKind:'reset';
    for(let i=0;i<columns*rows;i++){
      const color=colorAt(values?.[i]||0);
      for(let channel=0;channel<3;channel++)pixels.data[i*4+channel]=color[channel];
      pixels.data[i*4+3]=255;
    }
    ctx.putImageData(pixels,0,0);
    canvas.setAttribute('aria-label',available?`${count.toLocaleString()} observed neurons at ${frame} ms, in readout index order. Voltage difference ${previous?(referenceKind!=='previous-input'?`from ${referenceLabel} on the same input`:'from the previous input'):'from rest'}. Color saturates at minus 15 and plus 15 millivolts. Select a cell to inspect.`:'Neural readout unavailable for this view.');
    detail();
  }
  function inspect(index){if(!values)return;selected=Math.max(0,Math.min(count-1,index));detail();if(graph.nodes[selected].soma_um)onSelect(selected);}
  canvas.onclick=e=>{const box=canvas.getBoundingClientRect();inspect(Math.min(rows-1,Math.floor((e.clientY-box.top)/box.height*rows))*columns+Math.min(columns-1,Math.floor((e.clientX-box.left)/box.width*columns)));};
  canvas.onkeydown=e=>{
    const offsets={ArrowLeft:-1,ArrowRight:1,ArrowUp:-columns,ArrowDown:columns};
    if(e.key in offsets){e.preventDefault();inspect(selected<0?0:selected+offsets[e.key]);}
    else if(e.key==='Enter'||e.key==='Home'){e.preventDefault();inspect(e.key==='Home'?0:Math.max(0,selected));}
  };
  return {update};
}
