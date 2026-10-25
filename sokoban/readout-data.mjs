// The readout occupies the first graph.readouts entries of the recorded array.
// Input cells appended for anatomical display are deliberately excluded.
export function readoutFrame(data, frame, count, previous=null) {
  if(!data?.voltage)return null;
  const t=Math.max(0,Math.min(data.voltage.length-1,Math.floor(frame)));
  const current=data.voltage[t],reference=previous?.voltage?.[t]||data.voltage[0];
  if(count>current.length||reference.length!==current.length)throw Error('Incompatible neural readout.');
  return Float32Array.from(current.subarray(0,count),(value,i)=>value-reference[i]);
}
