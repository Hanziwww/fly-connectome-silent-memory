'use strict';
import {createAnatomy} from './anatomy-view.js?v=20261002-joint';
import {createReadout} from './io-view.js?v=20261002-joint';
const $=id=>document.getElementById(id), arrows=['↑','↓','←','→'],delta=[-10,10,-1,1];
let catalog,graph,levels,current,player,boxes,history=[],moves=[],generation=0,timer,anatomyView,previewSequence=0;
let pending=null,baseline=null,baselineInput=null,changedSquares=new Set(),busy=false,motion=null,queued=null,readoutView;
const canvas=$('network');
function buttons(){for(const id of ['run','step-agent'])$(id).disabled=busy||!anatomyView;$('inspect-current').disabled=busy||!anatomyView;}
function stop(discard=false){generation++;previewSequence++;queued=null;clearInterval(timer);timer=null;anatomyView?.pause();busy=false;if(discard){pending=null;motion=null;}buttons();}
function phase(name,note,action=null){
 $('decision-number').textContent=`Decision ${moves.length+(name==='action'?0:1)}`;
 $('decision-action').textContent=action===null?'—':`${arrows[action]} ${['Up','Down','Left','Right'][action]}`;
 document.querySelectorAll('[data-phase]').forEach(el=>el.setAttribute('aria-current',el.dataset.phase===name?'step':'false'));
 $('decision-note').textContent=note;
}
function channels(input){return [new Set(current.walls),new Set(current.goals),new Set(input.state[1]),new Set([input.state[0]])];}
function showInput(input){
 const now=channels(input),before=baselineInput?channels(baselineInput):null,names=['Walls','Targets','Crates','Player'];let total=0,changes=0;changedSquares=new Set();
 $('input-channels').innerHTML=now.map((active,ch)=>{total+=active.size;let rects='';
  for(let p=0;p<100;p++){const changed=before?active.has(p)!==before[ch].has(p):ch>=2&&active.has(p);if(changed){changes++;changedSquares.add(p);}
   rects+=`<rect x="${p%10*5}" y="${Math.floor(p/10)*5}" width="4" height="4" class="${active.has(p)?'on':''} ${changed?'changed':''}"/>`;}
  return `<figure><svg viewBox="0 0 50 50" role="img" aria-label="${names[ch]} input channel; ${active.size} active pixels">${rects}</svg><figcaption>${names[ch]}</figcaption></figure>`;
 }).join('');
 $('input-caption').textContent=`${total} on · ${input.model==='board'?'direct-board control':input.model==='cut_wire'?'disconnected':`${total*8} driven cells`}`;
 $('input-caption').title=`Each active channel drives 8 neurons. ${before?`${changes} changed channels are outlined.`:'Crate and player channels are outlined.'}`;
 renderBoard();
}
function policyOutput(response){const probabilities=response?.probabilities,names=['Up','Down','Left','Right'];$('policy-output').hidden=!probabilities;
 $('policy-output').innerHTML=probabilities?probabilities.map((p,a)=>`<div class="${a===response.action?'chosen':''}" aria-label="${names[a]}: ${(p*100).toFixed(1)} percent${a===response.action?', selected':''}"><span class="action-name">${arrows[a]} ${names[a]}</span><i style="--probability:${p*100}%"></i><span>${(p*100).toFixed(0)}%</span></div>`).join(''):'';
 $('policy-title').textContent=response?.manual?'Manual action':'Policy output';
 $('output-action').textContent=response?.action===undefined?'—':`${arrows[response.action]} ${names[response.action]}`;
 $('policy-empty').hidden=!!probabilities;
 $('policy-empty').textContent=response?.manual?'User-selected move':'Run or Step to read an action.';
 $('policy-caption').hidden=!probabilities;$('policy-caption').textContent=response?.readout==='linear'?'Linear readout · playback weights fixed':'Frozen readout · correction off';
 const reference=response?.reference?.action;$('reference-action').hidden=reference===undefined||$('model').value===catalog.suites[$('tier').value].reference_model;
 const referenceLabel=response?.reference_label||'before learning';
 $('reference-action').textContent=reference===undefined?'':`${referenceLabel[0].toUpperCase()+referenceLabel.slice(1)}: ${arrows[reference]} ${names[reference]}`;
 if(!response)readoutView?.update({data:null,frame:0});
}
function stateKey(){return [player,...Array.from(boxes).sort((a,b)=>a-b)].join(',');}
function levelGroup(level){return level.suite==='legacy'?String(level.size):level.suite;}
function labelLevels(){const records=catalog.results[$('model').value][$('tier').value].test.records,selected=$('level').value||'0';$('level').innerHTML=levels.map((l,i)=>`<option value="${i}">${l.display_name} · ${records.find(r=>r.level_id===l.id).status==='solved'?'solved':'not solved'}</option>`).join('');$('level').value=selected;}
function agentDetails(){const kind=$('model').value,seconds=catalog.repair_seconds[kind];$('learning-stats').textContent=catalog.joint_model_details?.[kind]|| (seconds?`Repair training: ${seconds.toFixed(1)} s · 13/13 teacher states · frozen readout`:'35,310 shared internal parameters · frozen readout · correction off');const hash=catalog.checkpoint_ids[kind];$('checkpoint-short').textContent=hash.slice(0,10);$('checkpoint-short').title=hash;}
function chooseTier(){stop();const suite=catalog.suites[$('tier').value];levels=catalog.levels.filter(l=>levelGroup(l)===$('tier').value);$('model').innerHTML=suite.models.map(k=>`<option value="${k}">${catalog.models[k]}</option>`).join('');$('model').value=suite.default_model;$('level').innerHTML='';labelLevels();agentDetails();reset();results();chart();}
function reset(inspect=true){stop(true);baseline=null;baselineInput=null;current=levels[Number($('level').value)];player=current.start[0];boxes=new Set(current.start[1]);history=[];moves=[];
 $('status').textContent='Ready';$('level-summary').textContent=`${current.boxes} ${current.boxes===1?'crate':'crates'} · ${current.descriptor.moves}-move reference`;$('signal').textContent='Current board';$('level-id').textContent=current.id;$('optimum').textContent=current.descriptor.moves;$('optimum-caption').hidden=false;
 phase('input','Choose Run or Step.');policyOutput(null);showInput(inputState());anatomyView?.clear();if(inspect&&anatomyView)previewActivity();}
function renderBoard(display=null){const size=current.size,offset=(10-size)/2,walls=new Set(current.walls),goals=new Set(current.goals),shownPlayer=display?display[0]:player,shownBoxes=display?new Set(display[1]):boxes;$('board').style.gridTemplateColumns=`repeat(${size},1fr)`;delete $('board').dataset.motion;delete $('board').dataset.progress;const html=[];
 for(let r=offset;r<offset+size;r++)for(let c=offset;c<offset+size;c++){const p=r*10+c;html.push(`<div data-position="${p}" class="square ${walls.has(p)?'wall':''} ${goals.has(p)?'goal':''} ${shownBoxes.has(p)?'box':''} ${p===shownPlayer?'player':''} ${changedSquares.has(p)?'input-changed':''}"></div>`)}
 $('board').innerHTML=html.join('');if(!motion){$('movement-clock').hidden=true;$('board').setAttribute('aria-label','Sokoban board');}$('moves').textContent=moves.length;$('trace').textContent=moves.slice(-16).map(a=>arrows[a]).join(' ');}
function syncMovement(time,window=motion?.window||{start:0,end:40}){if(!motion)return;motion.window=window;const board=$('board'),progress=window.end<=window.start?1:Math.max(0,Math.min(1,(time-window.start)/(window.end-window.start)));
 if(board.dataset.motion!==motion.id){renderBoard(motion.before);board.dataset.motion=motion.id;}
 const start=motion.before[0],target=motion.after[0],origins=[start];if(motion.before[1].includes(target))origins.push(target);
 for(const from of origins){const a=board.querySelector(`[data-position="${from}"]`),b=board.querySelector(`[data-position="${from+delta[motion.action]}"]`);if(!a||!b)continue;
  a.style.setProperty('--motion-x',`${(b.offsetLeft-a.offsetLeft)*progress}px`);a.style.setProperty('--motion-y',`${(b.offsetTop-a.offsetTop)*progress}px`);}
 for(const from of origins.slice(1)){const a=board.querySelector(`[data-position="${from}"]`);if(a)a.dataset.motionGoal=String(current.goals.includes(progress===0?from:from+delta[motion.action])&&(progress===0||progress===1));}
 board.dataset.progress=progress.toFixed(4);board.setAttribute('aria-label',`Sokoban board, synchronized replay of move ${motion.step}, ${Math.round(progress*100)} percent`);
 $('movement-clock').hidden=false;$('movement-clock').textContent=`${motion.committed?'Replay':'Move'} ${motion.step} · ${Math.floor(time)} / 40 ms`;
}
function move(a){const q=player+delta[a],walls=new Set(current.walls);if(walls.has(q))return false;if(boxes.has(q)){const z=q+delta[a];if(walls.has(z)||boxes.has(z))return false;}
 history.push({player,boxes:[...boxes]});if(boxes.has(q)){boxes.delete(q);boxes.add(q+delta[a]);}player=q;moves.push(a);renderBoard();$('status').textContent=current.goals.every(g=>boxes.has(g))?`Solved in ${moves.length} moves.`:`Move ${moves.length}.`;return true;}
function undo(){stop(true);const s=history.pop();if(s){player=s.player;boxes=new Set(s.boxes);moves.pop();}baseline=null;baselineInput=null;showInput(inputState());policyOutput(null);phase('input','Inspect this board as a fresh decision.');$('status').textContent=s?'Last move undone.':'Move canceled.';anatomyView?.clear();previewActivity();}
async function requestResponse(path,payload){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw Error(data.error||'Neural response unavailable');return data;}
function inputState(){return {level_id:current.id,state:[player,[...boxes]],model:$('model').value};}
function requestKey(input){return `${input.level_id}:${input.model}:${input.state[0]}:${[...input.state[1]].sort((a,b)=>a-b).join(',')}`;}
function prepareNext(response,input){if(response.terminal)return;const next={...input,state:response.state};queued={key:requestKey(next),promise:requestResponse('/api/move',{...next,trace:true}).then(response=>({response}),error=>({error}))};}
function nextState(action){const target=player+delta[action],walls=new Set(current.walls),nextBoxes=new Set(boxes);if(walls.has(target))return null;
 if(nextBoxes.has(target)){const pushed=target+delta[action];if(walls.has(pushed)||nextBoxes.has(pushed))return null;nextBoxes.delete(target);nextBoxes.add(pushed);}return [target,[...nextBoxes]];}
async function previewActivity(){if(!anatomyView||!current)return;const sequence=++previewSequence,token=generation;$('inspect-current').disabled=true;$('signal').textContent='Recording 40 ms of neural activity…';
 try{const r=await requestResponse('/api/activity',inputState());if(token!==generation||sequence!==previewSequence)return;
  $('io-response-label').textContent='Current board';anatomyView.setResponse(r,'Current board',true,baseline);$('signal').textContent=r.source==='direct-board'?'Direct-board control · no neural response':r.source==='disconnected'?'Disconnected control · zero signal':'Current input · no move is being applied';
 }catch(e){if(token===generation)$('signal').textContent=e.message;}finally{if(sequence===previewSequence)buttons();}}

async function prepareDecision(token,manualAction=null,ahead=false){
 motion=null;$('movement-clock').hidden=true;const input=inputState();showInput(input);policyOutput(null);phase('input','Recording the input for a synchronized move and neural replay…');
 $('status').textContent='Recording response…';
 const after=manualAction===null?null:nextState(manualAction);if(manualAction!==null&&!after)return null;
 let recorded;if(manualAction===null&&queued?.key===requestKey(input)){const next=queued;queued=null;const result=await next.promise;if(result.error)throw result.error;recorded=result.response;}else recorded=await requestResponse(manualAction===null?'/api/move':'/api/activity',{...input,trace:true});
 const response=manualAction===null?recorded:{...recorded,action:manualAction,state:after,terminal:current.goals.every(g=>after[1].includes(g)),manual:true};
 if(token!==generation)return null;
 if(response.action===undefined){$('status').textContent=response.terminal?`Solved in ${moves.length} moves.`:response.status==='proven_corner_deadlock'?'Crate trapped in a corner. Reset to continue.':'No legal move. Reset to continue.';return null;}
 pending={response,input,key:stateKey()};buttons();policyOutput(response);
 $('io-response-label').textContent=`Input to move ${moves.length+1}`;
 if(ahead)prepareNext(response,input);
 motion={id:`${generation}:${moves.length+1}`,before:input.state,after:response.state,action:response.action,step:moves.length+1,committed:false};
 phase('response','The player, pushed crate and neural response move on the same timeline.',response.action);
 $('status').textContent='Playing';
 $('signal').textContent=response.trace?(response.manual?'User-selected move · neural response to the input board':'Synchronized replay · response to the input before this move'):response.rates?'Disconnected control · zero signal':'Direct-board control · no neural response';
 const completed=await anatomyView.setResponse(response,`${response.manual?'Manual':'Move'} ${moves.length+1} ${arrows[response.action]}`,true,baseline);
 if(token!==generation)return null;
 return {response,completed};
}
function commitDecision(){
 if(!pending||pending.key!==stateKey())return false;
 const {response,input}=pending;pending=null;baseline=response;baselineInput=input;changedSquares.clear();
 if(!move(response.action))throw Error('The proposed move is no longer legal. Inspect the current board again.');
 if(motion){motion.committed=true;syncMovement(40);}
 buttons();phase('action',`Move ${moves.length} and its response finished together. Seek to replay both.`,response.action);
 $('signal').textContent=`Move ${moves.length} completed · timeline replays this movement and response`;
 if(response.terminal)$('status').textContent=`Solved in ${moves.length} moves.`;
 return response;
}
async function run(limit=512){stop();const token=generation,seen=new Set([stateKey()]);busy=true;buttons();
 try{for(let i=0;i<limit;i++){
  let result;if(pending){const response=pending.response;if(i+1<limit)prepareNext(response,pending.input);phase('response','Resuming movement and neural response together.',response.action);$('status').textContent='Playing';result={response,completed:await anatomyView.play()};}else result=await prepareDecision(token,null,i+1<limit);
  if(token!==generation||!result)return;
  if(!result.completed){$('status').textContent='Paused';return;}
  if(result.response.terminal)return;
  if(seen.has(stateKey())){$('status').textContent='Loop detected. Reset to continue.';return;}seen.add(stateKey());
  if(i===limit-1){if(limit>1)$('status').textContent=`Stopped at the ${limit}-move limit.`;return;}
  if(token!==generation)return;
 }}catch(e){if(token===generation)$('status').textContent=e.message;}finally{if(token===generation){busy=false;buttons();}}}
function replay(){reset(false);run();}
async function manualMove(a){if(!anatomyView||!nextState(a))return;stop(true);const token=generation;busy=true;buttons();
 try{await prepareDecision(token,a);}catch(e){if(token===generation)$('status').textContent=e.message;}finally{if(token===generation){busy=false;buttons();}}
}
function resizeNetwork(){anatomyView?.resize();}
function chart(){if(!catalog)return;const svg=$('training'),w=svg.clientWidth,h=190,left=40,right=w-12,top=16,bottom=155,colors=['#c7bc9b','#8e9eaa','#a7b39b','#c3aaa0'],training=catalog.joint_experiments?.[$('tier').value]?.training||($('tier').value==='repair'?catalog.repair_training:catalog.training);if(!w)return;let html='';
 const max=Math.max(1,...Object.values(training).flat().map(r=>r.generation));
 for(const p of [0,.5,1]){const y=bottom-p*(bottom-top);html+=`<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" stroke="#505050"/><text x="${left-8}" y="${y+5}" text-anchor="end" font-size="16" fill="#aaa7a2">${p*100}</text>`;}
 Object.entries(training).forEach(([name,rows],i)=>{const points=rows.map(r=>`${left+r.generation/max*(right-left)},${bottom-r.current_correct/r.reference_states*(bottom-top)}`).join(' ');html+=`<polyline fill="none" stroke="${colors[i]}" stroke-width="2" points="${points}"/>`;});
 for(const n of [0,Math.round(max/2),max])html+=`<text x="${left+n/max*(right-left)}" y="181" text-anchor="${n===max?'end':'middle'}" font-size="16" fill="#aaa7a2">${n}</text>`;
 svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.innerHTML=html;$('training-key').innerHTML=Object.keys(training).map((name,i)=>`<span style="--series:${colors[i]}">${name}</span>`).join('');}
function results(){const suiteKey=$('tier').value,suite=catalog.suites[suiteKey],experiment=catalog.joint_experiments?.[suiteKey],joint=!!experiment;
 $('training-heading').textContent=joint?'Curriculum completion · joint updates':'Training actions · search generation';
 $('training').setAttribute('aria-label',joint?'Verified suffix completion percentage by joint optimizer update':'Reference-state training accuracy by search generation');
 $('intervention-heading').textContent=joint?'Brain and readout substitutions':'Brain substitution and restoration';
 $('intervention-headings').innerHTML=joint?`<tr><th>Parameter pairing</th><th>${experiment.summary.task_id} outcome</th><th>Moves</th></tr>`:'<tr><th>Intervention</th><th>Solved</th><th>After restoration</th></tr>';
 $('intervention-caption').textContent=joint?'These controls test the saved parameter pair. A separately retrained readout with a frozen brain was not tested.':'Same input rules and frozen readout. Restore the learned brain after every intervention. Counts are paired controls on three fitted checkpoints, not independent brains or tasks.';
 if(joint){
  const info=experiment.summary,final=catalog.results[suite.default_model][suiteKey].test.records[0],before=catalog.results[suite.reference_model][suiteKey].test.records[0];
  const outcome=status=>status==='solved'?'Solved':status==='repeated_state'?'Loop':status.replaceAll('_',' ');
  $('retention-results').hidden=true;$('intervention-results').hidden=false;
  $('score-name').textContent='Level';$('score-before').textContent=`Before ${info.task_id} training`;$('score-after').textContent='Joint trained';
  $('summary').textContent=`${info.task_id} · ${outcome(final.status)} · ${info.size} × ${info.size} · ${info.boxes} crates · ${final.steps} moves · seed ${info.seed} · independent reload verified`;
  $('scores').innerHTML=`<tr><td>${info.task_id}</td><td>${outcome(before.status)}</td><td>${outcome(final.status)}</td><td>${final.steps}</td></tr>`;
  $('comparison').textContent=`${info.training_seconds.toFixed(1)} s training · ${info.updates} joint updates · ${info.core_changed.toLocaleString()} brain + ${info.head_changed.toLocaleString()} readout parameters changed · ${info.peak_gib.toFixed(2)} GiB peak. Previous ${info.previous_task_id}: ${outcome(info.old_task_status).toLowerCase()} after ${info.old_task_steps} moves; retention was not required.`;
  $('intervention-scores').innerHTML=suite.models.map(kind=>{const r=catalog.results[kind][suiteKey].test.records[0];return `<tr><td>${catalog.models[kind]}</td><td>${outcome(r.status)}</td><td>${r.steps}</td></tr>`;}).join('');
  $('evaluation-data').href=experiment.evaluation_path;return;
 }
 const repair=$('tier').value==='repair';$('retention-results').hidden=repair;$('intervention-results').hidden=!repair;
 $('score-name').textContent=repair?'Model':'Level';$('score-before').textContent=repair?'Before repair':'Before';$('score-after').textContent=repair?'After repair':'After';
 $('evaluation-data').href=repair?'/evidence/repair.json':'/evidence/pilot.json';
 if(repair){
  const kinds=catalog.suites.repair.models.filter(k=>k in catalog.repair_seconds),solved=kinds.filter(k=>catalog.results[k].repair.test.solved).length;
  $('summary').textContent=`B00 repair: ${solved}/${kinds.length} checkpoints solved · same starting brain · 300 s maximum per search seed`;
  $('scores').innerHTML=kinds.map(kind=>{const r=catalog.results[kind].repair.test.records[0],before=catalog.results.repair_source.repair.test.records[0];return `<tr><td>${catalog.models[kind]}<br><small>${catalog.repair_seconds[kind].toFixed(1)} s training</small></td><td>${before.status==='solved'?'Solved':'Not solved'}</td><td>${r.status==='solved'?'Solved':'Not solved'}</td><td>${r.steps}</td></tr>`;}).join('');
  $('comparison').textContent='Original method 0/3; objective/selection only 0/3; local search only 0/3; combined repair 3/3. Existing-checkpoint repair on one trained task. Old-task retention remains 1/4.';
  const labels={repair_source:'Replace with before-repair brain',original_core:'Remove all internal modulation',zero_signal:'Replace neural signal with zeros',mean_signal:'Replace neural signal with readout mean',constant_signal:'Hold initial neural signal fixed',undo_pre_log_gain:'Undo outgoing-gain repair',undo_post_log_gain:'Undo incoming-gain repair',undo_threshold_mv:'Undo threshold repair',shuffle_delta:'Shuffle repair increments (10 per model)'};
  $('intervention-scores').innerHTML=catalog.repair_interventions.map(r=>`<tr><td>${labels[r.condition]}</td><td>${r.solved}/${r.total}</td><td>${r.restored}/${r.total}</td></tr>`).join('');return;
 }
 const final=catalog.results.brain.pilot.test,initial=catalog.results.initial.pilot.test;
 $('summary').textContent=`Four levels retained: ${final.solved}/4 · Before learning: ${initial.solved}/4`;
 $('scores').innerHTML=catalog.levels.filter(l=>l.suite==='pilot').map(l=>{const a=initial.records.find(r=>r.level_id===l.id),b=final.records.find(r=>r.level_id===l.id);return `<tr><td>${l.display_name}</td><td>${a.status==='solved'?'Solved':'Not solved'}</td><td>${b.status==='solved'?'Solved':'Not solved'}</td><td>${b.steps}</td></tr>`;}).join('');
 $('comparison').textContent='Two newly acquired tasks; four retained. Fixed readout. Parameter retention, not dynamic working memory or an unseen-map test.';
 $('seed-scores').innerHTML=['initial','stage1','stage2','stage3','stage4'].map((kind,i)=>{const r=catalog.results[kind].pilot.test;const retained=r.records.slice(0,i).filter(x=>x.status==='solved').length;return `<tr><td>${catalog.models[kind]}</td><td>${i}</td><td>${i?`${retained}/${i}`:'—'}</td><td>${r.solved}/4</td></tr>`;}).join('');}
$('tier').onchange=chooseTier;$('level').onchange=()=>reset();$('model').onchange=()=>{labelLevels();agentDetails();reset();};
$('run').onclick=()=>run();$('step-agent').onclick=()=>run(1);
$('pause').onclick=()=>{stop();if(pending)phase('response','Paused. Run or Step to resume.',pending.response.action);$('status').textContent='Paused';};
$('undo').onclick=undo;$('reset').onclick=()=>reset();$('replay').onclick=replay;$('inspect-current').onclick=()=>{stop(true);showInput(inputState());phase('input','Inspecting the current board without applying a move.');policyOutput(null);previewActivity();};
document.querySelectorAll('[data-dir]').forEach(b=>b.onclick=()=>manualMove(Number(b.dataset.dir)));
document.addEventListener('keydown',e=>{if(e.repeat||!current||document.querySelector('dialog[open]')||['SELECT','INPUT','CANVAS','BUTTON'].includes(e.target.tagName))return;const a=['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].indexOf(e.key);if(a>=0){e.preventDefault();manualMove(a);}});
$('show-results').onclick=()=>{$('results-dialog').showModal();chart();};$('show-display').onclick=()=>$('display-dialog').showModal();document.querySelectorAll('[data-close]').forEach(button=>button.onclick=()=>$(button.dataset.close).close());$('methods').onclick=()=>$('about').showModal();$('close-about').onclick=()=>$('about').close();new ResizeObserver(()=>{resizeNetwork();if(catalog)chart();}).observe(canvas);
Promise.all([fetch('/api/catalog').then(r=>{if(!r.ok)throw Error('Experiment is unavailable');return r.json();}),fetch('artifacts/graph.json').then(r=>r.json())]).then(async([data,network])=>{catalog=data;$('tier').innerHTML=Object.entries(data.suites).map(([key,value])=>`<option value="${key}">${value.label}</option>`).join('');$('tier').value=data.default_suite;graph=network;readoutView=createReadout(graph,{onSelect:i=>anatomyView?.selectCell(i)});chooseTier();try{anatomyView=await createAnatomy(canvas,graph,{
 onInspect(){if(busy)stop();if(pending){phase('response','Paused. Replay or Step to resume.',pending.response.action);$('status').textContent='Paused for inspection';}},
 onTime:syncMovement,
 onActivity:frame=>readoutView.update(frame),
 onComplete(){if(pending&&motion&&!motion.committed)commitDecision();}
 });buttons();previewActivity();}catch(e){$('anatomy-loading').hidden=false;$('anatomy-loading').textContent=e.message;console.error(e);}}).catch(e=>$('status').textContent=e.message);
