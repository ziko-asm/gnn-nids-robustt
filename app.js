(() => {
"use strict";
const API = "https://gnn-nids-robustt.onrender.com";
const MODELS = ["mlp","gcn","gat","sage"];
const NAME = {mlp:"MLP",gcn:"GCN",gat:"GAT",sage:"GraphSAGE"};
const DEF_NAME = {none:"None",dropedge:"DropEdge",cosine_prune:"Cosine pruning",adv_train:"PGD adv. training"};
const ATK = {
  feature_pgd:{name:"PGD (features)",x:"ε (feature std)",pct:false},
  gradient:{name:"Gradient",x:"Links flipped",pct:true},
  dice:{name:"DICE",x:"Links flipped",pct:true},
  random_remove:{name:"Random removal",x:"Links removed",pct:true},
  random_add:{name:"Random injection",x:"Links added",pct:true},
};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = {mlp:css("--mlp"),gcn:css("--gcn"),gat:css("--gat"),sage:css("--sage")};
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmt = (v,d=3) => (v==null||isNaN(v)) ? "—" : v.toFixed(d);
const $ = s => document.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
function el(tag, attrs={}, parent){ const e=document.createElementNS(SVGNS,tag); for(const k in attrs) e.setAttribute(k,attrs[k]); if(parent) parent.appendChild(e); return e; }

let S = [], META = null;
const get = (m,d,a,b) => S.find(s => s.model===m && s.defense===d && s.attack===a && Math.abs(s.budget-b)<1e-9);
function series(m,d,a){
  const clean = get(m,d,"clean",0);
  if(!clean) return [];
  if(m==="mlp" && a!=="feature_pgd") return [];
  const pts = S.filter(s => s.model===m && s.defense===d && s.attack===a).sort((x,y)=>x.budget-y.budget);
  return [{...clean,budget:0}, ...pts];
}

/* ============ HERO NETWORK ============ */
const canvas = $("#net"), ctx = canvas.getContext("2d");
let W=0,H=0,DPR=1, nodes=[], edges=[], heroModel="sage", heroEps=0;
const EPS = [0,0.25,0.5,0.75,1.0];
function rng(seed){ return () => { seed=(seed*1664525+1013904223)>>>0; return seed/4294967296; }; }
function buildNet(){
  const r = rng(7), N = 110, K = 5;
  nodes=[]; edges=[];
  const centers = Array.from({length:K},(_,i)=>{const a=i/K*Math.PI*2+0.4; return {x:0.5+0.30*Math.cos(a), y:0.5+0.30*Math.sin(a)};});
  for(let i=0;i<N;i++){
    const c = i%K, gw = i<K;
    const ang=r()*Math.PI*2, rad = gw?0:0.04+r()*0.11;
    nodes.push({c, gw, bot:false, x:centers[c].x+Math.cos(ang)*rad, y:centers[c].y+Math.sin(ang)*rad*0.9, vx:0, vy:0, ph:r()*6.28, ev:0, evT:0, rank:r()});
  }
  const bots = nodes.filter(n=>!n.gw).sort(()=>r()-0.5).slice(0,Math.round(N*0.15));
  bots.forEach(b=>b.bot=true);
  const add=(a,b)=>{ if(a!==b && !edges.some(e=>(e[0]===a&&e[1]===b)||(e[0]===b&&e[1]===a))) edges.push([a,b]); };
  nodes.forEach((n,i)=>{ if(!n.gw) add(i,n.c); });
  for(let k=0;k<K;k++) add(k,(k+1)%K);
  nodes.forEach((n,i)=>{ if(!n.gw && r()<0.6){ const peers=nodes.map((m,j)=>j).filter(j=>nodes[j].c===n.c && j!==i); add(i,peers[Math.floor(r()*peers.length)]); }});
  const bi = nodes.map((n,i)=>n.bot?i:-1).filter(i=>i>=0);
  bi.forEach((i,k)=>{ add(i,bi[(k+1)%bi.length]); if(r()<0.5) add(i,bi[(k+3)%bi.length]);
    const foreign = nodes.map((m,j)=>j).filter(j=>nodes[j].c!==nodes[i].c && !nodes[j].bot); add(i,foreign[Math.floor(r()*foreign.length)]); });
  // bots ranked so that the same bots evade first for every model
  bi.sort((a,b)=>nodes[a].rank-nodes[b].rank).forEach((i,k)=>nodes[i].order=k/bi.length);
}
function resize(){
  const r = canvas.getBoundingClientRect(); DPR = Math.min(2, devicePixelRatio||1);
  W=r.width; H=r.height; canvas.width=Math.round(W*DPR); canvas.height=Math.round(H*DPR);
  ctx.setTransform(DPR,0,0,DPR,0,0); draw(0);
}
function evadedShare(){
  // fraction of bots the model misses at this eps, relative to clean recall
  const c = get(heroModel,"none","clean",0);
  if(!c) return 0;
  if(heroEps===0) return 1-c.recall;
  const a = get(heroModel,"none","feature_pgd",heroEps);
  return a ? 1-a.recall : 0;
}
let t0 = performance.now();
function draw(now){
  if(!W) return;
  const t = (now - t0)/1000;
  ctx.clearRect(0,0,W,H);
  const share = evadedShare();
  const pos = n => { const j = reduceMotion?0:1; return [ (n.x + j*0.004*Math.sin(t*0.6+n.ph))*W, (n.y + j*0.004*Math.cos(t*0.5+n.ph))*H ]; };
  // edges
  ctx.lineWidth=1;
  edges.forEach(([a,b])=>{
    const A=nodes[a],B=nodes[b], [x1,y1]=pos(A),[x2,y2]=pos(B);
    const botEdge = A.bot && B.bot;
    ctx.strokeStyle = botEdge ? "rgba(255,122,107,0.45)" : "rgba(125,139,163,0.22)";
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  });
  // nodes
  const base = Math.max(3.2, Math.min(W,H)/95);
  nodes.forEach(n=>{
    const [x,y]=pos(n);
    const target = n.bot && n.order < share ? 1 : 0;
    n.ev += (target - n.ev) * (reduceMotion?1:0.12);
    const r = n.gw ? base*1.6 : base;
    if(n.bot){
      const k = n.ev;
      // colour blends coral -> benign as the bot evades detection
      const c1=[255,122,107], c2=[93,117,147];
      const c = c1.map((v,i)=>Math.round(v+(c2[i]-v)*k));
      if(k<0.5){ ctx.fillStyle=`rgba(255,122,107,${0.18*(1-k*2)})`; ctx.beginPath(); ctx.arc(x,y,r*2.3,0,Math.PI*2); ctx.fill(); }
      ctx.fillStyle=`rgb(${c})`; ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();
      if(k>0.5){ ctx.strokeStyle="rgba(255,122,107,0.9)"; ctx.lineWidth=1.5; ctx.beginPath(); ctx.arc(x,y,r-0.75,0,Math.PI*2); ctx.stroke(); }
    } else {
      ctx.fillStyle = n.gw ? "#7f95b3" : "#5d7593";
      ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();
    }
  });
  if(!reduceMotion) requestAnimationFrame(draw);
}
function heroBars(){
  const box = $("#hero-bars"); box.innerHTML="";
  MODELS.forEach(m=>{
    const row = heroEps===0 ? get(m,"none","clean",0) : get(m,"none","feature_pgd",heroEps);
    const v = row ? row.f1 : 0;
    const name = document.createElement("span"); name.className="name"; name.textContent=NAME[m];
    if(m===heroModel){ name.style.color="var(--ink)"; name.style.fontWeight="600"; }
    const tr = document.createElement("div"); tr.className="track";
    const f = document.createElement("div"); f.className="fill"; f.style.width=(v*100)+"%"; f.style.background=COL[m]; tr.appendChild(f);
    const num = document.createElement("span"); num.className="num"; num.textContent=fmt(v,2);
    box.append(name,tr,num);
  });
}
function initHero(){
  buildNet(); resize(); new ResizeObserver(resize).observe(canvas);
  const eps = $("#eps");
  eps.addEventListener("input",()=>{ heroEps=EPS[+eps.value]; $("#eps-val").textContent=heroEps.toFixed(2); eps.setAttribute("aria-valuetext",`epsilon ${heroEps}`); heroBars(); if(reduceMotion) draw(performance.now()); });
  document.querySelectorAll(".model-pick .chip").forEach(b=>b.addEventListener("click",()=>{
    heroModel=b.dataset.m; document.querySelectorAll(".model-pick .chip").forEach(x=>x.setAttribute("aria-pressed",String(x===b))); heroBars(); if(reduceMotion) draw(performance.now());
  }));
  heroBars();
  if(!reduceMotion) requestAnimationFrame(draw);
}

/* ============ FINDINGS ============ */
function findings(){
  const f = (m,d,a,b,k="f1") => { const r=get(m,d,a,b); return r?r[k]:NaN; };
  const sage=f("sage","none","clean",0), gcn=f("gcn","none","clean",0), gat=f("gat","none","clean",0), mlp=f("mlp","none","clean",0);
  const pg = m => f(m,"none","feature_pgd",0.5);
  const drop = (m,a) => f(m,"none","clean",0) - f(m,"none",a,0.1);
  const adv = m => [pg(m), f(m,"adv_train","feature_pgd",0.5), f(m,"none","clean",0)-f(m,"adv_train","clean",0)];
  const [sg0,sg1,sgc] = adv("sage");
  const g = $("#findings-grid");
  g.innerHTML = `
  <article class="finding lead">
    <div>
      <p class="big">${fmt(pg("gcn"),2)}<small>GCN F1 under coordinated PGD, down from ${fmt(gcn,2)}</small></p>
      <h3>When bots coordinate, the graph works against the detector</h3>
      <p>If every compromised device shifts its own traffic statistics by half a standard deviation, F1 falls to ${fmt(pg("gcn"),2)} for GCN, ${fmt(pg("gat"),2)} for GAT and ${fmt(pg("sage"),2)} for GraphSAGE. The feature-only MLP keeps ${fmt(pg("mlp"),2)}. Bots are linked to each other, so their small changes add up through message passing: the structure that helps detection also amplifies the evasion.</p>
    </div>
    <div><svg class="chart" id="mini-pgd" viewBox="0 0 520 260" role="img" aria-label="F1 versus PGD budget for each model"></svg></div>
  </article>
  <article class="finding">
    <p class="big">${fmt(sage,2)}<small>GraphSAGE F1, AUC ${fmt(f("sage","none","clean",0,"roc_auc"),2)}</small></p>
    <h3>Only GraphSAGE gets real value from the graph</h3>
    <p>GCN (${fmt(gcn,2)}) only ties the feature-only MLP (${fmt(mlp,2)}), and GAT (${fmt(gat,2)}) falls below it. Bots mostly neighbour benign devices. GCN and GAT average each device with its neighbours and dilute its own signal, while SAGE keeps separate weights for the device and its neighbourhood.</p>
  </article>
  <article class="finding">
    <p class="big">−${fmt(drop("gcn","gradient"),2)}<small>GCN F1, gradient attack on 10% of links</small></p>
    <h3>Structural attacks hurt less, and transfer unevenly</h3>
    <p>Rewiring 10% of links costs 3 to 8 F1 points. The gradient attack, built on a GCN-like surrogate, is the strongest against GCN but loses only ${fmt(drop("sage","gradient"),2)} on GraphSAGE. Label-aware DICE does more damage to GAT (−${fmt(drop("gat","dice"),2)}) and SAGE (−${fmt(drop("sage","dice"),2)}).</p>
  </article>
  <article class="finding" style="grid-column:1/-1">
    <p class="big">${fmt(sg0,2)} → ${fmt(sg1,2)}<small>GraphSAGE F1 under PGD, before and after adversarial training</small></p>
    <h3>No defence is free, and none fixes coordinated evasion</h3>
    <p>PGD adversarial training recovers some robustness but costs ${fmt(sgc*100,0)} points of clean F1 on GraphSAGE and more on GCN and GAT. DropEdge is the best all-round choice for GCN, improving clean F1 and structural robustness together. Cosine pruning helps against the gradient attack on GCN but lowers clean F1 for GraphSAGE. Defending against coordinated feature evasion needs a different approach, for example detecting correlated shifts across neighbouring devices.</p>
  </article>`;
  lineChart($("#mini-pgd"), null, "none", "feature_pgd", "f1", {compact:true});
}

/* ============ LINE CHART ============ */
function lineChart(svg, tip, def, atk, metric, opt={}){
  svg.innerHTML="";
  const vb = svg.viewBox.baseVal, Wd=vb.width, Ht=vb.height;
  const m = opt.compact ? {l:38,r:84,t:12,b:36} : {l:48,r:104,t:14,b:44};
  const iw=Wd-m.l-m.r, ih=Ht-m.t-m.b;
  const models = MODELS.filter(k=>series(k,def,atk).length);
  const all = models.flatMap(k=>series(k,def,atk));
  if(!all.length){ el("text",{x:Wd/2,y:Ht/2,"text-anchor":"middle"},svg).textContent="No data for this combination"; return; }
  const xs=[...new Set(all.map(p=>p.budget))].sort((a,b)=>a-b);
  const xmax = xs[xs.length-1];
  const X = v => m.l + (v/xmax)*iw, Y = v => m.t + (1-v)*ih;
  const pct = ATK[atk].pct;
  // grid
  [0,.25,.5,.75,1].forEach(v=>{ el("line",{x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v),stroke:css("--grid"),"stroke-width":1},svg);
    el("text",{x:m.l-8,y:Y(v)+4,"text-anchor":"end"},svg).textContent=v.toFixed(v%0.5?2:1); });
  xs.forEach(v=>{ el("text",{x:X(v),y:m.t+ih+18,"text-anchor":"middle"},svg).textContent= pct? Math.round(v*100)+"%" : v.toString(); });
  el("text",{x:m.l+iw/2,y:Ht-4,"text-anchor":"middle"},svg).textContent = ATK[atk].x;
  // bands then lines
  const labelY=[];
  models.forEach(k=>{
    const pts = series(k,def,atk).map(p=>({x:p.budget,y:p[metric],s:p[metric+"_std"]||0}));
    const up = pts.map(p=>`${X(p.x)},${Y(Math.min(1,p.y+p.s))}`), dn = pts.slice().reverse().map(p=>`${X(p.x)},${Y(Math.max(0,p.y-p.s))}`);
    el("polygon",{points:up.concat(dn).join(" "),fill:COL[k],opacity:k==="mlp"?0.08:0.12},svg);
  });
  models.forEach(k=>{
    const pts = series(k,def,atk).map(p=>({x:p.budget,y:p[metric]}));
    el("polyline",{points:pts.map(p=>`${X(p.x)},${Y(p.y)}`).join(" "),fill:"none",stroke:COL[k],"stroke-width":2,"stroke-dasharray":k==="mlp"?"6 5":"","stroke-linejoin":"round"},svg);
    pts.forEach(p=>el("circle",{cx:X(p.x),cy:Y(p.y),r:4,fill:COL[k],stroke:css("--surface"),"stroke-width":2},svg));
    const last=pts[pts.length-1]; labelY.push({k,y:Y(last.y)});
  });
  // direct labels, de-collided
  labelY.sort((a,b)=>a.y-b.y); for(let i=1;i<labelY.length;i++) if(labelY[i].y-labelY[i-1].y<15) labelY[i].y=labelY[i-1].y+15;
  labelY.forEach(l=>{ const t=el("text",{x:m.l+iw+10,y:l.y+4,class:"lbl"},svg); t.textContent=NAME[l.k]; t.style.fill=css("--ink-2"); });
  if(!tip) return;
  // hover layer
  const cross = el("line",{y1:m.t,y2:m.t+ih,stroke:"#5a7193","stroke-width":1,opacity:0},svg);
  const hit = el("rect",{x:m.l-10,y:m.t,width:iw+20,height:ih,fill:"transparent",style:"cursor:crosshair"},svg);
  const move = ev => {
    const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left)/r.width*Wd;
    const b = xs.reduce((best,v)=>Math.abs(X(v)-sx)<Math.abs(X(best)-sx)?v:best, xs[0]);
    cross.setAttribute("x1",X(b)); cross.setAttribute("x2",X(b)); cross.setAttribute("opacity",1);
    const rows = models.map(k=>({k,p:series(k,def,atk).find(p=>Math.abs(p.budget-b)<1e-9)})).filter(o=>o.p).sort((a,c)=>c.p[metric]-a.p[metric]);
    tip.innerHTML = `<b>${b===0?"No attack":(pct?Math.round(b*100)+"% of links":"ε = "+b)}</b>` + rows.map(o=>`<div class="row"><span class="k"><i style="background:${COL[o.k]}"></i>${NAME[o.k]}</span><span>${fmt(o.p[metric])} ± ${fmt(o.p[metric+"_std"],2)}</span></div>`).join("");
    const card = svg.parentElement.getBoundingClientRect(); let lx = ev.clientX-card.left+14, ly = ev.clientY-card.top+12;
    tip.style.opacity=1; const tw=tip.offsetWidth; if(lx+tw>card.width-8) lx = ev.clientX-card.left-tw-14; tip.style.left=lx+"px"; tip.style.top=ly+"px";
  };
  hit.addEventListener("pointermove",move); hit.addEventListener("pointerdown",move);
  hit.addEventListener("pointerleave",()=>{tip.style.opacity=0;cross.setAttribute("opacity",0);});
}

/* ============ EXPLORER ============ */
let exAtk="feature_pgd", exDef="none", exMet="f1";
function explorer(){
  const seg=$("#atk"); seg.innerHTML="";
  Object.keys(ATK).forEach(a=>{ const b=document.createElement("button"); b.dataset.v=a; b.textContent=ATK[a].name; b.setAttribute("aria-pressed",String(a===exAtk)); seg.appendChild(b); });
  const bindSeg=(box,fn)=>box.addEventListener("click",e=>{ const b=e.target.closest("button"); if(!b) return; box.querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",String(x===b))); fn(b.dataset.v); });
  bindSeg(seg,v=>{exAtk=v;renderCurve();});
  bindSeg($("#met"),v=>{exMet=v;renderCurve();});
  $("#def").addEventListener("change",e=>{exDef=e.target.value;renderCurve();});
  $("#curve-table-btn").addEventListener("click",e=>{ const t=$("#curve-table"), open=t.hidden; t.hidden=!open; e.target.setAttribute("aria-expanded",String(open)); e.target.textContent=open?"Hide table":"Show as table"; });
  renderCurve();
}
function renderCurve(){
  const mName={f1:"F1",recall:"Detection rate (recall)",roc_auc:"ROC-AUC"}[exMet];
  $("#curve-title").textContent = `${mName} under ${ATK[exAtk].name.toLowerCase()} attack`;
  const noMlp = exAtk!=="feature_pgd" ? " The MLP ignores links, so structural attacks cannot affect it and it is omitted." : "";
  const noDef = exDef!=="none" && exDef!=="adv_train" ? " Graph defences do not apply to the MLP." : "";
  $("#curve-sub").textContent = `Defence: ${DEF_NAME[exDef]}. Attack-class metric on held-out devices.${noMlp}${noDef}`;
  const models = MODELS.filter(k=>series(k,exDef,exAtk).length);
  $("#curve-legend").innerHTML = models.map(k=>`<span><i style="border-color:${COL[k]};${k==="mlp"?"border-top-style:dashed":""}"></i>${NAME[k]}</span>`).join("");
  lineChart($("#curve"), $("#curve-tip"), exDef, exAtk, exMet);
  // table view
  const xs=[...new Set(models.flatMap(k=>series(k,exDef,exAtk).map(p=>p.budget)))].sort((a,b)=>a-b);
  const pct=ATK[exAtk].pct;
  let h=`<table><thead><tr><th>Model</th><th>Defence</th>${xs.map(b=>`<th>${b===0?"Clean":(pct?Math.round(b*100)+"%":"ε "+b)}</th>`).join("")}</tr></thead><tbody>`;
  models.forEach(k=>{ const s=series(k,exDef,exAtk); h+=`<tr><td>${NAME[k]}</td><td>${DEF_NAME[exDef]}</td>${xs.map(b=>{const p=s.find(q=>Math.abs(q.budget-b)<1e-9);return `<td>${p?fmt(p[exMet]):"—"}</td>`;}).join("")}</tr>`; });
  $("#curve-table").innerHTML=h+"</tbody></table>";
}

/* ============ TRADE-OFF SCATTER ============ */
let trAtk="feature_pgd";
function tradeoff(){
  const svg=$("#trade"), tip=$("#trade-tip"); svg.innerHTML="";
  const budget = trAtk==="feature_pgd"?0.5:0.1;
  const pts=[];
  MODELS.forEach(k=>["none","dropedge","cosine_prune","adv_train"].forEach(d=>{
    const c=get(k,d,"clean",0), a=get(k,d,trAtk,budget); if(c&&a) pts.push({k,d,x:c.f1,y:a.f1});
  }));
  const Wd=440,Ht=340,m={l:46,r:16,t:12,b:44}, iw=Wd-m.l-m.r, ih=Ht-m.t-m.b;
  const x0=0.4,x1=0.9, ymax = trAtk==="feature_pgd"?0.5:0.9, y0 = trAtk==="feature_pgd"?0:0.4;
  const X=v=>m.l+(v-x0)/(x1-x0)*iw, Y=v=>m.t+(1-(v-y0)/(ymax-y0))*ih;
  for(let v=y0; v<=ymax+1e-9; v+=0.1){ el("line",{x1:m.l,x2:m.l+iw,y1:Y(v),y2:Y(v),stroke:css("--grid")},svg); el("text",{x:m.l-8,y:Y(v)+4,"text-anchor":"end"},svg).textContent=v.toFixed(1); }
  for(let v=x0; v<=x1+1e-9; v+=0.1){ el("text",{x:X(v),y:m.t+ih+18,"text-anchor":"middle"},svg).textContent=v.toFixed(1); }
  el("text",{x:m.l+iw/2,y:Ht-4,"text-anchor":"middle"},svg).textContent="Clean F1";
  const yl=el("text",{x:12,y:m.t+ih/2,"text-anchor":"middle",transform:`rotate(-90 12 ${m.t+ih/2})`},svg); yl.textContent="F1 under attack";
  const SHAPE={none:"circle",dropedge:"square",cosine_prune:"diamond",adv_train:"triangle"};
  const draw=(p)=>{ const cx=X(p.x), cy=Y(p.y), s=6, g=el("g",{},svg); let e;
    if(SHAPE[p.d]==="circle") e=el("circle",{cx,cy,r:s},g);
    else if(SHAPE[p.d]==="square") e=el("rect",{x:cx-s+1,y:cy-s+1,width:2*s-2,height:2*s-2,rx:2},g);
    else if(SHAPE[p.d]==="diamond") e=el("polygon",{points:`${cx},${cy-s-1} ${cx+s+1},${cy} ${cx},${cy+s+1} ${cx-s-1},${cy}`},g);
    else e=el("polygon",{points:`${cx},${cy-s-1} ${cx+s+1},${cy+s} ${cx-s-1},${cy+s}`},g);
    e.setAttribute("fill",COL[p.k]); e.setAttribute("stroke",css("--surface")); e.setAttribute("stroke-width",2);
    const hit=el("circle",{cx,cy,r:14,fill:"transparent",tabindex:0,role:"img","aria-label":`${NAME[p.k]}, ${DEF_NAME[p.d]}: clean ${fmt(p.x,2)}, attacked ${fmt(p.y,2)}`},g);
    const show=(ev)=>{ tip.innerHTML=`<b>${NAME[p.k]} · ${DEF_NAME[p.d]}</b><div class="row"><span class="k">Clean F1</span><span>${fmt(p.x)}</span></div><div class="row"><span class="k">Under attack</span><span>${fmt(p.y)}</span></div>`;
      const card=svg.parentElement.getBoundingClientRect(), r=svg.getBoundingClientRect(); let lx=r.left-card.left+cx/Wd*r.width+12, ly=r.top-card.top+cy/Ht*r.height+10; tip.style.opacity=1; if(lx+tip.offsetWidth>card.width-8) lx-=tip.offsetWidth+24; tip.style.left=lx+"px"; tip.style.top=ly+"px"; };
    hit.addEventListener("pointerenter",show); hit.addEventListener("pointerdown",show); hit.addEventListener("focus",show);
    hit.addEventListener("pointerleave",()=>tip.style.opacity=0); hit.addEventListener("blur",()=>tip.style.opacity=0);
  };
  pts.forEach(draw);
  // shape legend
  const lg=el("g",{transform:`translate(${m.l+8},${m.t+6})`},svg);
  ["none","dropedge","cosine_prune","adv_train"].forEach((d,i)=>{ const y=i*16; const cx=6, cy=y+6, s=4.5;
    let e; if(SHAPE[d]==="circle") e=el("circle",{cx,cy,r:s},lg); else if(SHAPE[d]==="square") e=el("rect",{x:cx-s,y:cy-s,width:2*s,height:2*s,rx:1.5},lg);
    else if(SHAPE[d]==="diamond") e=el("polygon",{points:`${cx},${cy-s-1} ${cx+s+1},${cy} ${cx},${cy+s+1} ${cx-s-1},${cy}`},lg); else e=el("polygon",{points:`${cx},${cy-s-1} ${cx+s+1},${cy+s} ${cx-s-1},${cy+s}`},lg);
    e.setAttribute("fill",css("--ink-2")); el("text",{x:18,y:cy+4},lg).textContent=DEF_NAME[d]; });
}
function initTrade(){
  $("#trade-atk").addEventListener("click",e=>{ const b=e.target.closest("button"); if(!b) return; $("#trade-atk").querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",String(x===b))); trAtk=b.dataset.v; tradeoff(); });
  tradeoff();
}

/* ============ MAIN TABLE ============ */
function mainTable(){
  const cols=[["clean",0,"Clean"],["random_remove",0.1,"Random"],["dice",0.1,"DICE"],["gradient",0.1,"Gradient"],["feature_pgd",0.5,"PGD"]];
  const defs={mlp:["none","adv_train"],gcn:["none","dropedge","cosine_prune","adv_train"],gat:["none","dropedge","cosine_prune","adv_train"],sage:["none","dropedge","cosine_prune","adv_train"]};
  let h=`<table><thead><tr><th>Model</th><th>Defence</th>${cols.map(c=>`<th>${c[2]}</th>`).join("")}</tr></thead><tbody>`;
  MODELS.forEach(k=>defs[k].forEach(d=>{ h+=`<tr><td>${NAME[k]}</td><td>${DEF_NAME[d]}</td>${cols.map(([a,b])=>{const r=get(k,d,a,b); return `<td>${r?fmt(r.f1,2)+`<span style="color:var(--muted)"> ±${fmt(r.f1_std,2)}</span>`:"—"}</td>`;}).join("")}</tr>`; }));
  $("#main-table").innerHTML=h+"</tbody></table>";
}

/* ============ LIVE API (shown only when reachable) ============ */
async function probeApi(){
  try{
    const ctl=new AbortController(); setTimeout(()=>ctl.abort(),6000);
    const r=await fetch(API+"/health",{signal:ctl.signal}); if(!r.ok) return;
    const j=await r.json(); if(j.status!=="ok") return;
    $("#live").classList.add("on");
    $("#lv-run").addEventListener("click",runLive);
  }catch(e){ /* API asleep or offline: the static results above are complete */ }
}
async function runLive(){
  const out=$("#lv-out"), btn=$("#lv-run"); btn.disabled=true;
  const atk=$("#lv-atk").value;
  out.textContent="Training and attacking… the first run of a configuration takes up to a minute.";
  try{
    const r=await fetch(API+"/analyze",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({model:$("#lv-model").value,defense:"none",attack:atk,budget:atk==="feature_pgd"?0.5:0.1,nodes:300,seed:+$("#lv-seed").value})});
    if(!r.ok) throw new Error((await r.json()).detail||r.statusText);
    const j=await r.json();
    out.innerHTML=`F1 dropped from <b style="color:var(--ink)">${fmt(j.clean.f1)}</b> to <b style="color:var(--ink)">${fmt(j.attacked.f1)}</b>, and detection rate from ${fmt(j.clean.recall)} to ${fmt(j.attacked.recall)}. ${j.evaded_test_bots.length} test bots that were caught before the attack now pass as benign. ${j.links_added.length+j.links_removed.length?`The attack changed ${j.links_added.length+j.links_removed.length} links.`:""}`;
  }catch(e){ out.innerHTML=`<span class="error">The run failed: ${String(e.message||e)}. Try again in a minute; the API may still be starting.</span>`; }
  btn.disabled=false;
}

/* ============ BOOT ============ */
fetch("results/results.json").then(r=>{ if(!r.ok) throw new Error(r.status); return r.json(); }).then(j=>{
  S=j.summary; META=j.meta;
  const g=META.graph, runs=j.records.length;
  $("#runs-line").textContent=`${META.seeds.length} seeds, ${runs.toLocaleString("en-US")} measured runs`;
  $("#setup-graph").textContent=`Synthetic IoT network, ${Math.round(g.nodes).toLocaleString("en-US")} devices, ${Math.round(g.links).toLocaleString("en-US")} links on average, ${Math.round(g.attack_share*100)}% compromised`;
  initHero(); findings(); explorer(); initTrade(); mainTable(); probeApi();
}).catch(err=>{
  $("#findings-grid").innerHTML=`<p class="error">Could not load results/results.json (${err.message}). If you opened this file directly, serve the folder instead: python -m http.server</p>`;
});
})();
