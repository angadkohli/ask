/* ===== Yatri Personalisation Studio — live canvas preview ===== */
(function(){
  const canvas = document.getElementById('bagCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // ---- State ----
  const state = {
    bag: 'cabin',
    base: 4499,
    color: '#e4572e',
    name: '',
    font: "'Poppins',sans-serif",
    textColor: '#ffffff',
    textSize: 40,
    theme: 'none',
    image: null,        // HTMLImageElement once uploaded
  };

  const PALETTE = ['#e4572e','#1b998b','#3d3a8c','#f2a900','#e15a97','#2b2742','#0f8b8d','#b23a48','#5b8c5a','#222'];
  const TEXT_PALETTE = ['#ffffff','#2b2742','#f2a900','#ffd9c9','#1b998b','#000000'];

  // ---- Build colour swatches ----
  function buildSwatches(el, colors, current, onPick){
    el.innerHTML = '';
    colors.forEach(c=>{
      const s = document.createElement('div');
      s.className = 'swatch' + (c===current ? ' active' : '');
      s.style.background = c;
      s.title = c;
      s.addEventListener('click', ()=>{
        el.querySelectorAll('.swatch').forEach(x=>x.classList.remove('active'));
        s.classList.add('active');
        onPick(c); draw();
      });
      el.appendChild(s);
    });
  }
  buildSwatches(document.getElementById('swatches'), PALETTE, state.color, c=>state.color=c);
  buildSwatches(document.getElementById('textSwatches'), TEXT_PALETTE, state.textColor, c=>state.textColor=c);

  // ---- Inputs ----
  const bagType = document.getElementById('bagType');
  bagType.addEventListener('change', e=>{
    const opt = e.target.selectedOptions[0];
    state.bag = e.target.value;
    state.base = parseInt(opt.dataset.base,10);
    updatePrice(); draw();
  });

  document.getElementById('nameText').addEventListener('input', e=>{
    state.name = e.target.value; updatePrice(); draw();
  });
  document.getElementById('textSize').addEventListener('input', e=>{
    state.textSize = parseInt(e.target.value,10); draw();
  });

  function chipGroup(id, key, after){
    document.getElementById(id).addEventListener('click', e=>{
      const chip = e.target.closest('.chip'); if(!chip) return;
      e.currentTarget.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));
      chip.classList.add('active');
      state[key] = chip.dataset[key] || chip.dataset.font || chip.dataset.theme;
      if(after) after();
      updatePrice(); draw();
    });
  }
  // fonts
  document.getElementById('fontChips').addEventListener('click', e=>{
    const chip = e.target.closest('.chip'); if(!chip) return;
    document.querySelectorAll('#fontChips .chip').forEach(c=>c.classList.remove('active'));
    chip.classList.add('active'); state.font = chip.dataset.font; draw();
  });
  // themes
  document.getElementById('themeChips').addEventListener('click', e=>{
    const chip = e.target.closest('.chip'); if(!chip) return;
    document.querySelectorAll('#themeChips .chip').forEach(c=>c.classList.remove('active'));
    chip.classList.add('active'); state.theme = chip.dataset.theme; updatePrice(); draw();
  });

  // upload
  const imgUpload = document.getElementById('imgUpload');
  const uploadLabel = document.getElementById('uploadLabel');
  imgUpload.addEventListener('change', e=>{
    const file = e.target.files[0]; if(!file) return;
    const reader = new FileReader();
    reader.onload = ev=>{
      const img = new Image();
      img.onload = ()=>{ state.image = img; uploadLabel.textContent = '✅ '+file.name+' — tap to change'; updatePrice(); draw(); };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  });

  // ---- Pricing ----
  function inr(n){ return '₹' + n.toLocaleString('en-IN'); }
  function updatePrice(){
    const hasName = state.name.trim().length>0;
    const hasTheme = state.theme!=='none';
    const hasImg = !!state.image;
    document.getElementById('rowName').style.display  = hasName ? 'flex':'none';
    document.getElementById('rowTheme').style.display = hasTheme? 'flex':'none';
    document.getElementById('rowImg').style.display   = hasImg ? 'flex':'none';
    let total = state.base + (hasName?299:0) + (hasTheme?499:0) + (hasImg?399:0);
    document.getElementById('pBase').textContent  = inr(state.base);
    document.getElementById('pTotal').textContent = inr(total);
    return total;
  }

  // ---- Drawing helpers ----
  // Rounded rect path
  function rr(x,y,w,h,r){
    ctx.beginPath();
    ctx.moveTo(x+r,y);
    ctx.arcTo(x+w,y,x+w,y+h,r);
    ctx.arcTo(x+w,y+h,x,y+h,r);
    ctx.arcTo(x,y+h,x,y,r);
    ctx.arcTo(x,y,x+w,y,r);
    ctx.closePath();
  }
  function shade(hex, amt){
    const n = parseInt(hex.slice(1),16);
    let r=(n>>16)+amt, g=((n>>8)&255)+amt, b=(n&255)+amt;
    r=Math.max(0,Math.min(255,r)); g=Math.max(0,Math.min(255,g)); b=Math.max(0,Math.min(255,b));
    return 'rgb('+r+','+g+','+b+')';
  }

  // Bag geometry depends on type
  function geo(){
    switch(state.bag){
      case 'checkin': return {x:150,y:120,w:300,h:440,r:46};
      case 'duffel' : return {x:110,y:230,w:380,h:230,r:90};
      case 'kids'   : return {x:170,y:150,w:260,h:380,r:50};
      default       : return {x:160,y:130,w:280,h:420,r:44}; // cabin
    }
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    const g = geo();

    // ---- handle (telescopic) ----
    if(state.bag!=='duffel'){
      ctx.strokeStyle = '#2b2742'; ctx.lineWidth = 14; ctx.lineCap='round';
      ctx.beginPath();
      ctx.moveTo(g.x+g.w*0.30, g.y-70); ctx.lineTo(g.x+g.w*0.30, g.y+10);
      ctx.moveTo(g.x+g.w*0.70, g.y-70); ctx.lineTo(g.x+g.w*0.70, g.y+10);
      ctx.stroke();
      ctx.lineWidth = 16;
      ctx.beginPath();
      ctx.moveTo(g.x+g.w*0.30, g.y-70); ctx.lineTo(g.x+g.w*0.70, g.y-70);
      ctx.stroke();
    } else {
      // duffel strap
      ctx.strokeStyle = '#2b2742'; ctx.lineWidth=14; ctx.lineCap='round';
      ctx.beginPath();
      ctx.moveTo(g.x+g.w*0.32, g.y-6);
      ctx.quadraticCurveTo(g.x+g.w*0.5, g.y-80, g.x+g.w*0.68, g.y-6);
      ctx.stroke();
    }

    // ---- body with vertical gradient ----
    const grad = ctx.createLinearGradient(g.x,g.y,g.x+g.w,g.y+g.h);
    grad.addColorStop(0, shade(state.color, 30));
    grad.addColorStop(0.5, state.color);
    grad.addColorStop(1, shade(state.color, -28));
    rr(g.x,g.y,g.w,g.h,g.r); ctx.fillStyle = grad; ctx.fill();

    // clip to body for everything printed on it
    ctx.save();
    rr(g.x,g.y,g.w,g.h,g.r); ctx.clip();

    // ---- print theme ----
    drawTheme(g);

    // ---- ridges / texture ----
    if(state.bag!=='duffel'){
      ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 6;
      for(let i=1;i<=3;i++){
        const lx = g.x + g.w*(i/4);
        ctx.beginPath(); ctx.moveTo(lx, g.y+18); ctx.lineTo(lx, g.y+g.h-18); ctx.stroke();
      }
    }

    // ---- uploaded image (printed on a panel) ----
    if(state.image){
      const pw = g.w*0.6, ph = pw * (state.image.height/state.image.width);
      const px = g.x + (g.w-pw)/2, py = g.y + g.h*0.16;
      // white print panel
      ctx.fillStyle='rgba(255,255,255,.9)';
      rr(px-10,py-10,pw+20,ph+20,14); ctx.fill();
      ctx.drawImage(state.image, px, py, pw, ph);
    }

    // ---- name text ----
    if(state.name.trim()){
      ctx.fillStyle = state.textColor;
      ctx.font = '700 '+state.textSize+"px "+state.font;
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.shadowColor='rgba(0,0,0,.25)'; ctx.shadowBlur=6; ctx.shadowOffsetY=2;
      const ty = state.image ? g.y+g.h*0.82 : g.y+g.h*0.5;
      ctx.fillText(state.name, g.x+g.w/2, ty);
      ctx.shadowColor='transparent';
    }

    ctx.restore();

    // ---- glossy highlight ----
    const hi = ctx.createLinearGradient(g.x,g.y,g.x,g.y+g.h);
    hi.addColorStop(0,'rgba(255,255,255,.22)');
    hi.addColorStop(.25,'rgba(255,255,255,0)');
    rr(g.x,g.y,g.w,g.h,g.r); ctx.fillStyle=hi; ctx.fill();

    // ---- corner guards ----
    ctx.fillStyle='rgba(255,255,255,.45)';
    [[g.x+22,g.y+22],[g.x+g.w-22,g.y+22]].forEach(p=>{
      ctx.beginPath(); ctx.arc(p[0],p[1],10,0,7); ctx.fill();
    });

    // ---- wheels ----
    if(state.bag!=='duffel' || true){
      ctx.fillStyle='#2b2742';
      const wy = g.y+g.h+4;
      [g.x+40, g.x+g.w-40].forEach(wx=>{
        ctx.beginPath(); ctx.arc(wx,wy,16,0,7); ctx.fill();
        ctx.fillStyle='#555'; ctx.beginPath(); ctx.arc(wx,wy,6,0,7); ctx.fill();
        ctx.fillStyle='#2b2742';
      });
    }

    // ---- little brand tag ----
    ctx.fillStyle='rgba(255,255,255,.85)';
    ctx.font="600 16px 'Poppins',sans-serif"; ctx.textAlign='left';
    ctx.fillText('Yatri', g.x+18, g.y+g.h-22);
  }

  // ---- Print themes drawn as subtle patterns ----
  function drawTheme(g){
    if(state.theme==='none') return;
    ctx.save();
    ctx.globalAlpha = 0.16; ctx.strokeStyle='#fff'; ctx.fillStyle='#fff'; ctx.lineWidth=2;
    const step=44;
    if(state.theme==='dots'){
      for(let y=g.y; y<g.y+g.h; y+=step)
        for(let x=g.x; x<g.x+g.w; x+=step){
          ctx.beginPath(); ctx.arc(x+step/2,y+step/2,6,0,7); ctx.fill();
        }
    } else if(state.theme==='stripes'){
      for(let x=g.x-g.h; x<g.x+g.w; x+=34){
        ctx.beginPath(); ctx.moveTo(x,g.y); ctx.lineTo(x+g.h,g.y+g.h); ctx.stroke();
      }
    } else if(state.theme==='mandala'){
      for(let y=g.y+40; y<g.y+g.h; y+=110)
        for(let x=g.x+40; x<g.x+g.w; x+=110) mandala(x,y,30);
    } else if(state.theme==='paisley'){
      for(let y=g.y+40; y<g.y+g.h; y+=90)
        for(let x=g.x+40; x<g.x+g.w; x+=80) paisley(x,y,18);
    }
    ctx.restore();
  }
  function mandala(cx,cy,r){
    for(let i=0;i<12;i++){
      const a=(Math.PI*2/12)*i;
      ctx.beginPath();
      ctx.ellipse(cx+Math.cos(a)*r*0.6, cy+Math.sin(a)*r*0.6, r*0.35, r*0.14, a, 0, 7);
      ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(cx,cy,r*0.22,0,7); ctx.fill();
  }
  function paisley(cx,cy,r){
    ctx.beginPath();
    ctx.moveTo(cx,cy-r);
    ctx.bezierCurveTo(cx+r*1.4,cy-r, cx+r,cy+r*1.6, cx,cy+r);
    ctx.bezierCurveTo(cx-r*0.6,cy+r*0.6, cx-r*0.4,cy-r*0.4, cx,cy-r);
    ctx.stroke();
  }

  // ---- Download ----
  document.getElementById('downloadBtn').addEventListener('click', ()=>{
    const link = document.createElement('a');
    link.download = 'my-yatri-bag.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
    toast('Design downloaded! 🎉');
  });

  // ---- Cart ----
  let cart = 0;
  document.getElementById('addCart').addEventListener('click', ()=>{
    const total = updatePrice();
    cart++;
    document.getElementById('cartCount').textContent = cart;
    toast('Added to cart · '+inr(total));
  });

  // ---- Toast ----
  let toastTimer;
  function toast(msg){
    const t = document.getElementById('toast');
    t.textContent = msg; t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(()=>t.classList.remove('show'), 2200);
  }

  // ---- Menu ----
  const mt=document.getElementById('menuToggle'), nl=document.getElementById('navLinks');
  mt && mt.addEventListener('click',()=>nl.classList.toggle('open'));

  // ---- Init ----
  updatePrice();
  draw();
})();
