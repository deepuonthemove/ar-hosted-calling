"""Legacy /dashboard fallback page and the minimal /test page."""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .dashboard_html import DASHBOARD_HTML

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    try:
        from pathlib import Path
        path = Path(__file__).parent.parent.parent / "static" / "dashboard.html"
        if path.exists():
            return HTMLResponse(path.read_text())
    except Exception:
        pass
    return DASHBOARD_HTML


@router.get("/test", response_class=HTMLResponse)
async def voice_test():
    return """<!DOCTYPE html>
<html><head><title>Voice Test</title></head>
<body style="font-family:sans-serif;max-width:600px;margin:2rem auto;">
<h2>Voice Agent Test</h2>
<button id="toggle" onclick="toggle()" style="padding:1rem 2rem;font-size:1.2rem;background:#2563eb;color:white;border:none;border-radius:8px;cursor:pointer">Start Test</button>
<p id="status" style="margin:1rem 0;font-weight:bold;">Disconnected</p>
<div id="log" style="height:400px;overflow-y:auto;background:#1a1a1a;color:#0f0;padding:1rem;border-radius:8px;font-family:monospace;"></div>
<script>
let ws, stream, ctx, src, proc, playCtx, audioQ = [], playing = false, currentSrc = null;
const logDiv = document.getElementById('log'), btn = document.getElementById('toggle'), st = document.getElementById('status');
let audioRate = 22050;
function lg(m,c){const d=document.createElement('div');d.textContent=m;d.style.color=c||'#888';logDiv.appendChild(d);logDiv.scrollTop=logDiv.scrollHeight}
function toggle(){if(ws&&ws.readyState===1){ws.close();return}connect()}
async function connect(){
  playCtx = new AudioContext();
  ws=new WebSocket('wss://'+location.host+'/ws/test_'+Math.random().toString(36).slice(2));
  ws.binaryType='arraybuffer';
  btn.disabled=true;btn.textContent='Connecting...';st.textContent='Connecting';
  ws.onopen=()=>{btn.textContent='Stop';st.textContent='Connected';startMic()};
  ws.onclose=()=>{btn.textContent='Start Test';st.textContent='Disconnected';stopMic();ws=null};
  ws.onmessage=e=>{
    if(typeof e.data==='string'){const m=JSON.parse(e.data);
      if(m.type==='config')audioRate=m.sample_rate;
      else lg(m.text,m.type==='transcript'?'#8cf':'#fc8');
    }else{
      const v=new Uint8Array(e.data);
      if(v[0]===1){audioQ.push(v.slice(1));if(!playing)playNext();}
      else if(v[0]===2){audioQ=[];if(playing&&currentSrc){playing=false;try{currentSrc.stop()}catch(e){}}}
    }
  };
}
function playNext(){
  if(!audioQ.length||!playCtx){playing=false;return}
  playing=true;
  const total=audioQ.reduce((s,c)=>s+c.length,0);
  const pcm=new Int16Array(total/2);let off=0;
  while(audioQ.length){const c=audioQ.shift();pcm.set(new Int16Array(c.buffer,c.byteOffset,c.length/2),off);off+=c.length/2}
  const buf=playCtx.createBuffer(1,pcm.length,audioRate);
  const ch=buf.getChannelData(0);
  for(let i=0;i<pcm.length;i++)ch[i]=pcm[i]/32768;
  const s=playCtx.createBufferSource();currentSrc=s;
  s.buffer=buf;s.connect(playCtx.destination);
  s.onended=()=>{playing=false;currentSrc=null;if(audioQ.length)playNext()};
  s.start();
}
async function startMic(){
  ctx=new AudioContext({sampleRate:16000});
  stream=await navigator.mediaDevices.getUserMedia({audio:true});
  src=ctx.createMediaStreamSource(stream);
  proc=ctx.createScriptProcessor(4096,1,1);
  proc.onaudioprocess=e=>{if(!ws||ws.readyState!==1)return;const inp=e.inputBuffer.getChannelData(0);const b=new Int16Array(inp.length);for(let i=0;i<inp.length;i++)b[i]=Math.max(-32768,Math.min(32767,inp[i]*32768));ws.send(b.buffer)};
  src.connect(proc);proc.connect(ctx.destination);lg('Mic started','#4c4')
}
function stopMic(){if(proc){proc.disconnect();proc=null}if(src){src.disconnect();src=null}if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}if(ctx){ctx.close();ctx=null}}
</script></body></html>"""
