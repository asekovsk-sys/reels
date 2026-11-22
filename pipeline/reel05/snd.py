"""Музыка + SFX + голос для ролика «Два одинаковых человека»."""
import sys, numpy as np
sys.argv = ["x", "19.8"]
exec(open("audio.py", encoding="utf-8").read().split("# ---------------- музыка")[0])
from scipy.signal import resample_poly

def pad(n, d, g=1.0):  # тёмный синтезаторный пэд: детюн пилы + фильтр
    t = t_(d); f = mf(n)
    x = sum(np.sign(np.sin(2*np.pi*f*(1+k*0.003)*t + k)) * 0.5 + np.sin(2*np.pi*f*(1-k*0.002)*t) for k in (-1, 0, 1))
    x = lp(x, 900)
    return x * env(len(t), 1.2, 1.5) * 0.06 * g

def sub(d, f=41, g=1):
    t = t_(d); return np.sin(2*np.pi*f*t) * env(len(t), .02, .6, d=.9) * .5 * g

def chime(n, g=1):
    t = t_(1.6); f = mf(n)
    x = np.sin(2*np.pi*f*t) + .4*np.sin(2*np.pi*f*2.01*t) + .2*np.sin(2*np.pi*f*3.02*t)
    return x * np.exp(-t*2.8) * .25 * g

def ping(n=88, g=1):
    t = t_(.5); f = mf(n)
    return (np.sin(2*np.pi*f*t) + .3*np.sin(2*np.pi*f*1.5*t)) * np.exp(-t*9) * .35 * g

def click(g=1):
    t = t_(.05); return hp(rng.normal(0,1,len(t)), 1500) * np.exp(-t*120) * .5 * g

def zap(g=1):
    t = t_(.6); f = 120 * (1 + .5*np.sign(np.sin(2*np.pi*35*t)))
    return np.sin(2*np.pi*np.cumsum(f)/SR) * np.exp(-t*4) * (rng.uniform(.4,1,len(t))) * .25 * g

def riser(d, g=1):
    t = t_(d); x = bp(rng.normal(0,1,len(t)), 400, 6000) * (t/d)**2
    return (x + sweep(200, 900, d)*.3*(t/d)) * .35 * g

def paper(g=1):
    t = t_(.25); return bp(rng.normal(0,1,len(t)), 1500, 7000) * np.sin(np.pi*t/.25) * .35 * g


# ---------- музыка: 118 bpm, Dm-Bb-F-C ----------
V=lambda ms: ms/1100.0  # мс голоса -> сек видео
prog=[(62,65,69),(58,62,65),(53,57,60),(60,64,67)]
beat=60/118; t=0.0; i=0
while t<19.8:
    ch=prog[i%4]
    for n in ch: put(MUS,t,pad(n,beat*4+.6),.8)
    put(MUS,t,sub(beat*4,mf(ch[0]-24)),.5)
    t+=beat*4; i+=1
k=0; tt=0.0
slo=(V(9000),V(12000))
while tt<19.4:
    sl=slo[0]<tt<slo[1]
    put(MUS,tt,kick(.5 if tt>V(5300) else .3),1)
    if k%2 and not sl: put(MUS,tt,hat(.5),1)
    if tt>V(5300) and k%4==2 and not sl: put(MUS,tt,snare(.35),1)
    tt+=beat; k+=1
for j,tt in enumerate(np.arange(V(12100),V(17400),beat/2)):
    put(MUS,tt,pluck([69,72,76,79,76,72,74,77][j%8],.3,.7),.35,.4 if j%2 else -.4)
put(MUS,V(3000),riser(1.9),.6); put(MUS,V(10300),riser(1.6),.6)
# ---------- SFX ----------
put(FX,.02,zap(.6),.8); put(FX,.05,sub(.8,41),1); put(FX,.1,ping(91),.7)
for j in range(9): put(FX,V(900)+j*.07,click(.5),.5)
put(FX,V(2250)-.1,whoosh(.5,.8),.8)
for j,ms in enumerate((3100,3300,3500)): put(FX,V(ms),whoosh(.3,.6),.6); put(FX,V(ms)+.05,ping(88+j*3,.8),.7)
put(FX,V(4300),chime(93),1); put(FX,V(4300),thud(.8),.8)
put(FX,V(5300)-.1,whoosh(.6,.9),.9)
put(FX,V(6100),sweep(300,1200,.8),.5); put(FX,V(6800),chime(88),.8)
put(FX,V(8300)-.1,whoosh(.6,.8),.8)
put(FX,V(9000),sweep(900,150,1.0),.5); put(FX,V(9000),sub(1.2,38),.8)
put(FX,V(12100)-.1,whoosh(.6,.9),.9)
put(FX,V(12900),whoosh(.9,1,up=True),.8)
put(FX,V(15900)-.1,whoosh(.5,.7),.7)
for j in range(3): put(FX,V(16100+j*250),click(),1); put(FX,V(16100+j*250)+.02,ping(93+j*2),.6)
put(FX,V(17500)-.1,whoosh(.6,.9),.9); put(FX,V(17800),ping(96),.6)
put(FX,V(19200)-.1,whoosh(.6,.8),.8); put(FX,V(19600),click(),1); put(FX,V(19600)+.05,ping(98),.9)
put(FX,V(21300),whoosh(.6,.6),.7)
# ---------- голос + сведение с «приглушением» музыки ----------
from scipy.io import wavfile as wf
vsr, v = wf.read("voice.wav"); v = v.astype(float) / 32768
v = resample_poly(v, SR, vsr)
V = np.zeros_like(MUS[:, 0]); V[:len(v)] = v[:len(V)]
L = int(DUR*SR)
music = MUS[:L] / (np.max(np.abs(MUS[:L])) or 1) * .5
sfx = FX[:L] / (np.max(np.abs(FX[:L])) or 1) * .55
voice = V[:L] / (np.max(np.abs(V[:L])) or 1) * .95
# огибающая голоса для ducking
envv = np.convolve(np.abs(voice), np.ones(2400)/2400, mode="same")
duck = 1 - .55*np.clip(envv/ .08, 0, 1)
duck = np.convolve(duck, np.ones(4800)/4800, mode="same")
mix = music*duck[:, None]*.9 + sfx*.85 + voice[:, None]
mix = np.tanh(mix*1.1)/np.tanh(1.1)
fade = int(.25*SR); mix[-fade:] *= np.linspace(1, .3, fade)[:, None]
for name, b in (("mix", mix), ("music", music), ("sfx", sfx)):
    wf.write(f"{name}.wav", SR, (np.clip(b, -1, 1)*32767).astype(np.int16))
wf.write("voice48.wav", SR, (np.clip(voice, -1, 1)*32767).astype(np.int16))
print("audio ok")
