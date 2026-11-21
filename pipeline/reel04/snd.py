"""Музыка + SFX + голос для ролика «Два одинаковых человека»."""
import sys, numpy as np
sys.argv = ["x", "20.0"]
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
prog=[(62,65,69),(58,62,65),(53,57,60),(60,64,67)]
beat=60/118; t=0.0; i=0
while t<20.0:
    ch=prog[i%4]
    for n in ch: put(MUS,t,pad(n,beat*4+.6),.8)
    put(MUS,t,sub(beat*4,mf(ch[0]-24)),.5)
    t+=beat*4; i+=1
k=0; tt=0.0
while tt<19.6:
    if not (10.6<tt<11.2):
        put(MUS,tt,kick(.5 if tt>4.1 else .3),1)
        if k%2: put(MUS,tt,hat(.5),1)
        if tt>6.5 and k%4==2: put(MUS,tt,snare(.35),1)
    tt+=beat; k+=1
for j,tt in enumerate(np.arange(6.5,15.6,beat/2)):
    put(MUS,tt,pluck([69,72,76,79,76,72,74,77][j%8],.3,.7),.35,.4 if j%2 else -.4)
put(MUS,4.3,riser(2.1),.6); put(MUS,13.6,riser(2.1),.6)
# ---------- SFX ----------
put(FX,.02,zap(.6),.8); put(FX,.05,sub(.8,41),1)
for j in range(3): put(FX,.15+j*.23,thud(.6),.6)
put(FX,2.15,whoosh(.5,.8),.8)
for j in range(14): put(FX,2.4+j*.1,click(.6),.5)
put(FX,3.8,click(),1); put(FX,4.05,whoosh(.5,.7,up=False),.7)
put(FX,5.7,thud(1),1); put(FX,5.72,zap(.5),.6)
put(FX,6.45,whoosh(.6,.9),.9); put(FX,6.6,chime(88),.9)
for tt in (7.1,7.9,8.6,9.4): put(FX,tt,ping(91+((int(tt*10))%3)*2,.8),.7)
put(FX,10.7,whoosh(.6,.8),.8)
put(FX,12.0,whoosh(.4,.6),.6)
for tt in (12.1,12.5,13.0,14.0): put(FX,tt,click(),1); put(FX,tt+.02,ping(93),.5)
put(FX,14.7,chime(96,.9),.9)
put(FX,15.75,whoosh(.6,.9),.9); put(FX,16.1,ping(96),.6)
put(FX,17.55,whoosh(.6,.8),.8); put(FX,18.0,click(),1); put(FX,18.05,ping(98),.9)
put(FX,19.5,whoosh(.6,.6),.7)
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
