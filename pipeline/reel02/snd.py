"""Музыка + SFX + голос для ролика «Два одинаковых человека»."""
import sys, numpy as np
sys.argv = ["x", "24.6"]
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


# ---------- музыка: бодро, 112 bpm, Am-F-C-G ----------
prog=[(57,60,64),(53,57,60),(48,52,55),(55,59,62)]
beat=60/112; t=0.0; i=0
while t<24.6:
    ch=prog[i%4]
    for n in ch: put(MUS,t,pad(n,beat*4+.6),.8)
    put(MUS,t,sub(beat*4,mf(ch[0]-24)),.5)
    t+=beat*4; i+=1
k=0; tt=0.0
while tt<24.2:
    if not (6.6<tt<7.0 or 3.2<tt<3.45):
        put(MUS,tt,kick(.5 if tt>6.9 else .3),1)
        if k%2: put(MUS,tt,hat(.5),1)
        if tt>12.5 and k%4==2: put(MUS,tt,snare(.35),1)
    tt+=beat; k+=1
for j,tt in enumerate(np.arange(12.5,19.5,beat/2)):
    put(MUS,tt,pluck([69,72,76,79,76,72,74,77][j%8],.3,.7),.35,.4 if j%2 else -.4)
put(MUS,10.2,riser(2.3),.8)
# ---------- SFX ----------
put(FX,.05,zap(.6),.8); put(FX,.1,click(),1)
for j in range(10): put(FX,.2+j*.11,tick(1.0),.6)
put(FX,1.9,whoosh(.6,.9),1); put(FX,2.5,chime(88),.8)
for j in range(8): put(FX,4.6+j*.1,tick(1.0),.6)
put(FX,5.3,whoosh(.4,.6,up=False),.7); put(FX,6.2,thud(1),1); put(FX,6.25,ping(55,1),.8)
put(FX,6.9,whoosh(.5,.7),.8)
put(FX,8.0,thud(.8),.9)
for tt,n in ((9.3,84),(9.9,88),(10.6,91),(11.6,96)): put(FX,tt,click(),1); put(FX,tt+.02,chime(n,.8),.8)
put(FX,12.55,gulp_up(.8),.8); put(FX,13.95,whoosh(.7,.7),.8); put(FX,15.15,whoosh(.4,.8),.9)
put(FX,16.25,click(),1); put(FX,16.3,zap(.4),.5)
put(FX,17.7,whoosh(.8,.9,up=False),.8); put(FX,18.7,sub(1.0,41),1.2); put(FX,18.72,cymbal(1.5),.4)
put(FX,19.7,whoosh(.6,.8),.9); put(FX,21.6,ping(91),1); put(FX,22.3,ping(96),.9)
put(FX,23.9,whoosh(.6,.6),.7)
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
