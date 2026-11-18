"""Музыка + SFX + голос для ролика «Два одинаковых человека»."""
import sys, numpy as np
sys.argv = ["x", "42.5"]
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

# ---------- музыка: Am – F – C – G, медленно, 70 bpm ----------
prog = [(57,60,64),(53,57,60),(48,52,55),(55,59,62)]
beat = 60/70
t = 0.0; i = 0
while t < 28.4:
    ch = prog[i % 4]
    for n in ch: put(MUS, t, pad(n, beat*4+1.2), 1.0)
    put(MUS, t, sub(beat*4, mf(ch[0]-24)), .6)
    t += beat*4; i += 1
# мягкий пульс с 10.6 до 28.2
k = 0
tt = 10.6
while tt < 28.2:
    put(MUS, tt, kick(.35), 1)
    if k % 2: put(MUS, tt, hat(.5), 1)
    tt += beat; k += 1
# арпеджио-«тикание мысли» 7–24
for j, tt in enumerate(np.arange(7.0, 24.0, beat/2)):
    put(MUS, tt, pluck([69,72,76,72,71,74,79,74][j%8], .35, .6), .35, .4 if j%2 else -.4)
# напряжение 24–28.3, пауза, раскрытие
put(MUS, 24.4, riser(4.0), .9)
put(MUS, 28.45, sub(2.5, 36), 1.2)
for n in (57, 64, 69): put(MUS, 29.3, pad(n, 6.0), 1.2)    # таинственный Am при развороте
# 35.2 «поменяться местами» — мажор, свет
for n in (60, 64, 67, 72): put(MUS, 35.2, pad(n, 7.0), 1.3)
for j, n in enumerate([72, 76, 79, 84]): put(MUS, 35.25 + j*.12, chime(n), .9)
# финал
put(MUS, 37.5, sub(3.0, 41), 1.3)
put(MUS, 37.5, cymbal(3.0), .35)
for n in (48, 60, 64, 67, 72, 76): put(MUS, 37.5, pad(n, 5.0), 1.3)

# ---------- SFX ----------
put(FX, .15, click(), 1); put(FX, .18, zap(.4), .5)
put(FX, .85, click(), 1, .3); put(FX, .88, zap(.4), .5, .3)
put(FX, 4.6, whoosh(.6, .6), .7)
put(FX, 7.0, chime(84), .9, -.3); put(FX, 7.05, chime(91), .5, -.3)
for j in range(8): put(FX, 8.4 + j*.25, tick(1.4), 1, -.3)
put(FX, 10.4, ping(96), 1, -.3)
put(FX, 9.0, whoosh(.5, .4, up=False), .5, .3)
for tt in (10.9, 13.8, 18.7, 24.7): put(FX, tt, paper(), 1); put(FX, tt+.05, whoosh(.35, .4), .6)
put(FX, 15.2, slide_whistle(1200, 600, .5, .5), .6, -.3)
for j in range(6): put(FX, 16.8 + j*.22, paper(.7), 1, .3)
put(FX, 16.1, ping(91, .9), 1, -.3)
put(FX, 17.4, ping(70, .9), 1, .3)
put(FX, 20.1, ping(88), 1, -.4); put(FX, 20.6, ping(93), 1, -.4)
put(FX, 21.9, ping(64, 1.2), 1, .4); put(FX, 22.0, sub(.6, 55), .8, .4)
put(FX, 23.3, zap(1.0), 1, .4)
put(FX, 26.4, ping(81, .8), 1, .3)
put(FX, 29.0, whoosh(.8, .9), 1); put(FX, 29.4, chime(88, .8), .8)
put(FX, 31.7, paper(.8), 1)
for j, tt in enumerate((33.3, 33.6, 34.0)): put(FX, tt, ping(86 + j*3), 1)
put(FX, 35.2, click(), 1, .3); put(FX, 35.25, chime(96, .6), 1, .3)
put(FX, 37.5, whoosh(.7, .8), 1)
put(FX, 38.4, ping(91, .8), 1)

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
fade = int(1.0*SR); mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
for name, b in (("mix", mix), ("music", music), ("sfx", sfx)):
    wf.write(f"{name}.wav", SR, (np.clip(b, -1, 1)*32767).astype(np.int16))
wf.write("voice48.wav", SR, (np.clip(voice, -1, 1)*32767).astype(np.int16))
print("audio ok")
