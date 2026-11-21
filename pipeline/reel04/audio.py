"""Оригинальная музыка и звуковые эффекты для «Миссия: одна капля».
Всё синтезируется кодом: никаких сэмплов, никакой речи.
Выход: music.wav, sfx.wav, mix.wav (48 кГц, стерео)."""
import json, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 48000
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 121.0
rng = np.random.default_rng(7)


def buf():
    return np.zeros((int(DUR * SR) + SR, 2))


MUS, FX = buf(), buf()


def t_(d):
    return np.arange(int(d * SR)) / SR


def env(n, a=0.005, r=0.1, d=None):
    e = np.ones(n)
    na, nr = max(1, int(a * SR)), max(1, int(r * SR))
    e[:na] = np.linspace(0, 1, na)
    if nr < n:
        e[-nr:] *= np.linspace(1, 0, nr)
    if d:
        e *= np.exp(-np.arange(n) / (d * SR))
    return e


def lp(x, f, o=2):
    return sosfilt(butter(o, f, 'low', fs=SR, output='sos'), x)


def hp(x, f, o=2):
    return sosfilt(butter(o, f, 'high', fs=SR, output='sos'), x)


def bp(x, f1, f2, o=2):
    return sosfilt(butter(o, [f1, f2], 'band', fs=SR, output='sos'), x)


def put(b, t, x, g=1.0, pan=0.0):
    i = int(t * SR)
    if i >= len(b):
        return
    x = x[:len(b) - i] * g
    b[i:i + len(x), 0] += x * np.sqrt((1 - pan) / 2)
    b[i:i + len(x), 1] += x * np.sqrt((1 + pan) / 2)


def mf(n):  # midi -> Hz
    return 440 * 2 ** ((n - 69) / 12)


# ---------------- инструменты ----------------
def piano(n, d):
    f, t = mf(n), t_(d + 0.6)
    x = sum(a * np.sin(2 * np.pi * f * k * t * (1 + 0.0004 * k * k)) for k, a in [(1, 1), (2, .45), (3, .22), (4, .12), (5, .06)])
    return x * env(len(t), .003, .3, d=0.9) * 0.35


def xylo(n, d=0.35):
    f, t = mf(n), t_(d)
    x = np.sin(2 * np.pi * f * t) + .35 * np.sin(2 * np.pi * f * 3.93 * t) * np.exp(-t * 40)
    return x * env(len(t), .001, .05, d=0.12) * 0.4


def pluck(n, d=0.5, bright=0.5):  # Карплус–Стронг: пиццикато / щипковые
    f = mf(n)
    N = int(SR / f)
    n_out = int(d * SR)
    ring = lp(rng.uniform(-1, 1, N), 2000 + 6000 * bright)
    out = np.tile(ring, n_out // N + 1)[:n_out]
    t = np.arange(n_out) / SR
    # быстрое затухание верхов + общее затухание — похоже на щипок струны
    out = 0.6 * out * np.exp(-t * 18) + 0.4 * lp(out, f * 2.5) * np.exp(-t * 5)
    out += 0.5 * np.sin(2 * np.pi * f * t) * np.exp(-t * 6)
    return out * env(n_out, .001, .05) * 0.5


def bass(n, d=0.4):
    f, t = mf(n), t_(d)
    x = np.sin(2 * np.pi * f * t) + .5 * np.sin(4 * np.pi * f * t) + .2 * np.sin(6 * np.pi * f * t)
    return x * env(len(t), .004, .08, d=0.35) * 0.55


def flute(n, d):
    f, t = mf(n), t_(d)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.clip(t * 3, 0, 1)
    ph = 2 * np.cumsum(np.pi * f * vib) / SR
    x = np.sin(ph) + .12 * np.sin(2 * ph) + .04 * np.sin(3 * ph)
    x += lp(rng.normal(0, 1, len(t)), 3000) * 0.04
    return x * env(len(t), .06, .12) * 0.28


def brass(n, d, slide=0.0, mute=False):  # тромбон, с глиссандо
    t = t_(d)
    f = mf(n) * 2 ** (slide * t / d / 12)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(k * ph) / k ** (0.9 if not mute else 1.4) for k in range(1, 12))
    x = lp(x, 1800 if mute else 3200)
    return x * env(len(t), .04, .1) * 0.22


def violin(n, d, trem=False):
    t = t_(d)
    f = mf(n) * (1 + .005 * np.sin(2 * np.pi * 6 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = sum(((-1) ** k) * np.sin(k * ph) / k for k in range(1, 16))
    x = bp(x, 300, 5000)
    if trem:
        x *= 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 14 * t))
    return x * env(len(t), .02, .08) * 0.16


def kick(g=1):
    t = t_(.3)
    f = 50 + 120 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 12) * 0.8 * g


def snare(g=1):
    t = t_(.2)
    return (bp(rng.normal(0, 1, len(t)), 1500, 8000) * .6 + np.sin(2 * np.pi * 190 * t) * .4) * np.exp(-t * 22) * 0.5 * g


def hat(g=1):
    t = t_(.06)
    return hp(rng.normal(0, 1, len(t)), 7000) * np.exp(-t * 70) * 0.25 * g


def cymbal(d=1.5, g=1):
    t = t_(d)
    return hp(rng.normal(0, 1, len(t)), 4500) * np.exp(-t * 2.5) * 0.35 * g


def roll(d, g=1):
    t = t_(d)
    x = bp(rng.normal(0, 1, len(t)), 1500, 7000) * (0.6 + 0.4 * np.sin(2 * np.pi * 28 * t))
    return x * np.linspace(.2, 1, len(t)) * 0.3 * g


# ---------------- звуковые эффекты ----------------
def sweep(f0, f1, d, wave='sin', curve=1.0):
    t = t_(d)
    f = f0 + (f1 - f0) * (t / d) ** curve
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) if wave == 'sin' else np.sign(np.sin(ph)) * .5


def boing(g=1):
    t = t_(.8)
    f = 180 * (1 + .35 * np.sin(2 * np.pi * 14 * t) * np.exp(-t * 4)) * (1 + 0.6 * np.exp(-t * 6))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) + .3 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    return x * np.exp(-t * 3.5) * .5 * g


def bonk(g=1):
    t = t_(.25)
    x = np.sin(2 * np.pi * 820 * t) + .6 * np.sin(2 * np.pi * 1350 * t) + .3 * lp(rng.normal(0, 1, len(t)), 3000)
    return x * np.exp(-t * 28) * .5 * g


def thud(g=1):
    t = t_(.5)
    f = 45 + 70 * np.exp(-t * 18)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) + lp(rng.normal(0, 1, len(t)), 400) * .5
    return x * np.exp(-t * 9) * .9 * g


def crash(d=1.2, g=1):
    t = t_(d)
    x = bp(rng.normal(0, 1, len(t)), 300, 9000) * np.exp(-t * 4)
    for k in range(6):  # осколки
        s = int(rng.uniform(.02, d * .6) * SR)
        tt = t_(.08)
        x[s:s + len(tt)] += np.sin(2 * np.pi * rng.uniform(1800, 4500) * tt) * np.exp(-tt * 60) * .5
    th = thud(.8)
    m = min(len(th), len(x))
    x[:m] += th[:m]
    return x * .6 * g


def slide_whistle(f0, f1, d, g=1):
    t = t_(d)
    x = sweep(f0, f1, d, curve=0.8) * (1 + .01 * np.sin(2 * np.pi * 6 * t))
    x += lp(rng.normal(0, 1, len(t)), 2500) * .05
    return x * env(len(t), .03, .05) * .3 * g


def scratch(g=1):
    t = t_(.45)
    f = 900 * (1 - t / .45) ** 2 + 120
    car = np.sin(2 * np.pi * np.cumsum(f) / SR)
    x = bp(rng.normal(0, 1, len(t)), 500, 4000) * (0.5 + 0.5 * car)
    return x * env(len(t), .005, .05) * .7 * g


def splash(d=1.0, g=1):
    t = t_(d)
    x = bp(rng.normal(0, 1, len(t)), 800, 7000) * (np.exp(-t * 3) + .3 * np.abs(np.sin(2 * np.pi * 9 * t)) * np.exp(-t * 2))
    return x * env(len(t), .01, .2) * .6 * g


def pssh(d=1.2, g=1):
    t = t_(d)
    x = bp(rng.normal(0, 1, len(t)), 2500, 11000)
    return x * env(len(t), .01, .25) * (1 + .2 * np.sin(2 * np.pi * 11 * t)) * .45 * g


def poke(g=1):
    t = t_(.12)
    return (np.sin(2 * np.pi * np.cumsum(2400 - 9000 * t) / SR)) * np.exp(-t * 35) * .5 * g


def plink(n=96, g=1):
    t = t_(.6)
    f = mf(n) * (1 + .3 * np.exp(-t * 50))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) * .35 * g


def whoosh(d=.5, g=1, up=True):
    t = t_(d)
    x = rng.normal(0, 1, len(t))
    out = np.zeros_like(x)
    for i, c in enumerate(np.linspace(300, 3000, 8) if up else np.linspace(3000, 300, 8)):
        s = slice(i * len(x) // 8, (i + 1) * len(x) // 8)
        out[s] = bp(x, c * .7, c * 1.4)[s]
    return out * np.sin(np.pi * t / d) * .5 * g


def propeller(d, g=1):
    t = t_(d)
    rate = 18 + 10 * np.sin(np.pi * t / d)
    am = (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(rate) / SR)) ** 3
    return (bp(rng.normal(0, 1, len(t)), 150, 1500) * am + sweep(220, 330, d) * .2) * env(len(t), .1, .3) * .6 * g


def creak(d=.8, g=1):
    t = t_(d)
    pulses = np.zeros(len(t))
    rate = 35 + 25 * np.sin(np.pi * t / d)
    pos = np.cumsum(rate) / SR
    pulses[np.where(np.diff(np.floor(pos)) > 0)[0]] = 1
    x = lp(np.convolve(pulses, np.exp(-np.arange(400) / 60) * np.sin(np.arange(400) * .5))[:len(t)], 2500)
    return x * .5 * g


def tick(g=1):
    t = t_(.03)
    return bp(rng.normal(0, 1, len(t)), 2000, 6000) * np.exp(-t * 200) * .25 * g


def bird(g=1):
    out = []
    for _ in range(rng.integers(2, 5)):
        d = rng.uniform(.06, .14)
        f0 = rng.uniform(2800, 4200)
        t = t_(d)
        out.append(sweep(f0, f0 * rng.uniform(1.2, 1.6), d) * np.sin(np.pi * t / d))
        out.append(np.zeros(int(rng.uniform(.03, .09) * SR)))
    return np.concatenate(out) * .12 * g


def tiptoe(g=1):
    return pluck(rng.choice([50, 52, 55]), .2, .2) * .8 * g


def gulp_up(g=1):  # «вжух» вверх
    return slide_whistle(400, 1800, .45, g)


def heartbeat_drum(g=1):
    return kick(.7 * g)


def drop_ding(g=1):
    return plink(100, g)


# ---------------- музыка ----------------
def seq(t0, notes, bpm, inst, g=1.0, pan=0.0):
    """notes: [(midi | None, доли)]"""
    beat = 60 / bpm
    t = t0
    for n, b in notes:
        if n is not None:
            ns = n if isinstance(n, (list, tuple)) else [n]
            for m in ns:
                put(MUS, t, inst(m, b * beat), g, pan)
        t += b * beat
    return t


def chords(t0, prog, bpm, beats_per, inst, g=1.0, pattern='oompah', pan=0.0):
    beat = 60 / bpm
    t = t0
    for root, third, fifth in prog:
        for k in range(beats_per):
            tt = t + k * beat
            if pattern == 'oompah':
                if k % 2 == 0:
                    put(MUS, tt, inst(root - 12, beat * .9), g, pan)
                else:
                    for m in (root, third, fifth):
                        put(MUS, tt, inst(m, beat * .5), g * .5, pan)
            elif pattern == 'waltz':
                if k % 3 == 0:
                    put(MUS, tt, inst(root - 12, beat * 1.5), g, pan)
                else:
                    for m in (root + 12, third + 12, fifth + 12):
                        put(MUS, tt, inst(m, beat * .6), g * .35, pan)
        t += beats_per * beat
    return t


C, F, G, Am, Dm, E = (60, 64, 67), (53, 57, 60), (55, 59, 62), (57, 60, 64), (50, 53, 57), (52, 56, 59)

# --- Сцена 1: тихое утро (0–15) — фортепианный вальс + флейта ---
chords(0.3, [C, F, C, G, C, F, G, C], 108, 3, piano, .8, 'waltz')
seq(1.1, [(76, 2), (74, 1), (72, 3), (None, 1), (79, 2), (77, 1), (76, 3), (None, 1),
          (74, 1), (76, 1), (77, 1), (79, 2), (76, 1), (72, 4)], 108, flute, 1.0, .2)
for k in range(0, 15):
    put(FX, k + .5, tick(), 1, .6)
for tb in [.8, 2.1, 3.9, 6.2, 8.0, 9.4, 12.5]:
    put(FX, tb, bird(), 1, rng.uniform(-.8, .8))
# вход Бубы: дробь + фанфара тромбона (10–13)
put(MUS, 9.4, roll(.6), .9)
put(MUS, 10.0, cymbal(1.2), .6)
seq(10.0, [((55, 60, 64), .5), (None, .25), ((55, 60, 64), .25), ((57, 62, 65), .5), ((60, 64, 67), 1.5)], 120, lambda n, d: brass(n, d), 1.0)
put(FX, 12.6, whoosh(.3), .6)  # поправляет пояс
put(FX, 12.9, bonk(.4), .5)
# глаза расширяются (15–20): тремоло скрипок, нарастание, удар
put(MUS, 15.0, violin(76, 2.4, trem=True), 1.0, -.3)
put(MUS, 15.0, violin(79, 2.4, trem=True), .8, .3)
put(FX, 17.4, slide_whistle(500, 1600, .5), .8)
put(MUS, 17.6, violin(88, .25), 1.6)  # удивление — короткий звук скрипки
for m in (57, 60, 63):  # драматический аккорд
    put(MUS, 18.2, brass(m, 1.6), .9)
put(MUS, 18.2, kick(1.2), 1)
put(MUS, 18.2, cymbal(1.4), .7)

# --- Сцена 2: водная операция (20–40) — важный марш, тромбон + пиццикато ---
march = [C, C, F, G, C, Am, Dm, G, C, F, G, C]
chords(20.0, march, 112, 2, lambda n, d: bass(n, d) if n < 52 else pluck(n, d, .4), .9, 'oompah')
seq(20.0, [(67, 1), (72, 1), (71, .5), (69, .5), (67, 1), (65, 1), (64, 1), (62, 2)], 112, lambda n, d: brass(n, d, mute=True), .9)
put(FX, 22.3, splash(1.6), 1.2)
put(MUS, 22.3, xylo(84), 1.2)
put(MUS, 22.3, cymbal(1), .5)
put(FX, 23.0, slide_whistle(1400, 500, .6), .7)  # кактус гнётся
put(MUS, 25.6, brass(55, .9, slide=-3, mute=True), 1.1)  # сердитый взгляд «у-у-у»
put(MUS, 27.4, xylo(91, .2), 1.2)  # ДИНЬ — большой палец
put(MUS, 27.6, xylo(96, .4), 1.0)
for tt in (30.3, 30.6, 30.9):
    put(MUS, tt, pluck(72, .15, .1), .7)  # крадущаяся колючка
put(FX, 31.6, poke(), 1.3)
put(FX, 31.8, pssh(1.6), 1.2, -.2)
put(MUS, 31.8, snare(1.2), 1)
put(FX, 35.5, bonk(.6), .6)  # палец в дырку
put(FX, 36.4, pssh(1.0), 1, .4)
put(MUS, 36.4, xylo(88), 1)
put(FX, 37.6, pssh(1.2), 1, -.5)
put(MUS, 37.6, xylo(91), 1)
put(FX, 38.4, splash(1.6), .9)
put(MUS, 38.6, brass(60, 1.3, slide=-12), .9)  # сдувающийся тромбон

# --- Сцена 3: погоня (40–65) — галоп ксилофона 168 bpm ---
gal = [72, 76, 79, 76, 72, 76, 79, 84, 83, 79, 74, 79, 83, 79, 74, 71]
t = 40.0
for rep in range(9):
    t = seq(t, [(n + (5 if rep % 4 == 2 else 0), .5) for n in gal[:8 if rep % 2 else 16]], 168, xylo, .8, .3)
for k in range(int((65 - 40) / (60 / 168))):
    tt = 40 + k * 60 / 168
    put(MUS, tt, (kick(.6) if k % 2 == 0 else snare(.4)), 1)
    put(MUS, tt, bass([48, 55, 53, 55][k // 2 % 4], .2), .8)
put(FX, 40.4, slide_whistle(600, 2000, .5), 1)  # поскользнулся
for k in range(10):  # ноги молотят
    put(FX, 41.0 + k * .09, tiptoe(.6), .6, rng.uniform(-.3, .3))
put(FX, 42.6, whoosh(1.2), .9)  # вращение
put(FX, 46.0, bonk(1.2), 1.2)  # удар о кресло
put(FX, 46.1, creak(1.5), 1)
put(FX, 47.0, propeller(2.6, .6), 1)  # кресло крутится
put(FX, 50.6, slide_whistle(800, 2200, .6), 1)  # вылет
put(FX, 51.8, thud(1.3), 1.2)
put(FX, 51.85, crash(.6), .5)
put(FX, 52.5, boing(1), .7)
put(FX, 55.6, scratch(1.3), 1)  # резкая остановка — record scratch
put(MUS, 56.0, brass(52, 1.0, mute=True), 1.0)
put(FX, 60.2, whoosh(.4), .9)
put(FX, 61.3, gulp_up(1), .9)  # прыжок через столик
put(FX, 62.4, slide_whistle(1600, 500, .4), .9)
put(FX, 63.6, scratch(1.2), 1)

# --- Сцена 4: перчатки (65–90) — крадущийся пиццикато ---
sneak = [(45, .5), (None, .5), (48, .5), (None, .5), (52, .5), (None, .5), (51, .5), (None, .5)]
t = 65.0
while t < 74.5:
    t = seq(t, sneak, 96, lambda n, d: pluck(n, .25, .2), 1.0)
for tt in (66.0, 66.6, 67.2, 67.8):
    put(FX, tt, tiptoe(1.2), 1, -.3)
put(MUS, 68.3, xylo(79, .15), 1, -.7)  # смотрит влево
put(MUS, 69.0, xylo(79, .15), 1, .7)  # смотрит вправо
put(MUS, 70.0, flute(84, 1.2), 1)  # невинное лицо
put(MUS, 71.2, flute(88, 1.0), 1)
put(FX, 72.6, whoosh(.25), 1)
put(MUS, 72.8, cymbal(1.2), .8)  # испуг — резкий акцент
for m in (49, 55, 61, 67):
    put(MUS, 72.8, brass(m, .8), 1.0)
put(FX, 72.8, slide_whistle(400, 2400, .25), .8)
put(MUS, 75.0, violin(91, 4.5, trem=True), .9)  # заморозка
put(FX, 76.0, heartbeat_drum(), 1)
put(FX, 76.35, heartbeat_drum(.7), 1)
put(FX, 77.3, heartbeat_drum(), 1)
put(FX, 77.65, heartbeat_drum(.7), 1)
put(FX, 78.6, creak(.6), .8)  # колючка цепляет перчатку
put(FX, 80.3, poke(1), 1)
put(FX, 80.5, propeller(4.0), 1.2)  # пропеллер
put(MUS, 80.5, roll(4.0), .7)
put(FX, 83.6, whoosh(.8, up=False), 1)
put(FX, 84.4, slide_whistle(1800, 300, .7), .7)  # улетел…
put(FX, 85.4, whoosh(.8), 1)  # …прилетел
put(FX, 87.2, crash(1.4), 1.2)
put(FX, 87.25, boing(1), .8)
put(MUS, 87.2, kick(1.2), 1)

# --- Сцена 5: гениальный план (90–105) — миссия-саспенс ---
for k in range(14):
    tt = 90 + k * .5
    put(MUS, tt, bass([45, 45, 48, 45, 50, 49, 48, 47][k % 8], .3), 1.0)
    put(MUS, tt, hat(.6), 1)
seq(90.0, [(69, .5), (72, .5), (None, .5), (69, .5), (75, .5), (74, .5), (72, 1)], 120, lambda n, d: brass(n, d, mute=True), .7)
put(FX, 91.0, bonk(.5), .6)  # шлем
put(FX, 92.0, tick(2), 1)  # очки
for tt in np.arange(92.5, 95, .45):
    put(FX, tt, tiptoe(.8), .8)
put(MUS, 95.0, brass(48, 5.0, slide=14), 1.1)  # натуга — нарастающее глиссандо
put(MUS, 95.0, roll(5.0, 1.0), .8)
put(FX, 96.5, creak(2.5), .6)
put(FX, 100.2, slide_whistle(1500, 3000, .15), .8)  # срывается
put(FX, 100.4, whoosh(.4, up=False), 1)
put(FX, 100.9, thud(1.4), 1.2)
put(FX, 101.0, boing(1), 1)
put(MUS, 102.6, brass(55, .6, mute=True), 1)  # «уа-уа»
put(MUS, 103.2, brass(52, 1.2, slide=-2, mute=True), 1)

# --- Сцена 6: финал (105–120) ---
seq(105.0, [(79, 1), (77, .5), (76, .5), (74, 1), (72, 2)], 90, flute, 1.0)
chords(105.0, [F, C], 90, 3, piano, .6, 'waltz')
put(FX, 107.2, drop_ding(), 1.2)  # одна капля
put(MUS, 108.0, xylo(96, .4), .8)
put(MUS, 109.0, xylo(91, .2), 1.0)  # большой палец
put(MUS, 109.2, xylo(96, .4), 1.0)
put(FX, 110.4, poke(1.3), 1.3)  # колючка
put(FX, 110.5, gulp_up(1.4), 1.1)  # подпрыгивает
put(MUS, 110.5, cymbal(1.0), .8)
put(MUS, 111.2, violin(93, 1.0, trem=True), 1.0)  # замер в воздухе
put(FX, 112.2, slide_whistle(2200, 300, .8), .9)  # падение
put(FX, 113.0, crash(1.4), 1.2)
put(FX, 113.2, creak(1.0), 1.0)  # кресло складывается
put(FX, 113.6, thud(1), .8)
# хитрая улыбка в камеру + финальный акцент
seq(115.2, [(67, .5), (69, .5), (71, .5), (72, .5)], 120, lambda n, d: pluck(n, .3, .5), 1.0)
for m in (60, 64, 67, 72):
    put(MUS, 117.4, brass(m, .25), 1.0)
put(MUS, 117.4, kick(1.2), 1)
put(MUS, 117.4, snare(1.2), 1)
put(MUS, 117.4, cymbal(2.5), .9)
put(MUS, 117.4, xylo(96, .3), 1)


def norm(b, peak):
    m = np.max(np.abs(b)) or 1
    return b / m * peak


def soft(b):
    return np.tanh(b * 1.2) / np.tanh(1.2)


L = int(DUR * SR)
music = norm(MUS[:L], .55)
sfx = norm(FX[:L], .8)
mix = soft(music * .85 + sfx)
fade = int(2.0 * SR)
for b in (music, sfx, mix):
    b[-fade:] *= np.linspace(1, 0, fade)[:, None]
for name, b in (("music", music), ("sfx", sfx), ("mix", mix)):
    wavfile.write(f"{name}.wav", SR, (np.clip(b, -1, 1) * 32767).astype(np.int16))
print("ok", DUR)
