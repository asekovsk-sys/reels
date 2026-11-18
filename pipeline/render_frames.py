import sys, subprocess, time
from playwright.sync_api import sync_playwright
FPS=30; DUR=float(sys.argv[1]) if len(sys.argv)>1 else 42.5
only=[float(x) for x in sys.argv[2:]]  # если заданы — только эти моменты в png
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--allow-file-access-from-files','--disable-web-security'])
    pg=b.new_page(viewport={'width':1080,'height':1920})
    errs=[]; pg.on('pageerror',lambda e: errs.append(str(e)))
    pg.goto('file:///home/claude/reel1/anim.html'); pg.evaluate('window.ready'); 
    if errs: print('ERR',errs); sys.exit(1)
    if only:
        for t in only:
            pg.evaluate(f'render({t})'); pg.screenshot(path=f'chk_{t:05.1f}.png',type='png')
        print('ok'); sys.exit()
    ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','image2pipe','-framerate',str(FPS),'-c:v','mjpeg','-i','-',
        '-c:v','libx264','-preset','veryfast','-crf','16','-pix_fmt','yuv420p','video_noaudio.mp4'],stdin=subprocess.PIPE)
    n=int(DUR*FPS); t0=time.time()
    for i in range(n):
        pg.evaluate(f'render({i/FPS})')
        ff.stdin.write(pg.screenshot(type='jpeg',quality=93))
        if i%150==0: print(i,n,round(time.time()-t0),'s',flush=True)
    ff.stdin.close(); ff.wait(); print('done',round(time.time()-t0),'s', errs)
