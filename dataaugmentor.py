import os
import subprocess
# ffmpeg -i dnb1.wav -af 'asetrate=44100*0.8,aresample=44100' dnb1-4.wav
cmd = "ffmpeg -i {} -af 'asetrate=44100*{},aresample=44100' {}"

genres = os.listdir(os.getcwd())

for g in genres:
    if os.path.isdir(g):
        allfiles = os.listdir(os.getcwd() + '/' + g)
        counter = 0
        for f in allfiles:
            if '.wav' in f:
                for rate in range(8, 13):
                    print(cmd.format(os.getcwd()+'/'+g+'/'+f, rate/10, os.getcwd()+'/'+g+'/'+f[:-4]+'-'+str(counter)+f[-4:]))
                    os.system(cmd.format(os.getcwd()+'/'+g+'/'+f, rate/10, os.getcwd()+'/'+g+'/'+f[:-4]+'-'+str(counter)+f[-4:]))
                    counter += 1