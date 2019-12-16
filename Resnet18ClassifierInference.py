import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import os
import cv2
import simpleaudio as sa

import librosa
import numpy as np
import matplotlib.pyplot as plt

from ffpyplayer.player import MediaPlayer
import time

test_transforms = transforms.Compose([transforms.Resize(512),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
file_path = 'runtime_videos/'
runtime_path = 'runtime/'
model_path = 'models/resnet18_beatbox_classifier.pth'
results_dict = dict()
genre_list = {0:'DNB', 1:'Dubstep', 2:'Grime', 3:'House', 4:'Reggae', 5:'Trap'}

audio_extract = 'ffmpeg -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = torch.load(model_path)
model_ft = model_ft.to(device)
model_ft.eval()

print('Extracting Audio...')
# Generate Audio Files from Video Inputs
files = os.listdir(file_path)
for f in files:
    os.system(audio_extract.format(file_path+f, runtime_path+f[:-3]+'wav'))


# Generate melspectrogram for audio files
cmap = plt.get_cmap('inferno')
files = os.listdir(runtime_path)
for f in files:
    if '.wav' in f:
        songname = runtime_path + f
        print('Generating melspectrogram for {}'.format(songname))
        y, sr = librosa.load(songname, mono=True, duration=30)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(runtime_path+f[:-3]+'png')
        plt.clf()

files = os.listdir(runtime_path)

for f in files:
    if '.png' in f:
        print('Performing classification for {}'.format(f))
        image = cv2.imread(runtime_path+f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        output = model_ft(input)
        index = output.data.cpu().numpy().argmax()
        results_dict.update({f:genre_list[index]})

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

files = os.listdir(file_path)
playtrack = False
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', (1920, 1080))
    
for f in files:
    print('Playing {}'.format(f))
    time.sleep(1)
    source = cv2.VideoCapture(file_path+f)
    wave_obj = sa.WaveObject.from_wave_file(runtime_path+f[:-3]+'wav')
    spectrogram = cv2.imread(runtime_path+f[:-3]+'png')
    while True:
        ret, frame = source.read()
        if ret:
            frame = overlay_transparent(frame, spectrogram, 10, 90)
            cv2.putText(frame, 'Mel Spectrogram', (70, 60), 
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Genre: {}'.format(results_dict[f[:-3]+'png']), (1600, 60), 
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Result', frame)
            if not playtrack:
                wave_obj.play()
                playtrack = True
        else:
            break
            
        if cv2.waitKey(12) & 0x0F == ord('q'):
            break

    playtrack = False
cv2.destroyAllWindows()