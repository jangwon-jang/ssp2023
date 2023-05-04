import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import librosa
import os
import soundfile

cmap_plot = plt.cm.bone_r
minpercentile=80
FIG_SIZE = (14,10)

####################
# 주파수 그림 및 FIR filter 적용
####################
def decibel_sq_safe(x, epsilon=1e-10):
    return 10*np.log10(np.maximum(np.real(x*np.conj(x)),epsilon))

def getFIRFFT(H, order, winname=None) :
    NFFT = (len(H)-1)*2
    # Mirror Image of 'Real Value' & Initialize the imaginary value to 0
    H = np.concatenate((H, H[-2:0:-1])) + np.zeros(NFFT)*(1j)
    
    h = np.fft.ifft(H, NFFT)
    
    order = min(NFFT-1, order)
    h = np.concatenate((h[(len(h)-order//2):], h[0:order//2+1]))
    if winname != None:
        win = librosa.filters.get_window(winname, order+1, fftbins=False)
        h = h*win
    return h.real

def drawspectrogram2(x, Nf, Fs, Ns=None, winname='hamming', minpercentile=80, isdbscale=True, isdraw=True):
    # Short-time Fourier transform with half-overlap 
    # Nf: analysis size
    # Ns: shift size 
    # NFFT: FFT size, power of 2
    if Ns==None: Ns=Nf//2
    num_frames = (len(x)-Nf)//Ns+1  # 마지막 채워지지 않은 프레임은 버린다. 구현에 따라 zero-padding해서 사용 가능
    NFFT = int(2**(np.ceil(np.log2(Nf))))   # Nf보다 크거나 같은 2의 거듭제곱을 NFFT 로 정의
    hNo = NFFT//2+1
    X = np.zeros((hNo,num_frames))
    
    # generate window
    if winname=='rect' or winname=='rectangular':
        win = np.ones(Nf)
    else:   # hamming, hann, etc.
        win = librosa.filters.get_window(winname, Nf, fftbins=True)
        
    # STFT
    for i in range(num_frames):
        y = np.fft.fft(win*x[(i*Ns):(i*Ns+Nf)], n=NFFT)
        y = y[:hNo]
        if isdbscale: y = decibel_sq_safe(y)
        else: y = np.abs(y)
        X[:,i] = y

    if isdraw:
        # 상위 80% 정도만 scale 한다. imshow의 vmin vmax 이용 
        vmax = np.max(X[:])
        vmin = np.percentile(X[:], minpercentile)
        specgram_axis = [0,float(len(x))/float(Fs),0,float(Fs)/2]
        plt.imshow(X, cmap=cmap_plot, aspect='auto', origin='lower', extent=specgram_axis, vmax=vmax, vmin=vmin)
        plt.xlabel('time (seconds)')
        plt.ylabel('frequency (Hz)')
    
    return X


def midian_filter(x, len_filter=5) :
    wav = np.zeros(len(x))
    bi_len_filter = 0
    if len_filter%2 == 1:
        bi_len_filter = 1
    for i in range(len(x)) :
        wav[i] = np.median(np.concatenate([x[i-(len_filter//2):i], x[i:i+(len_filter//2)+bi_len_filter]]))
        
    return wav


###########################################
# noise 구간 에너지 구하기
###########################################
def Energy_Noise(x, Tf, Tn, Ts, fs) :
    #Ns : shift samples, Nf : process samples, Nn : Noise samples
    Ns = int(Ts*fs)
    Nf = int(Tf*fs)
    Nn = int(Tn*fs)
    NFFT = int(2**(np.ceil(np.log2(Nf))))
    hNo = NFFT//2+1
    Pnn = np.zeros(hNo) 
    nframes = int((Nn-Nf)//Ns+1)
    for n in range(nframes):
        y = x[(n*Ns):(n*Ns)+Nf]
        Y = np.fft.fft(y, NFFT)
        aX2 = (Y*np.conj(Y))
        Pnn += aX2[:hNo].real
    Pnn = Pnn/nframes
    return Pnn

############################################
# fir filter 적용
############################################
def FIR_process(x, Tf, Ts, fs, Pnn, order) :
    Ns = int(Ts*fs)
    Nf = int(Tf*fs)
    NFFT = int(2**(np.ceil(np.log2(Nf))))
    hNo = NFFT//2+1
    nframes = int((len(x)-Nf)//Ns+1)
    pre_x = np.zeros(Ns)
    
    win = librosa.filters.get_window('hamming', Nf, fftbins=True)
    x_hat = np.zeros(Ns*nframes)
    for n in range(nframes):
        y = x[(n*Ns):(n*Ns)+Nf]
        Y = np.fft.fft(y, NFFT)
        aX2 = (Y*np.conj(Y))
        
        # |X(w)|^2 / |X(w)+ N(x)|^2
        H_w = (aX2[:hNo].real - Pnn)/aX2[:hNo].real
        # set bound(-3db)
        H_w = np.maximum(H_w,0.224)
        H_w = np.minimum(H_w,1.0)
        
        h = getFIRFFT(np.array(H_w), order, winname='hamming')
        x2 = signal.lfilter(h, [1], x[(n*Ns):(n*Ns)+Nf])
        
        x2 = x2*win
        x_hat[n*Ns:(n+1)*Ns] = x2[:Ns]+pre_x
        pre_x = x2[-Ns:]
        
    return x_hat

###################################
# voice, unvoice 판단
###################################
def V_UV(x, Tf, Ts, fs, Pnn, rate=1.0) :
    Ns = int(Ts*fs)
    Nf = int(Tf*fs)
    NFFT = int(2**(np.ceil(np.log2(Nf))))
    hNo = NFFT//2+1
    nframes = int((len(x)-Nf)//Ns+1)
    VUV = np.zeros(nframes, dtype = float)

    threshold_noise = np.sqrt(np.average(Pnn)) * rate
    for n in range(nframes):
        y = x[(n*Ns):(n*Ns)+Nf]
        Y = np.fft.fft(y, NFFT)
        aX2 = (Y*np.conj(Y))
        if np.sqrt(np.average(aX2[:hNo].real)) > threshold_noise :
            VUV[n] = 1.0
    
    return VUV


###################################
# EPD(end point detection)
###################################
def EPD(x, Tf, Ts, Fs, E_N, VUV) :
    Ns = int(Fs*Ts)
    # threshold는 noise 평균의 1.2 (1.3~1.5까지 유사)
    VUV = V_UV(x, Tf, Ts, Fs, E_N, 1.2)
    # middian filter (앞뒤 2frame 하여 총 5frame)
    VUV_mid = midian_filter(VUV)

    # 음성구간을 찾기 위해 가장 긴 구간 찾기
    max_len = 0
    checking = False
    s_point = 0
    e_point = 0
    for i in range(len(VUV_mid)) :
        if VUV_mid[i] == 1.0 and not checking :
            checking = True
            temp_len = Ns
            temp_start = i*Ns
        elif VUV_mid[i] == 1.0 and checking :
            temp_len = temp_len + Ns
        elif (VUV_mid[i] == 0.0 or i == len(VUV_mid)-1)and checking :
            checking = False
            if temp_len > max_len :
                max_len = temp_len
                s_point = temp_start
                e_point = (i*Ns) -1
                if i == len(VUV_mid)-1 :
                    e_point = ((i+1)*Ns) -1
    
    return VUV, VUV_mid, s_point, e_point


def All_Noise_Avg(x, Tf, Ts, fs, start, end) :
    #Ns : shift samples, Nf : process samples, Nn : Noise samples
    Ns = int(Ts*fs)
    Nf = int(Tf*fs)
    NFFT = int(2**(np.ceil(np.log2(Nf))))
    hNo = NFFT//2+1
    Pnn = np.zeros(hNo) 
    nframes = int((len(x)-Nf)//Ns+1)

    num_noise = 0
    for n in range(nframes):
        if n*Ns >= start and n*Ns <= end:
            continue
        y = x[(n*Ns):(n*Ns)+Nf]
        Y = np.fft.fft(y, NFFT)
        aX2 = (Y*np.conj(Y))
        Pnn += aX2[:hNo].real
        num_noise = num_noise+1
    Pnn = Pnn/num_noise
    return Pnn

def main() :
    speechfile = "gjang-kdigits0-3"
    noisefiles = ['car', 'car2']
    path_save = 'result_jwJang'
    order = 62
    sr = 16000
    Ts = 0.01
    Tf = 0.02

    # (speech and noise) load
    speech, Fs = librosa.load(f'{speechfile}.wav', sr = sr)
    noises = []
    for noisefile in noisefiles :
        noise, Fs = librosa.load(f'{noisefile}.wav', sr = sr)
        noises.append(noise)

    # make mixed wav
    std_s = np.sqrt(np.mean(speech**2))
    mixsnrs = np.array([20, 10, 0, -10])
    std_n = []
    for i in range(len(noises)) :
        std_n.append(np.std(noises[i]))
    std_n = np.array(std_n)
    mixedSigs = []

    for n in range(len(noises)) :
        for snr in mixsnrs:
            gain = np.power(10, -snr/20)
            m = speech + noises[i][:len(speech)]/std_n[i]*std_s*gain
            mixedSigs.append(m)
    
    for num in range(len(mixedSigs)) :
        os.system('mkdir -p %s'%os.path.join(path_save, f'{num}_mixed_file'))
        plt.figure(figsize=FIG_SIZE)
        plt.subplot(5,2,1)
        X=drawspectrogram2(speech, int(Tf*Fs), Fs)
        plt.subplot(5,2,2)
        plt.plot(speech)
        soundfile.write(os.path.join(path_save, f'{num}_mixed_file', '0_speech.wav'), speech, Fs)
        
        plt.subplot(5,2,3)
        X=drawspectrogram2(mixedSigs[num], int(Tf*Fs), Fs)
        plt.subplot(5,2,4)
        plt.plot(mixedSigs[num])
        soundfile.write(os.path.join(path_save, f'{num}_mixed_file', '1_mix.wav'), mixedSigs[num], Fs)

        #0.5s에 대한 noise 계산
        Tn = 0.5
        E_N = Energy_Noise(mixedSigs[num], Tf, Tn, Ts, Fs)
        
        #1.0.5ms에 대한 noise를 이용하여 이를 차감하는 FIR filter 구현
        x_hat = FIR_process(mixedSigs[num], Tf, Ts, Fs, E_N, order)
        plt.subplot(5,2,5)
        X2 =drawspectrogram2(x_hat, int(Tf*Fs), Fs)
        plt.subplot(5,2,6)
        plt.plot(x_hat)
        soundfile.write(os.path.join(path_save, f'{num}_mixed_file', '2_FIR_500ms.wav'), x_hat, Fs)

        # 2. energy 기반 EPD(end point dectection) 구현
        (VUV, VUV_mid, start_point, end_point) = EPD(mixedSigs[num], Tf, Ts, Fs, E_N, 1.2)

        # middian filter 적용한 그림.
        vuv_mid = np.zeros(len(mixedSigs[num]), dtype=float)
        for i in range(len(VUV_mid)) :
            vuv_mid[i*int(Ts*Fs):(i+1)*int(Ts*Fs)] = VUV_mid[i]
        plt.subplot(5,2,7)
        plt.plot(mixedSigs[num])
        plt.plot(vuv_mid)

        # middian filter 적용하지 않은 그림.
        vuv = np.zeros(len(mixedSigs[num]), dtype=float)
        for i in range(len(VUV_mid)) :
            vuv[i*int(Ts*Fs):(i+1)*int(Ts*Fs)] = VUV[i]
        plt.subplot(5,2,8)
        plt.plot(vuv)

        print(f'{num}_mixed wav ({start_point}, {end_point})')
        # 무성자음 및 fading-out 포함(40ms)
        padding = 0.04
        # reference : 후행하는 유,무성자음에 의한 모음의 지속시간 고찰
        start_point = np.maximum(start_point - int(padding*Fs), 0)
        end_point = np.minimum(end_point + int(padding*Fs), len(mixedSigs[num])-1)
        print(f'{num}_mixed wav ({start_point}, {end_point})')

        # 모든 Noise 구간에 대하여 |N(w)|^2 계산
        N_all = All_Noise_Avg(mixedSigs[num], Tf, Ts, Fs, start_point, end_point)
        # 모든 Noise 구간에 대한 |N(w)|^2 으로 FIR filter 적용
        x_hat_2 = FIR_process(mixedSigs[num], Tf, Ts, Fs, N_all, order)
        plt.subplot(5,2,9)
        X2 =drawspectrogram2(x_hat_2, int(Tf*Fs), Fs)
        plt.subplot(5,2,10)
        plt.plot(x_hat_2)
        soundfile.write(os.path.join(path_save, f'{num}_mixed_file', '3_FIR_all_noise.wav'), x_hat_2, Fs)
        plt.savefig(os.path.join(path_save, f'{num}_mixed_file', 'fig.png'))
        plt.close()
    
    

if __name__ == '__main__' :
    main()