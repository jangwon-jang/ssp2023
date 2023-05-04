import numpy as np
from matplotlib import pyplot as plt
import librosa.display
import os
import copy
import soundfile

#from scipy.io import wavfile
from scipy import signal
from scipy.fft import fftshift

cmap_plot = plt.cm.bone_r
minpercentile=80
FIG_SIZE = (15, 15)

# 이전 lab에서 사용한 함수
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

def decibel_sq_safe(x, epsilon=1e-10):
    return 10*np.log10(np.maximum(np.real(x*np.conj(x)),epsilon))

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

#############################################################################################################################
#############################################################################################################################

# 1.probabilstic voice activity detection (VAD) for noise spectrum estimation
    # i.확률 모델을 이용하여 probabilistic voice activity detection 수행
    # ii.각 frame 별로 noise 확률 계산 - find P(voice|y), y 는 한 frame
    # iii.(deterministic decision) 확률값을 thresholding 하여 binary classification, 그리고 검출된 noise frame 들의 평균 제곱 Fourier 성분으로 noise spectrum 예측
    # iv.(soft decision and maximum a posteriori estimation) 각 frame 별로 posterior probability ( P( voice | y ) ) 를 계산함. 그리고 noise spectrum 을 posterior probability로 weighted estimation 한다. E[N] = sum (1-P(v|y)) y
    # v.두 가지 방법(deterministic/soft)으로 추정한 noise spectrum 으로 suppression 한 결과 비교
    # vi.noise 차감은 lab04 의 time-domain Wiener filtering 이용

# short_time function
# return timeframe x_t and |X(w)|^2
def short_time(x, Tf, Ts, Fs ) :
    Nf = int(Tf*Fs)
    Ns = int(Ts*Fs)
    NFFT = int(2**(np.ceil(np.log2(Nf))))
    hNo = NFFT//2+1
    nframes = int((len(x)-Nf)//Ns)
    x_t = []
    X2_w = []
    for k in range(nframes) :
        x_k = x[(k*Ns):(k*Ns)+Nf]
        x_t.append(x_k)
        X = np.fft.fft(x_k, NFFT)
        X2 = (X*np.conj(X))
        X2_w.append(X2[:hNo].real)
    
    return np.array(x_t), np.array(X2_w)

def calcsnr(target, est):
    diff = np.sum((target-est)**2)
    diff = max(diff, 1e-30)   # to prevent divide by zero
    return 10*(np.log10(np.sum(target**2))-np.log10(diff))

# Winerfilter
def wf_process(Y2_w, Nest, x, Ts, Tf, Fs, order) :
    Ns = int(Fs*Ts)
    Nf = int(Fs*Tf)
    pre_x = np.zeros(Ns)
    x_hat = np.zeros(Ns * len(Y2_w))
    win = librosa.filters.get_window('hamming', Nf, fftbins=True)
    for n in range(len(Y2_w)) :
        Y = np.maximum(1e-100, Y2_w[n])
        H = (Y-Nest)/Y
        H = np.maximum(H, 0.224)
        h = getFIRFFT(np.array(H), order, winname='hamming')
        x2 = signal.lfilter(h, [1], x[(n*Ns):(n*Ns)+Nf])
        x2 = x2*win
        x_hat[n*Ns:(n+1)*Ns] = x2[:Ns]+pre_x
        pre_x = x2[-Ns:]
    return x_hat

# 2.time domain VAD
    # i.time domain signal 에 대해서 dual Gaussian mixture model 을 이용하여 probabilistic voice activity detection
#log norm sqare
def log_normpdf_sq(x, var) :
    return -0.5*(np.log(var)+np.log(2*np.pi))-x/(2*var)

# swap parameter
def swap_para(a, b) :
    return copy.deepcopy(b), copy.deepcopy(a)

def VAD_time(pow_x,epochs=10,log_ratio=False,dispstep=1) :
    #initial parameter
    # speech크기가 noise보다 크다고 가정, 시작은 speech와 noise가 절반씩 있다고 가정
    sorted = np.sort(pow_x)
    pr_x = 0.5
    pr_n = 1. - pr_x
    var_x = np.mean(sorted[int(len(sorted)*pr_n):])
    var_n = np.mean(sorted[:int(len(sorted)*pr_n)])

    # epoch 만큼 돌면서 update
    for epoch in range(epochs) :
        # gamma setting
        lfx = log_normpdf_sq(pow_x, var_x)
        lfn = log_normpdf_sq(pow_x, var_n)
        if log_ratio :
            gamma = 1/(1+pr_n/pr_x*np.exp(lfn-lfx))
        else:
            fx = np.exp(lfx)
            fn = np.exp(lfn)
            gamma = pr_x*fx/(pr_x*fx+pr_n*fn)
        
        # display step(defalt=1, everystep)
        if dispstep > 0 :
            # display step or startpoint and endpoint
            if epoch==0 or epoch==epochs-1 or epoch%dispstep==0 :
                print(f'Epoch {epoch+1}: Var(X):{var_x}, Var(N):{var_n}')
                print(f'\t\tPr(X):{pr_x}, Pr(N):{pr_n}')
                plt.plot(gamma, label=(f'epoch:{epoch+1},Pr(X):{pr_x}'))
                plt.xlim(0,len(gamma))
                plt.ylim(-0.1,1.1)
        
        # update parameter
        pr_x = np.mean(gamma)
        pr_n = 1. - pr_x
        var_x = np.sum(gamma*pow_x)/np.sum(gamma)
        var_n = np.sum((1.-gamma)*pow_x)/np.sum(1.-gamma)

        # speech의 크기가 noise보다 크다는 가정하에 서로를 swap한다.
        if var_n > var_x :
            print(f'Waring at epoch {epoch} : VAR(X) < VAR(N) : {var_x} < {var_n}')
            print(f'\tswap parameter VAR(X) and VAR(N), Pr(X) and Pr(N)')
            var_x,var_n = swap_para(var_x, var_n)
            pr_x, pr_n = swap_para(pr_x, pr_n)
    
    return gamma


#3. frequency domain VAD using dual Rayleigh mixture model
    # i. Fourier transform on y(t) -> Y(w)
    # ii. Rayleigh distribution 으로 |X(w)|^2, |N(w)|^2 의 dual Rayleigh mixture model 추정
    # iii. 각 Rayleigh distribution 의 sigma parameter 로 E[N^2] 추정
def log_naive_rayleigh(x, sigmasq):
    return np.log(x)-np.log(sigmasq)-((x*x)/sigmasq)/2

def log_multi_sample_rayleigh(X, sigmasq):
    #sigmasq = np.maximum(sigmasq, 1e-100)
    return np.sum( np.log(1+(X/sigmasq))-((X*X)/sigmasq)/2, axis=-1 )
    #return np.sum( np.log(X)-np.log(sigmasq)-((X*X)/sigmasq)/2, axis=-1 )

def VAD_dual_Rayleigh(X2_w,epochs=10,log_ratio=False,dispstep=1) :
    X_w = np.sqrt(X2_w)
    max_X = np.max(X_w)
    mX2_w = np.mean(X2_w, axis=-1)
    mmX2_w = np.mean(mX2_w)
    sigmasq_X = np.mean(X_w[mX2_w>=mmX2_w,:]**2,axis=0)/2
    sigmasq_N = np.mean(X_w[mX2_w<mmX2_w,:]**2,axis=0)/2
    pr_X = np.sum(mX2_w>=mmX2_w)/len(mX2_w)
    pr_N = 1 - pr_X

    gamma = np.zeros(len(X_w))
    
    # epoch 만큼 돌면서 update
    for epoch in range(epochs) :
        lfX = log_multi_sample_rayleigh(X_w, sigmasq_X)
        lfN = log_multi_sample_rayleigh(X_w, sigmasq_N)

        # gamma setting
        if log_ratio == True:
            ldiff = lfN-lfX
            Il = ldiff<=-100; Ih = ldiff>=100                
            Im = np.logical_and(Il==False, Ih==False)
            gamma[Il] = 1.0
            gamma[Ih] = 0.0
            gamma[Im] = 1/(1+pr_N/pr_X*np.exp(ldiff[Im]))
        else:
            fX = np.exp(lfX)
            fN = np.exp(lfN)
            gamma[:] = pr_X*fX/(pr_X*fX+pr_N*fN)

        # display step(defalt=1, everystep)
        if dispstep > 0 :
            # display step or startpoint and endpoint
            if epoch==0 or epoch==epochs-1 or epoch%dispstep==0 :
                sigma_X = np.mean(sigmasq_X)/max_X
                sigma_N = np.mean(sigmasq_N)/max_X
                print(f'Epoch {epoch+1}: sigma(X):{sigma_X}, sigma(N):{sigma_N}')
                print(f'\t\tPr(X):{pr_X}, Pr(N):{pr_N}')
                plt.plot(gamma, label=(f'epoch:{epoch+1},Pr(X):{pr_X}'))
                plt.xlim(0,len(gamma))
                plt.ylim(-0.1,1.1)

        # update parameter
        sigmasq_X = np.dot(gamma,X2_w)/(2*sum(gamma))
        sigmasq_N = np.dot(1-gamma,X2_w)/(2*sum(1-gamma))
        pr_X = np.sum(gamma)/len(gamma)
        pr_N = 1-pr_X

    return gamma, sigmasq_N


# 4.log-frequency domain VAD using dual LogNormal mixture model
    # i. Fourier transform on y(t) -> Y(w)
    # ii. Compute log PSD - log|Y(w)|^2
    # iii. Gaussian distribution 으로 log|X(w)|^2, log|N(w)|^2 의 mixture model 추정 (mean and variance)
    # iv. noise Gaussian 으로 E[ log|N(w)|^2 ]
    # v. E[|N(w)|^2] = exp E[ log|N(w)|^2 ]
    
def log_naive_normal(x, mu, sigmasq):
    Cn = -0.9189385332046727   # -0.5*np.log(2*np.pi)
    #return Cn - 0.5*np.log(sigmasq)-((x-mu)**2/sigmasq)/2
    return Cn - 0.5*np.log(1+sigmasq)-((x-mu)**2/sigmasq)/2

def log_multi_sample_normal(X, Mu, Sigmasq):
    return np.sum( log_naive_normal(X, Mu, Sigmasq), axis=1 )

def VAD_log_frequency(X2_w,epochs=10,log_ratio=False,dispstep=1):
    logX2 = 0.5*np.log(1+X2_w)
    mlogX2 = np.mean(logX2, axis=-1)
    mmlogX2 = np.mean(mlogX2)
    mu_X = np.mean(logX2[mlogX2>=mmlogX2, :], axis=0)
    mu_N = np.mean(logX2[mlogX2<mmlogX2, :], axis=0)
    print(logX2.shape, mu_X.shape)
    sigmasq_X = np.mean((logX2[mlogX2>=mmlogX2,:]-mu_X)**2,axis=0)
    sigmasq_N = np.mean((logX2[mlogX2>=mmlogX2,:]-mu_N)**2,axis=0)
    pr_X = np.sum(mlogX2>=mmlogX2)/len(mlogX2)
    pr_N = 1-pr_X

    maxlogX2 = np.max(logX2)
    gamma = np.zeros(logX2.shape[0])  
    for epoch in range(epochs) :
        lfX = log_multi_sample_normal(logX2, mu_X, sigmasq_X)
        lfN = log_multi_sample_normal(logX2, mu_N, sigmasq_N)
        
        if log_ratio == True:
            ldiff = lfN-lfX
            Il = ldiff<=-100; Ih = ldiff>=100                
            Im = np.logical_and(Il==False, Ih==False)
            print('gamma 1/~/0 counts = (%d, %d, %d)' % (sum(Il), sum(Im), sum(Ih)))
            gamma[Il] = 1.0
            gamma[Ih] = 0.0
            gamma[Im] = 1/(1+pr_N/pr_X*np.exp(ldiff[Im]))
        else:
            fX = np.exp(lfX)
            fN = np.exp(lfN)
            gamma[:] = pr_X*fX/(pr_X*fX+pr_N*fN)

        # display step(defalt=1, everystep)
        if dispstep > 0 :
            # display step or startpoint and endpoint
            if epoch==0 or epoch==epochs-1 or epoch%dispstep==0 :
                print(f'Epoch {epoch+1}: sigma(X):{np.mean(sigmasq_X)}, Pr(X):{pr_X}')
                plt.plot(gamma, label=(f'epoch:{epoch+1},Pr(X):{pr_X}'))
                plt.xlim(0,len(gamma))
                plt.ylim(-0.1,1.1)

        # update parameter
        mu_X = np.dot(gamma,logX2)/sum(gamma)
        mu_N = np.dot(1-gamma,logX2)/sum(1-gamma)
        sigmasq_X = np.dot(gamma,(logX2-mu_X)**2)/sum(gamma)
        sigmasq_X = np.dot(1-gamma,(logX2-mu_X)**2)/sum(1-gamma)
        pr_X = np.sum(gamma)/len(gamma)
        pr_N = 1- pr_X
    return gamma, mu_N

def main() :
    #######################################
    # option
    #######################################
    sr = 16000
    Tf = 0.02
    Ts = 0.01
    order = 62
    epochs = 10
    path_data = 'input'
    path_result = 'result'
    speech, Fs = librosa.load('gjang-kdigits0-3.wav', sr = sr)
    
    for wavfile in os.listdir(path_data)[3:] :
        if not wavfile.endswith('.wav') :
            continue
        path_save = os.path.join(path_result, wavfile[:-4])
        os.system('mkdir -p %s'%path_save)
        
        wav, Fs = librosa.load(os.path.join(path_data, wavfile), sr = sr)
        x_t, X2_w = short_time(wav, Tf, Ts, Fs)
        E_x = np.mean(x_t*x_t, axis=-1)
        print(x_t.shape, X2_w.shape, E_x.shape)
        print(calcsnr(speech,wav))
        
        #####################################################
        # 2.time domain VAD
        #####################################################
        
        # wav(mixed wav)
        plt.figure(figsize=FIG_SIZE)
        plt.subplot(5,1,1)
        drawspectrogram2(wav, int(Tf*Fs), Fs)
        plt.subplot(5,1,2)
        plt.plot(wav)
        
        # gmm(gaussian)
        plt.subplot(5,1,3)
        gamma = VAD_time(E_x,epochs=epochs,dispstep = 3)
        plt.legend(loc='upper right')
        
        Nest = np.dot(1-gamma,X2_w)/np.sum(1-gamma)
        x_hat = wf_process(X2_w, Nest, wav, Ts, Tf, Fs, order)
        
        # x_hat plot
        plt.subplot(5,1,4)
        drawspectrogram2(x_hat, int(Tf*Fs), Fs)
        plt.subplot(5,1,5)
        plt.plot(x_hat)
        plt.savefig(os.path.join(path_save, '2_time_domain_VAD'))
        soundfile.write(os.path.join(path_save, 'w2_time_domain.wav'), x_hat, Fs)
        plt.close()

        
        #####################################################
        # 3. c using dual Rayleigh mixture model
        #####################################################
        # wav(mixed wav)
        plt.figure(figsize=FIG_SIZE)
        plt.subplot(5,1,1)
        drawspectrogram2(wav, int(Tf*Fs), Fs)
        plt.subplot(5,1,2)
        plt.plot(wav)
        
        # gmm(gaussian)
        plt.subplot(5,1,3)
        gamma, sigmasq_N = VAD_dual_Rayleigh(X2_w,epochs=epochs,dispstep=3)
        plt.legend(loc='upper right')
        
        x_hat = wf_process(X2_w, sigmasq_N, wav, Ts, Tf, Fs, order)
        
        # x_hat plot
        plt.subplot(5,1,4)
        drawspectrogram2(x_hat, int(Tf*Fs), Fs)
        plt.subplot(5,1,5)
        plt.plot(x_hat)
        plt.savefig(os.path.join(path_save, '3_frequency_domain_VAD'))
        soundfile.write(os.path.join(path_save, 'w3_frequency_domain.wav'), x_hat, Fs)
        plt.close()

        #####################################################
        # 4.log-frequency domain VAD using dual LogNormal mixture model
        #####################################################
        # wav(mixed wav)
        plt.figure(figsize=FIG_SIZE)
        plt.subplot(5,1,1)
        drawspectrogram2(wav, int(Tf*Fs), Fs)
        plt.subplot(5,1,2)
        plt.plot(wav)
        
        # gmm(gaussian)
        plt.subplot(5,1,3)
        gamma, mu_N = VAD_log_frequency(X2_w,epochs=epochs,dispstep=3)
        plt.legend(loc='upper right')
        Nest = np.exp(2*mu_N)
        
        x_hat = wf_process(X2_w, Nest, wav, Ts, Tf, Fs, order)
        
        # x_hat plot
        plt.subplot(5,1,4)
        drawspectrogram2(x_hat, int(Tf*Fs), Fs)
        plt.subplot(5,1,5)
        plt.plot(x_hat)
        plt.savefig(os.path.join(path_save, '4_log_frequency_domain_VAD'))
        soundfile.write(os.path.join(path_save, 'w4_log_frequency_domain.wav'), x_hat, Fs)
        plt.close()



if __name__ == '__main__' :
    main()