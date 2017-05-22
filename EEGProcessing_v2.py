__author__ = 'Irina Knyazeva'


import pandas as pd
import numpy as np
from numpy.fft import fft,ifft

class EEGAnalyser:

    wt = []
    data =  pd.DataFrame()
    freq = []
    good_trials = []
    norm_data = []
    min_length = 0
    phase_syncr = {}
    spect_syncr = {}
    num_trials = 0


    def __init__(self, srate, min_freq ,max_freq, num_freq):

        """
        Define sample rate and parameters for frequencies analysis
        """
        self.srate = srate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_freq = num_freq

    def load_data(self, sig_name, t_name, ftr_name):
        '''
        function for loading, segmenting and cutting out failed trials
        sig_name: name of the csv file with the signal
        t_name: name of the TXT file with timing
        ftr_name: name of the txt file with labels of failed trials
        srate: sampling rate of signal
        '''

        #load csv file
        df = pd.read_csv(sig_name, sep = ',', header=0)

        p =np.loadtxt(ftr_name)
        #загружаем файл с номерами неудачных проб
        #преобразуем номера неудачных проб в int
        p = p.astype(int)
        #загружаем файл с метками начала новой пробы по времени
        t = np.loadtxt(t_name, skiprows=1, usecols=(0,))
        #переводим метки из секунд в отсчеты (2 ms)
        t = t*self.srate
        #преобразуем номера отсчетов в int
        t = t.astype(int)
        #помечаем отсчеты до первой пробы как -1
        #df.iloc[:t[0], 19] = -1
        df['LABEL'][:t[0]] = -1
        #помечаем все остальные отсчеты номерами проб, к которым они относятся
        for i in range(len(t)-1):
            #df.iloc[t[i]:t[i+1], 19] = i+1
            df['LABEL'][t[i]:t[i+1]] = i+1
        #помечаем последнюю пробу
        #df.iloc[t[-1]:, 19] = len(t)
        df['LABEL'][t[-1]:] = len(t)
        #копируем данные (не знаю зачем, на всякий случай)
        df_copy = df
        #удаляем неудачные пробы
        for k in range(len(p)):
            #df_copy.drop(df_copy.index[df_copy['LABEL'] == p[k]], inplace = True)
            df_copy.drop(df_copy.index[df_copy['LABEL'] == p[k]], inplace = True)
        #удаляем отсчеты до начала первой пробы
        df_copy.drop(df_copy.index[df_copy['LABEL'] == -1], inplace = True)

        self.data = df_copy

    def normalize_data(self):
        """
        Function for data normalization: concatenating all trials of equal min_trial_length
        in one series
        """
        N = np.array(self.data['LABEL'].unique())
        sig  = self.data.drop('LABEL', 1)
        #L = sig[:, 19].real.astype(int)
        #sig = np.delete(sig, 19, 1)
        #N = np.unique(L)
        self.good_trials = N
        self.num_trials = len(N)

        self.min_length = int(min([sum(self.data['LABEL'] == i) for i in N]))

        for i in N:
            sig.drop(self.data.loc[self.data['LABEL'] == i][self.min_length:(self.data.loc[self.data['LABEL'] == i].shape[0])].index, inplace=True)

        self.norm_data = sig.as_matrix()
   
"""       #calculating minimal trial length
        if (self.min_length == 0):
            trial_lengths = np.zeros(len(N))
            for i in range(0, len(N)):
                trial_lengths[i] = np.where(L == N[i])[0].shape[0]
            self.min_length = int(np.amin(trial_lengths))
"""
        #Собираем все триалы в одну серию, но сразу режем на куски равной длины, чтобы потом было усреднять проще
        #количество проб использованных в данном конкретном случае
"""        for k in np.arange(0, self.num_trials):
            start = np.where(L == N[k])[0][0] #начало пробы
            if (k==0):
                alldata = sig[start:(start+self.min_length), :]
            else:
                alldata = np.vstack((alldata,sig[start:(start+self.min_length), :]))
"""  


    def wavelet_transform(self, data, family):

        """
        Function for wavelet transform computing
        return wt: wavelet transformation for time series or matrix
        freq: vector of used frequencies in Hz
        Function for wavelet transform computing
        data: time series (points x electrodes)
        srate: sampling rate in Hz
        min_freq: minimal frequency of wavelet
        max_freq: max frequency of wavelet
        num_freq: number of frequency in interval from min to max
        family: name of wavelet family
        """

        freq = np.logspace(np.log10(self.min_freq),np.log10(self.max_freq),self.num_freq)
        range_cycles = [4, 8]
        s = np.logspace(np.log10(range_cycles[0]),np.log10(range_cycles[1]),self.num_freq)/(2*np.pi*freq)
        wavtime = np.arange(-1,1+1/self.srate,1/self.srate)
        half_wave =  int((len(wavtime)-1)//2)

        nWave = wavtime.shape[0]
        nData = data.shape[0]
        nConv = nWave + nData - 1
        wt = np.zeros(shape = (self.num_freq,nData, data.shape[1]), dtype=complex)

        for el in range(0, data.shape[1]):
            for fi in range(0, self.num_freq):
                wavelet = np.exp(2*1j*np.pi*freq[fi]*wavtime)* np.exp(-np.power(wavtime,2)/(2*np.power(s[fi],2)))
                waveletX = fft(wavelet,nConv)
                dataX = fft(data[:, el],nConv)
                convData = ifft(waveletX * dataX)
                convData = convData[half_wave:-half_wave]
                wt[fi, :, el] = convData

        self.wt = wt
        self.freq = freq




    def phase_coherence(self, chanel1_id, chanel2_id, num_points = 300):
        """

        num_points: num_points for computing
        Function for wavelet coherence  computing
        based on experiment results
        now accept data after wavelet transform
        """

        #length of time window for phase averaging from 1.5 for lowest freq to 3 cycles
        timewindow = np.linspace(1.5,3,self.num_freq)
        #length of the largest time window in points
        time_window_largest = np.floor((1000/self.freq[0])*timewindow[0]/(1000/self.srate)).astype(int)
        times2saveidx = np.linspace(time_window_largest,self.min_length-time_window_largest,num_points).astype(int)


        ispc = np.zeros(shape = (self.num_freq,len(times2saveidx)), dtype=float)

        data1 = self.wt[:,:,chanel1_id]
        data2 = self.wt[:,:,chanel2_id]
        for fi in range(0, self.num_freq):

            phase_sig1 = np.angle(data1[fi, :].reshape(self.min_length,self.num_trials,order='F'))
            phase_sig2 = np.angle(data2[fi, :].reshape(self.min_length,self.num_trials,order='F'))

            #Phase difference between the signals
            phase_diffs = phase_sig1-phase_sig2

            #Averaging in the sliding window
            #compute time window in indices for this frequency and average inside the window
            time_window_idx = np.floor((1000/self.freq[fi])*timewindow[fi]/(1000/self.srate)).astype(int)
            for ti in range(0,len(times2saveidx)):
                phasesynch = abs(np.mean(np.exp(1j*phase_diffs[times2saveidx[ti]-time_window_idx:times2saveidx[ti]+time_window_idx,:]),axis = 0))

                # then average over trials
                ispc[fi,ti] = np.mean(phasesynch)


        self.phase_syncr  = {'ispcs':ispc,'chanIds':[chanel1_id,chanel2_id],'times': times2saveidx}


    #Computing phase coherence

    # Spectral Coherence (Magnitude-Squared Coherence)
    def spectral_coher(self, chanel1_id, chanel2_id, num_points = 300):

        """
        compute spectral coherence by wavelet transformed data
        """

        spectcoher = np.zeros(shape=(self.num_freq, num_points), dtype=float)
        times2saveidx = np.linspace(0,self.min_length-1,num_points).astype(int)

        data1 = self.wt[:,:,chanel1_id]
        data2 = self.wt[:,:,chanel2_id]

        for fi in range(0, self.num_freq):
            sig1 = data1[fi, :].reshape(self.min_length,self.num_trials,order='F')
            sig2 = data2[fi, :].reshape(self.min_length,self.num_trials,order='F')

            spec1 = np.mean(np.power(abs(sig1),2), 1)
            spec2 = np.mean(np.power(abs(sig1),2), 1)
            cross_spec = np.power(abs(np.mean(sig1*np.conj(sig2), 1)), 2)
            spectcoher[fi, :] = cross_spec[times2saveidx]/(spec1[times2saveidx] * spec2[times2saveidx])

        self.spect_syncr  = {'spc':spectcoher,'chanIds':[chanel1_id,chanel2_id],'times': times2saveidx}



