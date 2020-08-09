from bitstring import BitArray
import struct
import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

class Fsk:
    def __init__(self, signal, rate, f_div, Fs, f_base=0):
        self.signal = signal
        self.rate = rate
        self.chip_l = int(Fs / rate)
        self.f_div = f_div
        self.Fs = Fs
        self.f_base = f_base
        self.Ts = 1/Fs
        self.sig_size = self.chip_l * len(signal)
        self.signal_duration = self.sig_size * self.Ts
        self.dt = np.arange(0, self.signal_duration, self.Ts)
        self.symbol_duration = np.arange(0, self.chip_l*self.Ts, self.Ts)

        f1 = f_base + f_div
        f2 = f_base - f_div
        self.f1 = np.exp(1j*2*np.pi*f1*self.symbol_duration)
        self.f2 = np.exp(1j*2*np.pi*f2*self.symbol_duration)

    def __repr__(self):
        return f'sample-rate={self.Fs}, rate={self.rate}, chip_l={self.chip_l}'

    def calc_fsk(self):
        signal = np.repeat(self.signal, self.chip_l)
        y = np.exp(1j*2*np.pi*(self.f_base + signal*self.f_div)*self.dt)
        return y

    '''
    def demod_fsk_online(data, signal, f_base, f_div, chip_l, Fs):
        f1 = f_base + f_div
        f2 = f_base - f_div

        Ts = 1/Fs
        symbol_duration = np.arange(0, chip_l*Ts, Ts)

        y1 = np.exp(1j*2*np.pi*f1*symbol_duration)
        y2 = np.exp(1j*2*np.pi*f2*symbol_duration)

        corrs = []
        curr_data = data[:len(signal)*chip_l]

        print(len(data) - len(signal)*chip_l)
        for i in range(len(data) - len(signal)*chip_l):
            print(i)
            curr_data = data[i:len(signal)*chip_l + i]
            print(datetime.datetime.now())
            x1 = scipy.signal.correlate(curr_data, y1, mode="same")
            print(datetime.datetime.now())
            x2 = scipy.signal.correlate(curr_data, y2, mode="same")
            print(datetime.datetime.now())

            d_fsk = (np.abs(x1) - np.abs(x2))/(np.abs(x1) + np.abs(x2))
            d_fsk = d_fsk[::chip_l]

            print(datetime.datetime.now())
            corr = scipy.signal.correlate(signal, d_fsk)
            print(datetime.datetime.now())
            corrs.append(abs(corr))

        return corrs
    '''

    def demod_fsk(self, signal):
        x1 = scipy.signal.correlate(signal, self.f1, mode="same")
        x2 = scipy.signal.correlate(signal, self.f2, mode="same")

        d_fsk = (np.abs(x1) - np.abs(x2))/(np.abs(x1) + np.abs(x2))
        return d_fsk
