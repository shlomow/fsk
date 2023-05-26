import math
import socket
from bitstring import BitArray
import struct
import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

class Fsk:
    def __init__(self, signal, rate, f_div, Fs, f_base=0):
        self.signal = signal
        self.signal_snr = self.signal / np.linalg.norm(self.signal)
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

    def detect_signal(self, data):
        signal = np.repeat(self.signal, self.chip_l)
        offset = int(self.sig_size / 2)
        max_sample = int(self.sig_size*1.5)
        
        for i in range(0, len(data), offset):
            curr_data = data[i:i + max_sample]
            d_fsk = self.demod_fsk(curr_data)
            
            corr = scipy.signal.correlate(signal, d_fsk, 'full')
            peak = np.argmax(corr) - int(len(corr) / 2)
            # print(peak)
            # print(self.sig_size)
            # print(len(d_fsk))
            snr = self.estimate_snr(d_fsk[peak:peak + self.sig_size])

            print(f'snr={snr}, corr={max(abs(corr))}')

    def detect_signal_tcp(self, host, port):
        signal = np.repeat(self.signal, self.chip_l)
        offset = int(self.sig_size / 2)
        max_sample = int(self.sig_size*1.5)

        sock = socket.socket()
        sock.connect((host, port))

        def wait_bytes(sock, count):
            x = b''
            count = count * 8 # 2 floats for each sample
            while len(x) < count:
                x += sock.recv(count - len(x))
            
            return np.frombuffer(x, dtype=np.complex64)
        curr_data = wait_bytes(sock, max_sample)

        while True:
            d_fsk = self.demod_fsk(curr_data)
            corr = scipy.signal.correlate(signal, d_fsk)

            print(max(abs(corr)))

            curr_data = curr_data[offset:]
            curr_data = np.append(curr_data, wait_bytes(sock, offset))

    def demod_fsk_online(self, signal):
        curr_data = signal[:self.sig_size]

        corr_len = len(self.symbol_duration)
        x1 = scipy.signal.correlate(curr_data, self.f1, mode="full")
        x2 = scipy.signal.correlate(curr_data, self.f2, mode="full")

        f1_shift = np.exp(-1j*2*np.pi*self.f_div)
        f2_shift = np.exp(1j*2*np.pi*self.f_div)

        f1_shift2 = np.exp(-1j*2*np.pi*corr_len*self.f_div)
        f2_shift2 = np.exp(1j*2*np.pi*corr_len*self.f_div)

        for i in range(len(signal) - self.sig_size):
            # print(i)

            curr_x1 = (x1[-1] - signal[i]*f1_shift)*f2_shift + signal[i + corr_len]*f1_shift2
            # x1 = np.append(x1, curr_x1)

            curr_x2 = (x2[-1] - signal[i]*f2_shift)*f1_shift + signal[i + corr_len]*f2_shift2
            # x2 = np.append(x2, curr_x2)
      
        d_fsk = (np.abs(x1) - np.abs(x2))/(np.abs(x1) + np.abs(x2))
        return d_fsk

    def estimate_snr(self, d_fsk):
        d_fsk = d_fsk[::self.chip_l]
        if len(d_fsk) < len(self.signal_snr):
            return -100
        signal_2 = np.dot(d_fsk, self.signal_snr)**2
        noise_2 = np.linalg.norm(d_fsk)**2 - signal_2
        snr = signal_2 / noise_2
        snr = 10*np.log10(snr/self.chip_l)
        return snr

    def demod_fsk(self, signal):
        x1 = scipy.signal.correlate(signal, self.f1, mode="valid")
        x2 = scipy.signal.correlate(signal, self.f2, mode="valid")

        d_fsk = (np.abs(x1) - np.abs(x2))/(np.abs(x1) + np.abs(x2))
        return d_fsk

    def awgn(self, signal, snr):
        sig_energy = np.linalg.norm(signal) ** 2
        noise_energy = sig_energy / (10**(snr/10))
        noise_variance = noise_energy / (len(signal) - 1)
        noise_std = np.sqrt(noise_variance)
        noise = noise_std * np.random.normal(size=len(signal))
        return signal + noise
