from bitstring import BitArray
import scipy.signal
import numpy as np
import click
import fsk
import matplotlib.pyplot as plt
import datetime as dt

# python3 -m fsk play -f data/rtl/test.bin  --data data/rtl/signal.hex -r 625 -S 250e3 -d 5e3
# python3 -m fsk play -f data/usrp/usrp_samples.dat  --data data/usrp/signal.hex -r 1e3 -S 1e6 -d 70e3 --format short

@click.group()
def cli():
    pass

@cli.command()
@click.option('-s', '--size', default=5000)
@click.option('-r', '--rate', default=1e3)
@click.option('-S', '--sample-rate', default=1e6)
@click.option('-d', '--deviation', default=500)
def simulate(size, rate, sample_rate, deviation):
    signal = np.random.choice([1, -1], size=size)

    sim = fsk.Fsk(signal, rate, deviation, sample_rate)

    mod_sig = sim.calc_fsk()
    mod_sig = sim.awgn(mod_sig, -18)
    plt.subplot(411)
    plt.plot(sim.dt, mod_sig)

    sig_tx = np.repeat(signal, sim.chip_l)
    plt.subplot(412)
    plt.plot(sim.dt, sig_tx)

    y = sim.demod_fsk(mod_sig)
    print(sim.estimate_snr(y))
    plt.subplot(413)
    plt.plot(sim.dt, y)

    y = y[::sim.chip_l]
    t = scipy.signal.correlate(y, signal)
    plt.subplot(414)
    plt.plot(t)
    plt.show()


@cli.command()
@click.option('-f', '--capture', required=True)
@click.option('--format', default='float')
@click.option('--data', required=True)
@click.option('-r', '--rate', default=1e3)
@click.option('-S', '--sample-rate', default=1e6)
@click.option('-d', '--deviation', default=1e3)
def play(capture, format, data, rate, sample_rate, deviation):
    if format == 'float':
        capture = np.fromfile(open(capture, 'rb'), dtype=np.complex64)
    elif format == 'short':
        capture = np.fromfile(open(capture, 'rb'), dtype=np.int16)
        i = np.array(capture[::2])
        q = np.array(capture[1::2])
        capture = i + 1j*q
    
    with open(data) as f:
        signal = BitArray(hex=f.read())
        for i in range(len(signal)):
            if signal[i] == '0':
                signal[i] = -1
            else:
                signal[i] = 1

    print(len(capture))
    sim = fsk.Fsk(signal, rate, deviation, sample_rate)
    print(dt.datetime.now())
    d_fsk = sim.demod_fsk(capture)
    print(dt.datetime.now())
    signal = np.repeat(signal, sim.chip_l)

    corr = scipy.signal.correlate(signal, d_fsk)

    plt.plot(abs(corr))
    plt.show()

@cli.command()
@click.option('-f', '--capture', required=True)
@click.option('--format', default='float')
@click.option('--data', required=True)
@click.option('-r', '--rate', default=1e3)
@click.option('-S', '--sample-rate', default=1e6)
@click.option('-d', '--deviation', default=1e3)
def detect(capture, format, data, rate, sample_rate, deviation):
    if format == 'float':
        capture = np.fromfile(open(capture, 'rb'), dtype=np.complex64)
    elif format == 'short':
        capture = np.fromfile(open(capture, 'rb'), dtype=np.int16)
        i = np.array(capture[::2])
        q = np.array(capture[1::2])
        capture = i + 1j*q
    
    with open(data) as f:
        signal = BitArray(hex=f.read())
        for i in range(len(signal)):
            if signal[i] == '0':
                signal[i] = -1
            else:
                signal[i] = 1

    sim = fsk.Fsk(signal, rate, deviation, sample_rate)
    print(dt.datetime.now())
    d_fsk = sim.detect_signal(capture)
    print(dt.datetime.now())

@cli.command()
@click.option('--data', required=True)
@click.option('-r', '--rate', default=1e3)
@click.option('-S', '--sample-rate', default=1e6)
@click.option('-d', '--deviation', default=1e3)
def server(data, rate, sample_rate, deviation):
    with open(data) as f:
        signal = BitArray(hex=f.read())
        for i in range(len(signal)):
            if signal[i] == '0':
                signal[i] = -1
            else:
                signal[i] = 1

    sim = fsk.Fsk(signal, rate, deviation, sample_rate)
    sim.detect_signal_tcp('localhost', 8888)

cli()
