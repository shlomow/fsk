from bitstring import BitArray
import scipy.signal
import numpy as np
import click
import fsk
import matplotlib.pyplot as plt

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
    plt.subplot(411)
    plt.plot(sim.dt, mod_sig)

    sig_tx = np.repeat(signal, sim.chip_l)
    plt.subplot(412)
    plt.plot(sim.dt, sig_tx)

    y = sim.demod_fsk(mod_sig)
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
    if format == 'complex':
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
    d_fsk = sim.demod_fsk(capture)

    signal = np.repeat(signal, sim.chip_l)

    corr = scipy.signal.correlate(signal, d_fsk)

    plt.plot(abs(corr))
    plt.show()

cli()