import numpy as np
import fsk

with open('signal.dat', 'rb') as f:
    sig = f.read()

sig = list(sig)

for i in range(len(sig)):
    sig[i] = int(sig[i])
    if sig[i] == 0:
        sig[i] = -1

x = fsk.Fsk(sig, 1953.125, 3e3, 1e6)

capture = np.fromfile(open('matlab.dat', 'rb'), dtype=np.complex64)

v = x.demod_fsk(capture)

print(v[:10])

print(x.estimate_snr(v))
