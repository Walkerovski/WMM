import matplotlib.pyplot as plt
import numpy as np


def scal(cos, *odwolywania):
    for odwolanie in odwolywania:
        cos = odwolanie(cos)
    return cos

def widmo_amplitudowe(sygnal):
    widmo = np.fft.fft(sygnal)
    return np.abs(widmo)

def widmo_fazowe(sygnal):
    widmo = np.fft.fft(sygnal)
    return np.angle(widmo)

def moc_sygnalu(sygnal):
    return sum([i**2 for i in sygnal]) / len(sygnal)

def tw_parsevala(sygnal):
    fft =  np.fft.fft(sygnal)
    parseval = sum([np.abs(n)**2 for n in fft]) / len(sygnal)
    moc = moc_sygnalu(sygnal) * 4
    if parseval == moc:
        return "Prawda", moc
    else:
        return "Fałsz", moc

s1 = np.array([2, 0, 1, 3], dtype="d")
s2 = np.array([1, 0, 3, 0], dtype="d")
N = 4

####################################################
# Zad1A
 
plt.title("Widma sygnału 1")
plt.stem(widmo_amplitudowe(s1), label="widmo amplitudowe", linefmt="c-", markerfmt="co", basefmt="k:")
plt.stem(widmo_fazowe(s1), label="widmo fazowe", linefmt="r-", markerfmt="ro", basefmt="k:")
plt.legend()
plt.show()

plt.title("Widma sygnału 2")
plt.stem(widmo_amplitudowe(s2), label="widmo amplitudowe", linefmt="c-", markerfmt="co", basefmt="k:")
plt.stem(widmo_fazowe(s2), label="widmo fazowe", linefmt="r-", markerfmt="ro", basefmt="k:")
plt.legend()
plt.show()

print(tw_parsevala(s1), tw_parsevala(s2))
 
# Zad1B

def splot_kolowy_reczny(sygnal1, sygnal2):
    splot = [0.0 for i in range(N)]
    for s1 in range(N):
        for s2 in range(N):
            splot[s1] += sygnal1[s2] * sygnal2[s1-s2]
    return splot

def splot_kolowy_dft(sygnal1, sygnal2):
    x = np.fft.fft(sygnal1) * np.fft.fft(sygnal2)
    return np.abs(np.fft.ifft(x))

print(splot_kolowy_reczny(s1, s2))
print(splot_kolowy_dft(s1, s2))

#######################################################
# Zad2

A = 4
N = 52
sygnal_bazowy = A * np.cos(np.arange(0, N, dtype="d") * (2 * np.pi / N))

sygnaly_z_przesunieciem = [
    ("0"   , "c", sygnal_bazowy),
    ("N/4" , "r", np.append(sygnal_bazowy[N//4:], sygnal_bazowy[:N//4])),
    ("N/2" , "k", np.append(sygnal_bazowy[N//2:], sygnal_bazowy[:N//2])),
    ("3N/4", "g", np.append(sygnal_bazowy[3*N//4:], sygnal_bazowy[:3*N//4])),
]

plt.title("Widma amplitudowe")
for przesuniecie, kolor, sygnal in sygnaly_z_przesunieciem:
    plt.plot(scal(sygnal, np.fft.fft, np.abs), kolor, label=f"Sygnał przesunięty o {przesuniecie}")
plt.legend()
plt.show()

plt.title("Widma fazowe")
for przesuniecie, kolor, sygnal in sygnaly_z_przesunieciem:
    sygnal_fft = np.fft.fft(sygnal)
    sygnal_fft = np.ma.masked_where(np.abs(sygnal_fft) < 1e-6, sygnal_fft)
    sygnal_fft = np.ma.angle(sygnal_fft)
    sygnal_fft = np.ma.filled(sygnal_fft, 0.0)
    plt.plot(sygnal_fft, kolor, label=f"Sygnał przesunięty o {przesuniecie}")
plt.legend()
plt.show()
#####################################################################################
# Zad3

A = 3
N = 11
sygnal_bazowy = A * (1 - N * np.arange(N))
wypelnienie_zerami = [(0, "c", "0"), (N, "r", "1N"), (4*N, "k", "4N"), (9*N, "g", "9N")]

plt.title("Widma amplitudowe")
for dopelnienie, kolor, ilosc_zer in wypelnienie_zerami:
    plt.plot(
        scal(
            sygnal_bazowy,
            lambda i: np.fft.fft(i, i.size + dopelnienie),
            np.abs,
        ),
        kolor,
        label=f"Sygnał dopełniony {ilosc_zer} zerami",
    )
plt.legend()
plt.show()

plt.title("Widma fazowe")
for dopelnienie, kolor, ilosc_zer in wypelnienie_zerami:
    plt.scatter(
        np.arange(N + dopelnienie),
        scal(
            sygnal_bazowy,
            lambda i: np.fft.fft(i, i.size + dopelnienie),
            np.angle,
        ),
        c=kolor,
        label=f"Sygnał dopełniony {ilosc_zer} zerami",
    )
plt.grid()
plt.legend()
plt.show()
#####################################################################
# Zad4
A1, F1 = 0.1, 3_000
A2, F2 = 0.4, 4_000
A3, F3 = 0.8, 10_000
FS = 48_000

N = 2048
def wartosc_w_czasie(t):
    wartosc = A1 * np.sin(2 * np.pi * F1 * t)
    wartosc += A2 * np.sin(2 * np.pi * F2 * t)
    wartosc += A3 * np.sin(2 * np.pi * F3 * t)
    return wartosc

sygnal = wartosc_w_czasie(np.arange(N) / FS)
moc = scal(sygnal, np.fft.rfft, np.abs, lambda i: i * (2 / N))
plt.title("Widmowa gęstość sygnału dla N = 2048")
plt.stem(np.arange(moc.size) * FS / N, moc, linefmt="r", markerfmt="r")
plt.show()

N = 3072
sygnal = wartosc_w_czasie(np.arange(N) / FS)
moc = scal(sygnal, np.fft.rfft, np.abs, lambda i: i * (2 / N))
plt.title("Widmowa gęstość sygnału dla N = 3072")
plt.stem(np.arange(moc.size) * FS / N, moc, linefmt="r", markerfmt="r")
plt.show()
