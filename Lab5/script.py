import os
import sys
import time

import cv2
from cv2 import calcHist
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col

color_img = cv2.imread("./images/house1_col.png", cv2.IMREAD_UNCHANGED)
color_path = "./images/house1_col.png"
mono_img = cv2.imread("./images/house1_mono.png", cv2.IMREAD_UNCHANGED)
mono_path = "./images/house1_mono.png"


def bitratecalculate(image_path):
    bitrate = 8*os.stat(image_path).st_size/(mono_img.shape[0]*mono_img.shape[1])
    print(f"Bitrate: {bitrate:.4f}")
bitratecalculate(mono_path)


def enthropycalculate(hist):
    pdf = hist/hist.sum() ### normalizacja histogramu -> rozkład prawdopodobieństwa; UWAGA: niebezpieczeństwo '/0' dla 'zerowego' histogramu!!!
    # entropy = -(pdf*np.log2(pdf)).sum() ### zapis na tablicach, ale problem z '/0'
    entropy = -sum([x*np.log2(x) for x in pdf if x != 0])
    return entropy
hist_mono = cv2.calcHist([mono_img], [0], None, [256], [0, 256]).flatten()
entropy_mono = enthropycalculate(hist_mono)
print(f"Entropy: {entropy_mono:.4f}") 

def printi(img, img_title="image"):
    """ Pomocnicza funkcja do wypisania informacji o obrazie. """
    print(f"{img_title}, wymiary: {img.shape}, typ danych: {img.dtype}, wartości: {img.min()} - {img.max()}")


def cv_imshow(img, img_title="image"):
    """
    Funkcja do wyświetlania obrazu w wykorzystaniem okna OpenCV.
    Wykonywane jest przeskalowanie obrazu z rzeczywistymi lub 16-bitowymi całkowitoliczbowymi wartościami pikseli,
    żeby jedną funkcją wywietlać obrazy różnych typów.
    """
    # cv2.namedWindow(img_title, cv2.WINDOW_AUTOSIZE) # cv2.WINDOW_NORMAL

    if (img.dtype == np.float32) or (img.dtype == np.float64):
        img_ = img / 255
    elif img.dtype == np.int16:
        img_ = img*128
    else:
        img_ = img
    cv2.imshow(img_title, img_)
    cv2.waitKey(1)  ### oczekiwanie przez bardzo krótki czas - okno się wyświetli, ale program się nie zablokuje, tylko będzie kontynuowany

def hdiff():
    img_tmp1 = mono_img[:, 1:] 
    img_tmp2 = mono_img[:, :-1] 
    image_hdiff = cv2.addWeighted(img_tmp1, 1, img_tmp2, -1, 0, dtype=cv2.CV_16S)
    image_hdiff_0 = cv2.addWeighted(mono_img[:, 0], 1, 0, 0, -127, dtype=cv2.CV_16S) ### od 'zerowej' kolumny obrazu oryginalnego odejmowana stała wartość '127'
    image_hdiff = np.hstack((image_hdiff_0, image_hdiff)) ### połączenie tablic w kierunku poziomym, czyli 'kolumna za kolumną'
    printi(image_hdiff, "image_hdiff") 
    cv_imshow(image_hdiff, "image_hdiff")
    image_hdiff = cv2.calcHist([(image_hdiff + 255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
    plt.plot(np.arange(0, 256), hist_mono, color="blue", label="obrazu wejściowy")
    plt.plot(np.arange(-255, 256), image_hdiff, color="red", label="obraz różnicowy")
    plt.legend()
    plt.gcf().set_dpi(150)
    plt.show()
    entropy_hdiff = enthropycalculate(image_hdiff)
    print(f"Entropy: mono {entropy_mono:.4f} Entropy: hdiff {entropy_hdiff:.4f}")
hdiff()

def dwt(img):
    """
    Bardzo prosta i podstawowa implementacja, nie uwzględniająca efektywnych metod obliczania DWT
    i dopuszczająca pewne niedokładności.
    """
    maskL = np.array([0.02674875741080976, -0.01686411844287795, -0.07822326652898785, 0.2668641184428723,
        0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287795, 0.02674875741080976])
    maskH = np.array([0.09127176311424948, -0.05754352622849957, -0.5912717631142470, 1.115087052456994,
        -0.5912717631142470, -0.05754352622849957, 0.09127176311424948])

    ll = cv2.sepFilter2D(img,         -1, maskL, maskL)[::2, ::2]
    lh = cv2.sepFilter2D(img, cv2.CV_16S, maskL, maskH)[::2, ::2] ### ze względu na filtrację górnoprzepustową -> wartości ujemne, dlatego wynik 16-bitowy ze znakiem
    hl = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskL)[::2, ::2]
    hh = cv2.sepFilter2D(img, cv2.CV_16S, maskH, maskH)[::2, ::2]
    
    cv_imshow(ll, 'LL2')
    cv_imshow(cv2.multiply(hh, 2), 'HH2')
    cv_imshow(cv2.multiply(lh, 2), 'LH2')
    cv_imshow(cv2.multiply(hl, 2), 'HL2')

    hist_ll = cv2.calcHist([ll], [0], None, [256], [0, 256]).flatten()
    hist_lh = cv2.calcHist([(lh+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten() ### zmiana zakresu wartości i typu danych ze względu na cv2.calcHist() (jak wcześniej przy obrazach różnicowych)
    hist_hl = cv2.calcHist([(hl+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
    hist_hh = cv2.calcHist([(hh+255).astype(np.uint16)], [0], None, [511], [0, 511]).flatten()
    H_ll = enthropycalculate(hist_ll)
    H_lh = enthropycalculate(hist_lh)
    H_hl = enthropycalculate(hist_hl)
    H_hh = enthropycalculate(hist_hh)
    print(f"H(LL) = {H_ll:.4f} \nH(LH) = {H_lh:.4f} \nH(HL) = {H_hl:.4f} \nH(HH) = {H_hh:.4f} \nH_śr = {(H_ll+H_lh+H_hl+H_hh)/4:.4f}")

    fig = plt.figure()
    fig.set_figheight(fig.get_figheight()*2) ### zwiększenie rozmiarów okna
    fig.set_figwidth(fig.get_figwidth()*2)
    plt.subplot(2, 2, 1)
    plt.plot(hist_ll, color="blue")
    plt.title("hist_ll")
    plt.xlim([0, 255])
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(-255, 256, 1), hist_lh, color="red")
    plt.title("hist_lh")
    plt.xlim([-255, 255])
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(-255, 256, 1), hist_hl, color="red")
    plt.title("hist_hl")
    plt.xlim([-255, 255])
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(-255, 256, 1), hist_hh, color="red")
    plt.title("hist_hh")
    plt.xlim([-255, 255])
    plt.show()

dwt(mono_img)


def entrophycol(img):
    printi(img, "image_col")

    image_R = img[:, :, 2] ### cv2.imread() zwraca obrazy w formacie BGR
    image_G = img[:, :, 1]
    image_B = img[:, :, 0]

    hist_R = cv2.calcHist([image_R], [0], None, [256], [0, 256]).flatten()
    hist_G = cv2.calcHist([image_G], [0], None, [256], [0, 256]).flatten()
    hist_B = cv2.calcHist([image_B], [0], None, [256], [0, 256]).flatten()

    H_R = enthropycalculate(hist_R)
    H_G = enthropycalculate(hist_G)
    H_B = enthropycalculate(hist_B)
    print(f"H(R) = {H_R:.4f} \nH(G) = {H_G:.4f} \nH(B) = {H_B:.4f} \nH_śr = {(H_R+H_G+H_B)/3:.4f}")

    cv2.imwrite(f'./images/image_R.png', image_R)
    cv2.imwrite(f'./images/image_G.png', image_G)
    cv2.imwrite(f'./images/image_B.png', image_B)
    
    plt.figure()
    plt.plot(hist_R, color="red")
    plt.plot(hist_G, color="green")
    plt.plot(hist_B, color="blue")
    plt.title("hist RGB")
    plt.xlim([0, 254])
    plt.gcf().set_dpi(200)
    plt.show()

def BGR2YUV(img):
    image_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    hist_Y = calcHist(image_yuv[...,0], [0], None, [256], [0, 256]).flatten()
    hist_U = calcHist(image_yuv[...,1], [0], None, [256], [0, 256]).flatten()
    hist_V = calcHist(image_yuv[...,2], [0], None, [256], [0, 256]).flatten()

    H_Y = enthropycalculate(hist_Y)
    H_U = enthropycalculate(hist_U)
    H_V = enthropycalculate(hist_V)
    print(f"H_Y = {H_Y:.4f} \nH_U = {H_U:.4f} \nH_V = {H_V:.4f} \nH_śr = {(H_Y+H_U+H_V)/3:.4f}")
    cv2.imwrite(f'./images/image_Y.png', image_yuv[...,0])
    cv2.imwrite(f'./images/image_U.png', image_yuv[...,1])
    cv2.imwrite(f'./images/image_V.png', image_yuv[...,2])
    plt.plot(hist_Y, color="yellow", label="Y")
    plt.plot(hist_U, color="cyan", label="U")
    plt.plot(hist_V, color="violet", label="V")
    plt.title("Histogram obrazu YUV")
    plt.xlim([0, 255])
    plt.legend()
    plt.gcf().set_dpi(200)
    plt.show()


entrophycol(color_img)
BGR2YUV(color_img)

def calc_mse_psnr(img1, img2):
    """ Funkcja obliczająca MSE i PSNR dla różnicy podanych obrazów, zakładana wartość pikseli z przedziału [0, 255]. """

    imax = 255.**2 ### maksymalna wartość sygnału -> 255
    """
    W różnicy obrazów istotne są wartości ujemne, dlatego img1 konwertowany jest do typu np.float64 (liczby rzeczywiste) 
    aby nie ograniczać wyniku do przedziału [0, 255].
    """
    mse = ((img1.astype(np.float64)-img2)**2).sum()/img1.size ###img1.size - liczba elementów w img1, ==img1.shape[0]*img1.shape[1] dla obrazów mono, ==img1.shape[0]*img1.shape[1]*img1.shape[2] dla obrazów barwnych
    psnr = 10.0*np.log10(imax/mse)
    return (mse, psnr)


image = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
xx = [] ### tablica na wartości osi X -> bitrate
ym = [] ### tablica na wartości osi Y dla MSE
yp = [] ### tablica na wartości osi Y dla PSNR
Quality = [x for x in range(5, 86, 5)]

for quality in Quality: ### wartości dla parametru 'quality' należałoby dobrać tak, aby uzyskać 'gładkie' wykresy...
    out_file_name = f"./images/out_image_q{quality:03d}.jpg"
    """ Zapis do pliku w formacie .jpg z ustaloną 'jakością' """
    cv2.imwrite(out_file_name, image, (cv2.IMWRITE_JPEG_QUALITY, quality))
    """ Odczyt skompresowanego obrazu, policzenie bitrate'u i PSNR """
    image_compressed = cv2.imread(out_file_name, cv2.IMREAD_UNCHANGED)
    bitrate = 8*os.stat(out_file_name).st_size/(image.shape[0]*image.shape[1]) ### image.shape == image_compressed.shape
    mse, psnr = calc_mse_psnr(image, image_compressed)
    """ Zapamiętanie wyników do pózniejszego wykorzystania """
    xx.append(bitrate)
    ym.append(mse)
    yp.append(psnr)

""" Narysowanie wykresów """
fig = plt.figure()
fig.set_figwidth(fig.get_figwidth()*2)
plt.suptitle("Charakterystyki R-D")
plt.subplot(1, 3, 1)
plt.plot(xx, ym, "-.")
plt.title("MSE(R)")
plt.xlabel("bitrate")
plt.ylabel("MSE", labelpad=0)
plt.subplot(1, 3, 2)
plt.plot(xx, yp, "-o")
plt.title("PSNR(R)")
plt.xlabel("bitrate")
plt.ylabel("PSNR [dB]", labelpad=0)
plt.subplot(1, 3, 3)
plt.plot(Quality, xx, "-o")
plt.title("Quality")
plt.xlabel("quality")
plt.ylabel("bitrate", labelpad=0)
plt.show()

png_bitrate = 8 * os.stat(color_path).st_size / color_img.size
print(f"Bitrate of color PNG: {png_bitrate:.4f}") 