import matplotlib.pyplot as plt
import numpy as np
import cv2

unchanged_img = cv2.imread("./images/house1_col.png", cv2.IMREAD_UNCHANGED)
inoise_img = cv2.imread("./images/house1_col_inoise.png", cv2.IMREAD_UNCHANGED)
inoise_img2 = cv2.imread("./images/house1_col_inoise2.png", cv2.IMREAD_UNCHANGED)
noise_img  = cv2.imread("./images/house1_col_noise.png", cv2.IMREAD_UNCHANGED)

def save_to_file(name, image): # save a picture to a file
    cv2.imwrite("./"+name, image)

def plt_show_img(image, img_title): # print a picture
    plt.figure()
    plt.title(img_title)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert order of the channels from BGR to RGB
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.xticks([]), plt.yticks([])
    plt.show()

def calcPSNR(image1, image2): # calculate PSNR
    imax = 255.**2
    mse = ((image1.astype(np.float64)-image2)**2).sum()/image1.size
    return 10.0*np.log10(imax/mse)

def draw_histogram(image, img_title): #print the histogram of the provided picture
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.flatten()
    plt.figure()
    plt.title(img_title)
    plt.plot(histogram)
    plt.xlim([1, 256])
    plt.ylim([0, 7000])
    plt.show()
 
def zad_1_gauss(image, title):
    for i in [3, 5, 7]: # do for masks 3x3, 5x5 and 7x7
        print(f'Maska: {i} x {i}')
        gauss_blur = cv2.GaussianBlur(image, (i, i), 0)
        print(calcPSNR(unchanged_img, gauss_blur))
        plt_show_img(gauss_blur, f'{title} gauss_blur {i} x {i}')
        #save_to_file(f'./zad1/{title} gauss_blur {i} x {i}.png', gauss_blur)

def zad_1_median(image, title):
    for i in [3, 5, 7]: # do for masks 3x3, 5x5 and 7x7
        print(f'Maska: {i} x {i}')
        median_blur = cv2.medianBlur(image, i)
        print(calcPSNR(unchanged_img, median_blur))
        plt_show_img(median_blur, f'median_blur {i} x {i}')
        #save_to_file(f'./zad1/{title} median_blur {i} x {i}.png', median_blur)

def zad_2(image):
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image_YCrCb[:, :, 0] = cv2.equalizeHist(image_YCrCb[:, :, 0])
    image_end = cv2.cvtColor(image_YCrCb, cv2.COLOR_YCrCb2BGR)
    plt_show_img(image, "Before")
    plt_show_img(image_end, "After")
    draw_histogram(image, "Provided histogram")
    draw_histogram(image_end, "Modified histogram")
    #save_to_file("./zad2/norlmal.png", unchanged_img)
    #save_to_file("./zad2/new_hist.png", image_end)

def zad_3(image):
    for x in [-10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10]:
        gauss_image = cv2.GaussianBlur(image, (3,3), 0)
        laplacian_image = cv2.Laplacian(gauss_image, cv2.CV_64F)
        img = np.asarray(image, np.float64)
        img_out = cv2.addWeighted(img, 1, laplacian_image, x, 0)
        cv2.imwrite(f'./zad3/laplacian-{x}.png', img_out)


#print(calcPSNR(unchanged_img, inoise_img))
#print(calcPSNR(unchanged_img, inoise_img2))
#print(calcPSNR(unchanged_img, noise_img))

#zad_1_gauss(inoise_img, "inoise")
#zad_1_gauss(inoise_img2, "inoise2")
#zad_1_gauss(noise_img, "noise")
#
#zad_1_median(inoise_img, "inoise")
#zad_1_median(inoise_img2, "inoise2")
#zad_1_median(noise_img, "noise")
#
zad_2(unchanged_img)
#
#zad_3(unchanged_img)