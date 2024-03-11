import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
from scipy.fftpack import fft, dct, idct
import math


quality_factor = 100

quantization_matrix_Y = np.array(
    [[16, 11, 10, 16, 24, 40, 41, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])

quantization_matrix_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                         [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
                         [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]])

matriz_conversao = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
matriz_conversao_inversa = np.linalg.inv(matriz_conversao)

def clamping(matriz, maximo, minimo):
    matriz[matriz>maximo] = maximo
    matriz[matriz<minimo] = minimo
    return matriz.astype('uint8')



def chama_dct(canal):
    canal_dct = dct(dct(canal, norm="ortho").T, norm="ortho").T
    # dct_logado = np.log10(np.absolute(luma_dct) + 0.0001)
    # plt.imshow(dct_logado, color_map_fixo("gray", (1,1,1)))
    # plt.show()
    return canal_dct


def chama_idct(canal_dct):
    canal_idct = idct(idct(canal_dct, norm="ortho").T, norm="ortho").T
    # plt.figure("lumaa")
    # plt.axis('off')
    # plt.imshow(luma_idct, color_map_fixo("gray", (1,1,1)))
    # plt.show()
    return canal_idct


def chama_dct_blocos(n, canal):
    size = canal.shape
    output = np.zeros(canal.shape)
    for lin in range(0, size[0], n):
        for col in range(0, size[1], n):
            block = canal[lin:lin + n, col:col + n]
            block_dct = chama_dct(block)
            output[lin:lin + n, col:col + n] = block_dct
    return output


def chama_idct_blocos(n, canal):
    size = canal.shape
    output = np.zeros(canal.shape)
    for lin in range(0, size[0], n):
        for col in range(0, size[1], n):
            block = canal[lin:lin + n, col:col + n]
            block_idct = chama_idct(block)
            output[lin:lin + n, col:col + n] = block_idct
    return output


def color_map_fixo2(nome, max_cor, cor_min):
    return clr.LinearSegmentedColormap.from_list(nome, [cor_min, tuple(max_cor)], 256)


def color_map_fixo(nome, max_cor):
    return clr.LinearSegmentedColormap.from_list(nome, [(0, 0, 0), tuple(max_cor)], 256)


def divide_imagem(imagem):
    R = imagem[:, :, 0]
    G = imagem[:, :, 1]
    B = imagem[:, :, 2]
    return R, G, B


def agrega_imagem(dim1, dim2, dim3):
    reconstruida = np.dstack((dim1, dim2, dim3))
    return reconstruida


def YCbCr_to_RGB(imagem):
    Y = imagem[:, :, 0]
    Cb = imagem[:, :, 1]
    Cr = imagem[:, :, 2]
    R = Y * matriz_conversao_inversa[0][0] + matriz_conversao_inversa[0][1] * (Cb - 128) + matriz_conversao_inversa[0][2] * (Cr - 128)
    G = Y * matriz_conversao_inversa[1][0] + matriz_conversao_inversa[1][1] * (Cb - 128) + matriz_conversao_inversa[1][2] * (Cr - 128)
    B = Y * matriz_conversao_inversa[2][0] + matriz_conversao_inversa[2][1] * (Cb - 128) + matriz_conversao_inversa[2][2] * (Cr - 128)
    nova = np.dstack((R, G, B))
    return clamping(nova, 255, 0)
    


def RBG_to_YCbCr(imagem):
    R = imagem[:, :, 0]
    G = imagem[:, :, 1]
    B = imagem[:, :, 2]
    Y = R * matriz_conversao[0][0] + G * matriz_conversao[0][1] + B * matriz_conversao[0][2]
    Cb = matriz_conversao[1][0] * R + (matriz_conversao[1][1]) * G + matriz_conversao[1][2] * B + 128
    Cr = matriz_conversao[2][0] * R + (matriz_conversao[2][1] * G) + (matriz_conversao[2][2] * B) + 128
    nova = np.dstack((Y, Cb, Cr))
    return nova


def remove_padding(image, shape_original):
    outi = image[:shape_original[0], :shape_original[1], :]
    return outi


# padding, primeiro linhas, depois colunas
def padding(imagem):
    value_multiple = 32  # tem de ser divisível por

    # calcular linhas e colunas em falta
    nlinhas, ncolunas = imagem.shape[0], imagem.shape[1]
    resto_linhas = nlinhas % value_multiple
    resto_colunas = ncolunas % value_multiple
    linhas_falta, colunas_falta = 0, 0
    if (resto_linhas != 0):
        linhas_falta = value_multiple - resto_linhas
    if (resto_colunas != 0):
        colunas_falta = value_multiple - (ncolunas % value_multiple)

    output = imagem
    # junção de matrizes
    if (linhas_falta != 0):
        ultima_linha = imagem[nlinhas - 1, :][np.newaxis, :]
        ultimas_linhas = ultima_linha.repeat(linhas_falta, axis=0)
        output = np.vstack((imagem, ultimas_linhas))

    if (colunas_falta != 0):
        ultima_coluna = output[:, ncolunas - 1][:, np.newaxis]
        ultimas_colunas = ultima_coluna.repeat(colunas_falta, axis=1)
        output = np.append(output, ultimas_colunas, axis=1)
    return output, (linhas_falta, colunas_falta)


def subamostragem(imagem, ratio):
    # ratio is 1x3 tuple
    imagem = imagem.astype('float32')
    tamanhos = imagem.shape
    if ratio[2] == 0:
        cb_tam = (tamanhos[1] * (ratio[1] / ratio[0]), tamanhos[0] * (ratio[1] / ratio[0]))
        cr_tam = (tamanhos[1] * (ratio[1] / ratio[0]), tamanhos[0] * (ratio[1] / ratio[0]))
    else:
        cb_tam = (tamanhos[1] * (ratio[1] / ratio[0]), tamanhos[0])
        cr_tam = (tamanhos[1] * (ratio[2] / ratio[0]), tamanhos[0])
    cb_tam = np.round(cb_tam).astype(int)
    cr_tam = np.round(cr_tam).astype(int)
    cb = imagem[:, :, 1]
    cr = imagem[:, :, 2]
    cb_resized = cv2.resize(cb, cb_tam)
    cr_resized = cv2.resize(cr, cr_tam)
    """
    plt.figure("cb")
    plt.axis('off')
    plt.imshow(cb_resized)
    plt.figure("cr")
    plt.axis('off')
    plt.imshow(cr_resized)
    plt.show()
    """
    return imagem[:, :, 0], cb_resized, cr_resized


def up_sampling(components):
    Y, Cb, Cr = components
    dims = (Y.shape[1], Y.shape[0])
    cb_resized = cv2.resize(Cb, dims)
    cr_resized = cv2.resize(Cr, dims)
    return agrega_imagem(Y, cb_resized, cr_resized)


def quantizacao_dpcm(canal_dct, quantization_matrix, quality_factor):
    if quality_factor >= 50:
        scale_factor = ((100 - quality_factor) / 50)
    else:
        scale_factor = 50 / quality_factor
    if scale_factor != 0:
        true_quantization_matrix = np.ceil(scale_factor * quantization_matrix)
        true_quantization_matrix = clamping(true_quantization_matrix, 255, 1)
    else:
        true_quantization_matrix = quantization_matrix
    size = canal_dct.shape
    output = np.zeros(canal_dct.shape)
    valor_anterior = 0
    aux_anterior = 0
    for lin in range(0, size[0], 8):
        for col in range(0, size[1], 8):
            block = canal_dct[lin:lin + 8, col:col + 8]
            if quality_factor != 100:
                quantized_block = np.divide(block, true_quantization_matrix)
                quantized_block = np.round(quantized_block).astype(int)
            else:
                quantized_block = np.round(block).astype(int)
            # EXERCICIO 9 #
            aux_anterior = quantized_block[0][0]
            quantized_block[0][0] = quantized_block[0][0] - valor_anterior
            valor_anterior = aux_anterior
            # # # # # # # #
            output[lin:lin + 8, col:col + 8] = quantized_block
    """        
    plt.figure("Canal Quantizado")
    plt.axis('off')
    plt.imshow(np.log10(np.absolute(output) + 0.0001), color_map_fixo("gray", (1, 1, 1)))
    plt.show()
    """
    return output


def inv_quantizacao_dpcm(canal_dct, quantization_matrix, quality_factor):
    if quality_factor >= 50:
        scale_factor = ((100 - quality_factor) / 50)
    else:
        scale_factor = 50 / quality_factor
    if scale_factor != 0:
        true_quantization_matrix = np.ceil(scale_factor * quantization_matrix)
        true_quantization_matrix = clamping(true_quantization_matrix, 255, 1)
    else:
        true_quantization_matrix = quantization_matrix
    size = canal_dct.shape
    output = np.zeros(canal_dct.shape)
    valor_anterior = 0
    for lin in range(0, size[0], 8):
        for col in range(0, size[1], 8):
            block = canal_dct[lin:lin + 8, col:col + 8]
            # EXERCICIO 9 #
            block[0][0] = block[0][0] + valor_anterior
            valor_anterior = block[0][0]
            # # # # # # # #
            if quality_factor != 100:
                quantized_block = np.multiply(block, true_quantization_matrix)
            else:
                quantized_block = block
            output[lin:lin + 8, col:col + 8] = quantized_block
    output = output.astype(float)
    #plt.figure("Y canal desquantizado")
    #plt.axis('off')
    #plt.imshow(np.log10(np.absolute(output) + 0.0001), color_map_fixo("gray", (1, 1, 1)))
    #plt.show()
    return output


def encoder(img):
    # EXERCICIO 3
    # R, G, B = divide_imagem(img)
    """
    plt.figure("Imagem original")
    plt.axis('off')
    plt.imshow(img)
    #Utilização dos color map RGB
    plt.figure("Imagem R")
    plt.axis('off')
    plt.imshow(R, color_map_fixo("Red",(1,0,0)))
    plt.figure("Imagem G")
    plt.axis('off')
    plt.imshow(G, color_map_fixo("Green",(0,1,0)))
    plt.figure("Imagem B")
    plt.axis('off')
    plt.imshow(B, color_map_fixo("Blue",(0,0,1)))
    """
    # plt.show()
    # EXERCICIO 4
    with_padding, padding_sizes = padding(img)
    """
    plt.figure("Imagem com padding")
    plt.axis('off')
    plt.imshow(imagem_com_padding)
    plt.show()
    """
    # EXERCICIO 5
    image_YCbCr = RBG_to_YCbCr(with_padding)
    # imagemYCbCr = remove_padding(imagemYCbCr, padding_sizes)
    color_map_cinza = color_map_fixo("gray", (1, 1, 1))
    """
    plt.figure("Y")
    plt.axis('off')
    plt.imshow(imagemYCbCr[:,:,0], color_map_cinza)
    plt.figure("Cb")
    plt.axis('off')
    plt.imshow(imagemYCbCr[:,:,1], color_map_cinza)
    plt.figure("Cr")
    plt.axis('off')
    plt.imshow(imagemYCbCr[:,:,2], color_map_cinza)
    plt.figure("YCbCr")
    plt.axis('off')
    plt.imshow(imagemYCbCr)
    plt.show()
    """
    # EXERCICIO 6
    Y, Cb, Cr = subamostragem(image_YCbCr, (4, 2, 2))
    # EXERCICIO 7
    block_size = 8
    Y_dct = chama_dct_blocos(block_size, Y)
    Cb_dct = chama_dct_blocos(block_size, Cb)
    Cr_dct = chama_dct_blocos(block_size, Cr)
    plt.figure("Y ")
    plt.axis('off')
    plt.imshow(np.log10(np.absolute(Y_dct) + 0.0001), color_map_fixo("gray", (1, 1, 1)))
    
    plt.figure("Cb")
    plt.axis('off')
    plt.imshow(np.log10(np.absolute( Cb_dct) + 0.0001), color_map_fixo("gray", (1, 1, 1)))
    
    plt.figure("Cr")
    plt.axis('off')
    plt.imshow(np.log10(np.absolute( Cr_dct) + 0.0001), color_map_fixo("gray", (1, 1, 1)))
    plt.show()
    # EXERCICIO 8/9
    Y_quant = quantizacao_dpcm(Y_dct, quantization_matrix_Y, quality_factor)
    Cb_quant = quantizacao_dpcm(Cb_dct, quantization_matrix_CbCr, quality_factor)
    Cr_quant = quantizacao_dpcm(Cr_dct, quantization_matrix_CbCr, quality_factor)
    # plt.figure("NovaRG")
    # plt.axis('off')
    # plt.show()

    return [Y_quant, Cb_quant, Cr_quant]


def decoder(componentes_ycbcr_quantizadas, original_shape):
    componentes_ycbcr_quantizadas[0] = inv_quantizacao_dpcm(componentes_ycbcr_quantizadas[0], quantization_matrix_Y,
                                                       quality_factor)
    componentes_ycbcr_quantizadas[1] = inv_quantizacao_dpcm(componentes_ycbcr_quantizadas[1], quantization_matrix_CbCr,
                                                       quality_factor)
    componentes_ycbcr_quantizadas[2] = inv_quantizacao_dpcm(componentes_ycbcr_quantizadas[2], quantization_matrix_CbCr,
                                                       quality_factor)
    block_size = 8
    componentes_ycbcr_quantizadas[0] = chama_idct_blocos(block_size, componentes_ycbcr_quantizadas[0])
    componentes_ycbcr_quantizadas[1] = chama_idct_blocos(block_size, componentes_ycbcr_quantizadas[1])
    componentes_ycbcr_quantizadas[2] = chama_idct_blocos(block_size, componentes_ycbcr_quantizadas[2])
    image = up_sampling(componentes_ycbcr_quantizadas)
    new_imageRGB = YCbCr_to_RGB(image)
    no_padding = remove_padding(new_imageRGB, original_shape)
    """
    plt.figure("NovaRGB")
    plt.axis('off')
    plt.imshow(sem_padding)
    """
    # plt.show()
    return no_padding


def calcula_metricas(image_original, image_rec):
    image_original = image_original.astype(float)
    image_rec = image_rec.astype(float)
    YCbCr_original = RBG_to_YCbCr(image_original)   
    YCbCr_reconstruida = RBG_to_YCbCr(image_rec)
    img_diferencas=np.absolute(YCbCr_original[:,:,0]-YCbCr_reconstruida[:,:,0])
    """
    plt.figure("Imagem de diferencas")
    plt.axis('off')
    plt.imshow((img_diferencas).astype('uint8'), color_map_fixo("gray", (1, 1, 1)))
    plt.show()
    """
    mse = np.sum((image_original-image_rec)**2)/(image_original.shape[0]*image_original.shape[1])
    rmse = math.sqrt(mse)
    p = (np.sum((image_original)**2)) / (image_original.shape[0]*image_original.shape[1])
    snr = 10 * math.log10(p/mse)
    psnr = 10 * math.log10((np.max(image_original)**2)/mse)
    print("ErroMax:",np.max(img_diferencas))
    print("ErroAVG:",np.mean(img_diferencas))
    print ("MSE:", mse)
    print ("RMSE:", rmse)
    print ("SNR:", snr)
    print ("PSNR:", psnr)



def main():
    
    print(matriz_conversao_inversa)
    img = plt.imread("peppers.bmp")
    componentes_subamostradas = encoder(img)
    
    imagem_descodificada = decoder(componentes_subamostradas, img.shape)
    """
    plt.figure("img original")
    plt.axis('off')
    plt.imshow(img)
    plt.figure("img reconstruida")
    plt.axis('off')
    plt.imshow(imagem_descodificada)
    plt.show()
    """
    calcula_metricas(img, imagem_descodificada)


if __name__ == "__main__":
    main()
