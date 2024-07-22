from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def read_img(img_path):
    img = Image.open(img_path)
    return img

def show_img(img):
    img.show()

def save_img(img, img_path):
    img.save(img_path)

def rename_img_path(img_path, feature):
    img_path = img_path.replace('/', '\\')
    img_root = img_path.split('\\')[-1]
    img_name = img_root.split('.')[0]
    img_format = img_root.split('.')[-1]
    img_new_name = img_name + "_" + feature + '.' + img_format; 
    slash_idx = img_path.rfind('\\')
    img_new_path = img_path[:slash_idx+1] + img_new_name
    return img_new_path

def change_brightness(img, img_path, brightness_factor=30):
    print("Thay đổi độ sáng . . .")
    img_array = np.array(img)
    brightened_image = np.clip(img_array + brightness_factor, 0, 255)
    brightened_image = Image.fromarray(brightened_image.astype('uint8'))
    save_img(brightened_image, rename_img_path(img_path, "brightness"))

def change_constrast(img, img_path, constrast_factor=1):
    print("Thay đổi độ tương phản . . .")
    img_array = np.array(img)
    mean = np.mean(img_array, axis=(0, 1), keepdims=True)
    contrasted_image = np.clip((img_array - mean) * constrast_factor + mean, 0, 255)
    contrasted_image = Image.fromarray(contrasted_image.astype('uint8'))
    save_img(contrasted_image, rename_img_path(img_path, "contrast"))

def flip_image(img, img_path, mode):
    print(f"Lật ảnh {mode} . . .")
    mode = 'vertically' if mode == 'dọc' else 'horizontally' if mode == 'ngang' else mode
    img_array = np.array(img)
    if mode == 'vertically':
        flipped_image = np.flipud(img_array)
    elif mode == 'horizontally':
        flipped_image = np.fliplr(img_array)
    flipped_image = Image.fromarray(flipped_image.astype('uint8'))
    save_img(flipped_image, rename_img_path(img_path, f'flip_{mode}'))

def RGB_to_gray(img, img_path):
    print("Chuyển đổi ảnh RGB thành ảnh xám . . .")
    img_array = np.array(img)
    gray_filter = [0.2989, 0.587, 0.114]
    gray_image = np.dot(img_array[...,:3], gray_filter)
    gray_image = Image.fromarray(gray_image.astype('uint8'))
    save_img(gray_image, rename_img_path(img_path, 'filter_gray'))

def RGB_to_sepia(img, img_path):
    print("Chuyển đổi ảnh RGB thành ảnh sepia . . .")
    img_array = np.array(img)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    print(sepia_filter.T)
    print(sepia_filter)
    sepia_image = np.dot(img_array[...,:3], sepia_filter.T)
    sepia_image = Image.fromarray(sepia_image.astype('uint8'))
    save_img(sepia_image, rename_img_path(img_path, 'filter_sepia'))

def apply_kernel(img_array, kernel):
    # -k + 1 + 2p = 0
    # p = (k - 1) / 2
    kernel_height, kernel_width = kernel.shape
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2
    padded_image = np.pad(img_array, ((padding_height, padding_height), (padding_width, padding_width), (0, 0)), mode='constant')
    result_image = np.zeros_like(img_array)

    for x in range(img_array.shape[0]):
        for y in range(img_array.shape[1]):
            for z in range(img_array.shape[2]):
                region = padded_image[x:x+kernel_height, y:y+kernel_width, z]
                result_image[x][y][z] = np.sum(region*kernel)
    
    result_image = np.clip(result_image, 0, 255)
    return result_image
    

def Blur_image(img, img_path):
    print("Làm mờ ảnh . . .")
    img_array = np.array(img)
    kernel = np.ones((5, 5)) / 25
    blur_image = apply_kernel(img_array, kernel)
    blur_image = Image.fromarray(blur_image.astype('uint8'))
    save_img(blur_image, rename_img_path(img_path, 'blur'))

def Sharpen_image(img, img_path):
    print("Làm sắc nét ảnh . . .")

def Cut_image_in_size(img, img_path):
    print("")

def Cut_image_circle(img, img_path):
    print("Cut image in circle")

def Cut_image_elips(img, img_path):
    print("Cut image in elips")

def ZoomIn2x(img, img_path):
    print("ZoomIn2x")

def ZoomOut2x(img, img_path):
    print("ZoomOut2x")

def main():
    #
    img_path = 'C:\\Users\\nguye\\OneDrive\\Desktop\\LAB02_TUD\\image.jpg'
    img = read_img(img_path)
    # 
    print("0. Thực hiện tất cả.")
    print("1. Thay đổi độ sáng.")
    print("2. Thay đổi độ tương phản.")
    print("3. Lật ảnh (ngang - dọc).")
    print("4. Chuyển đổi ảnh RGB thành ảnh xám/sepia.")
    print("5. Làm mờ/sắc nét ảnh.")
    print("6. Cắt ảnh theo kích thước.")
    print("7. Cắt ảnh theo khung.")
    print("8. Phóng to/Thu nhỏ 2x")
    choose = int(input("Lựa chọn chức năng xử lí ảnh: "))
    # 
    if choose == 0:
        change_brightness(img, img_path)
        change_constrast(img, img_path)
        flip_image(img, img_path, 'ngang')
        flip_image(img, img_path, 'dọc')
        RGB_to_gray(img, img_path)
        RGB_to_sepia(img, img_path)
    elif choose == 1:
        change_brightness(img, img_path)
    elif choose == 2:
        change_constrast(img, img_path)
    elif choose == 3:
        print("0. Lật ngang.")
        print("1. Lật dọc.")
        mode = int(input("Lựa chọn chế độ lật ảnh: "))
        if mode == 0:
            flip_image(img, img_path, 'ngang')
        elif mode == 1:
            flip_image(img, img_path, 'dọc')
        else:
            print("Giá trị không hợp lệ !")
    elif choose == 4:
        print("0. Gray.")
        print("1. Sepia.")
        mode = int(input("Lựa chọn chế độ đổi màu: "))
        if mode == 0:
            RGB_to_gray(img, img_path)
        elif mode == 1:
            RGB_to_sepia(img, img_path)
        else:
            print("Giá trị không hợp lệ !")
    elif choose == 5:
        print("0. Blur.")
        print("1. Sharpen.")
        mode = int(input("Lựa chọn chế độ: "))
        if mode == 0:
            Blur_image(img, img_path)
        elif mode == 1:
            Sharpen_image(img, img_path)
        else:
            print("Giá trị không hợp lệ !")
    elif choose == 6:
        print("6")
    elif choose == 7:
        print("7")
    elif choose == 8:
        print("8")
    else:
        print("Giá trị không hợp lệ !")

if __name__ == '__main__':
    main()
    # blur_kernel = np.ones((5, 5)) / 25
    # print(blur_kernel.shape)


