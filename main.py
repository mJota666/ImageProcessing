from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
    kernel_height, kernel_width = kernel.shape
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2
    padded_image = np.pad(img_array, ((padding_height, padding_height), (padding_width, padding_width), (0, 0)), mode='edge')
    result_image = np.zeros_like(img_array)

    for x in range(img_array.shape[0]):
        for y in range(img_array.shape[1]):
            for z in range(img_array.shape[2]):
                region = padded_image[x:x+kernel_height, y:y+kernel_width, z]
                result_image[x, y, z] = np.sum(region * kernel)
    
    result_image = np.clip(result_image, 0, 255)
    return result_image

def blur_image(img, img_path):
    print("Làm mờ ảnh . . .")
    img_array = np.array(img)
    kernel = np.ones((3, 3)) / 9
    blur_image = apply_kernel(img_array, kernel)
    blur_image = Image.fromarray(blur_image.astype('uint8'))
    save_img(blur_image, rename_img_path(img_path, 'blur'))

def sharpen_image(img, img_path):
    # img = cv2.imread('C:/Users/nguye/OneDrive/Desktop/LAB02_TUD/alo.png')
    # kernel = np.array([[0, -1, 0],
    #                 [-1, 5, -1],
    #                 [0, -1, 0]])
    # result_image = cv2.filter2D(img, -1, kernel)
    # cv2.imwrite('result_image.jpg', result_image)     
    print("Làm sắc nét ảnh . . .")
    img_array = np.array(img)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen_image = apply_kernel(img_array, kernel)
    sharpen_image = Image.fromarray(sharpen_image.astype('uint8'))
    save_img(sharpen_image, rename_img_path(img_path, 'sharpen'))

def crop_center(img, img_path):
    print("Cắt ảnh trung tâm . . .")
    img_array = np.array(img)
    crop_width = img_array.shape[1] // 2
    crop_height = img_array.shape[0] // 2
    center_x = img_array.shape[1] // 2
    center_y = img_array.shape[0] // 2
    start_x = center_x - crop_width // 2
    start_y = center_y - crop_height // 2
    cropped_image = img_array[start_x:start_x + crop_width, start_y:start_y + crop_height]
    cropped_image = Image.fromarray(cropped_image.astype('uint8'))
    save_img(cropped_image, rename_img_path(img_path, 'cropped_center'))

def crop_circle(img, img_path):
    print("Cắt ảnh theo khung tròn . . .")
    img_array = np.array(img)
    img_width = img_array.shape[1]
    img_height = img_array.shape[0]
    center_x = img_width // 2
    center_y = img_height // 2
    radius = min(center_x, center_y)
    Y, X = np.ogrid[0:img_width, 0:img_height]
    distance_to_center = np.sqrt((X-center_x)**2 + (Y-center_y)**2)
    mask = distance_to_center <= radius
    cropped_circle_image = np.zeros_like(img_array)
    cropped_circle_image[mask] = img_array[mask]
    cropped_circle_image = Image.fromarray(cropped_circle_image.astype('uint8'))
    save_img(cropped_circle_image, rename_img_path(img_path, 'cropped_circle')) 

def create_ellipse_mask(X, Y, img_width, img_height, theta):
    a = np.sqrt((img_width / 2)**2 + (img_height / 2)**2) / 2
    b = (np.sqrt((img_width / 2)**2 + (img_height / 2)**2) - img_width / 2) / 2 + img_width / 2

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    term1 = ((cos_theta**2 / a**2) + (sin_theta**2 / b**2)) * X**2
    term2 = 2 * cos_theta * sin_theta * (1/a**2 - 1/b**2) * X * Y
    term3 = ((sin_theta**2 / a**2) + (cos_theta**2 / b**2)) * Y**2

    mask = term1 + term2 + term3 <= 1

    return mask

def crop_ellipse(img, img_path):
    print("Cắt ảnh theo khung ellipse . . .")
    img_array = np.array(img)
    img_width = img_array.shape[1]
    img_height = img_array.shape[0]
    center_x = img_width // 2
    center_y = img_height // 2
    Y, X = np.ogrid[0:img_height, 0:img_width]
    Y = Y - center_y
    X = X - center_x

    mask1 = create_ellipse_mask(X, Y, img_width, img_height, np.pi / 4)
    mask2 = create_ellipse_mask(X, Y, img_width, img_height, -np.pi / 4)

    cropped_ellipe_image = np.zeros_like(img_array)
    cropped_ellipe_image[mask1] = img_array[mask1]
    cropped_ellipe_image[mask2] = img_array[mask2]
    cropped_ellipe_image = Image.fromarray(cropped_ellipe_image.astype('uint8'))
    save_img(cropped_ellipe_image, rename_img_path(img_path, 'cropped_ellipse'))

def nearest_neighbor_interpolate(img_array, i, j):
    i = round(i)
    j = round(j)

    i = min(max(i, 0), img_array.shape[0] - 1)
    j = min(max(j, 0), img_array.shape[1] - 1)

    return img_array[i, j]

def scale_image(img, img_path, scale):
    if scale >= 1:
        mode = 'zoom_in_2x'
        print("Phóng to 2 lần . . .")
    else:
        mode = 'zoom_out_2x'
        print("Thu nhỏ 2 lần . . .")
    img_array = np.array(img)
    img_height, img_width, img_channels = img_array.shape
    new_height = int(img_height * scale)
    new_width = int(img_width * scale)
    scaled_image = np.zeros((new_height, new_width, img_channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            i = y / scale
            j = x / scale
            for c in range(img_channels):
                scaled_image[y, x, c] = nearest_neighbor_interpolate(img_array[..., c], i, j)

    scaled_image = Image.fromarray(scaled_image.astype('uint8'))
    save_img(scaled_image, rename_img_path(img_path, mode))

def main():
    #
    img_path = 'C:\\Users\\nguye\\OneDrive\\Desktop\\LAB02_TUD\\lena.png'
    img = read_img(img_path)
    # 
    print("0. Thực hiện tất cả.")
    print("1. Thay đổi độ sáng.")
    print("2. Thay đổi độ tương phản.")
    print("3. Lật ảnh (ngang - dọc).")
    print("4. Chuyển đổi ảnh RGB thành ảnh xám/sepia.")
    print("5. Làm mờ/sắc nét ảnh.")
    print("6. Cắt ảnh theo kích thước.")
    print("7. Cắt ảnh theo khung tròn/ellipse.")
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
        RGB_to_gray(img, img_path)
        RGB_to_sepia(img, img_path)
    elif choose == 5:
        blur_image(img, img_path)
        sharpen_image(img, img_path)
    elif choose == 6:
        crop_center(img, img_path)
    elif choose == 7:
        crop_circle(img, img_path)
        crop_ellipse(img, img_path)
    elif choose == 8:
        scale_image(img, img_path, 2)
        scale_image(img, img_path, 1/2)
    else:
        print("Giá trị không hợp lệ !")

if __name__ == '__main__':
    main()