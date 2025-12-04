import glob
from PIL import Image
import imageio
png_filename = r"D:\coding\pycharm_projects\gpu_project\mymodel\github\MMIFAMIN\fusion_out\YUV\GFP_PCI"
def png_to_bmp(png_filename):
    """
    该文件的目的是将一个文件夹中的png图片批量改为bmp格式的图片
    :param png_filename:
    :return:
    """
    #导入图片的路径
    img_path  =png_filename
    print(img_path)
    print(img_path)
    img_path_list = glob.glob(img_path+"\*")
    print("img_path_list")
    print(img_path_list)
    #glob.glob()匹配的结果就是文件的绝对路径名

    for i in range (len(img_path_list)):
        #读取图像
        img1 = Image.open(img_path_list[i])
        #获得Y分量的灰度图像
        gray1 = img1.convert('L')
        # 读取图片Y分类，然后保存Y分类
        print(img_path_list[i].split("\\")[-1].split(".")[0])
        imageio.imwrite(r'D:\coding\pycharm_projects\gpu_project\mymodel\github\MMIFAMIN\fusion_out\Y\GFP_PCI\{}.bmp'.format(img_path_list[i].split("\\")[-1].split(".")[0]),gray1)
png_to_bmp(png_filename)
