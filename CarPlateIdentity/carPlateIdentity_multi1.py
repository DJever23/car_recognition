import cv2
import os
import sys
import numpy as np
import tensorflow as tf

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']

def hist_image(img):
    assert img.ndim==2
    hist = [0 for i in range(256)]
    img_h,img_w = img.shape[0],img.shape[1]

    for row in range(img_h):
        for col in range(img_w):
            hist[img[row,col]] += 1
    p = [hist[n]/(img_w*img_h) for n in range(256)]
    p1 = np.cumsum(p)
    for row in range(img_h):
        for col in range(img_w):
            v = img[row,col]
            img[row,col] = p1[v]*255
    return img

def find_board_area(img):
    assert img.ndim==2
    img_h,img_w = img.shape[0],img.shape[1]
    top,bottom,left,right = 0,img_h,0,img_w
    flag = False
    h_proj = [0 for i in range(img_h)]
    v_proj = [0 for i in range(img_w)]

    for row in range(round(img_h*0.5),round(img_h*0.8),3):
        for col in range(img_w):
            if img[row,col]==255:
                h_proj[row] += 1
        if flag==False and h_proj[row]>12:
            flag = True
            top = row
        if flag==True and row>top+8 and h_proj[row]<12:
            bottom = row
            flag = False

    for col in range(round(img_w*0.3),img_w,1):
        for row in range(top,bottom,1):
            if img[row,col]==255:
                v_proj[col] += 1
        if flag==False and (v_proj[col]>10 or v_proj[col]-v_proj[col-1]>5):
            left = col
            break
    return left,top,120,bottom-top-10

def verify_scale(rotate_rect):
   error = 0.4
   aspect = 4#4.7272
   min_area = 10*(10*aspect)#min_area=10*(10*4)=400
   max_area = 150*(150*aspect)#max_area=150*(150*4)=90000
   min_aspect = aspect*(1-error)#min_aspect=4*(1-0.4)=2.4
   max_aspect = aspect*(1+error)#max_aspect=4*(1+0.4)=6.4
   theta = 30

   # 宽或高为0，不满足矩形直接返回False
   if rotate_rect[1][0]==0 or rotate_rect[1][1]==0:    
       return False
   '''
   rotate_rect[0]为外接矩形的中心坐标(x,y);[1][0]为宽，[1][1]为高,[2]为旋转角度.
   旋转角度θ是水平轴（x轴）逆时针旋转，直到碰到矩形的第一条边停住，此时该边与水平轴的夹角。并且这个边的边长是width，另一条边边长是height
   在opencv中，坐标系原点在左上角，相对于x轴，逆时针旋转角度为负，顺时针旋转角度为正。所以，θ∈（-90度，0]
   '''

   r = rotate_rect[1][0]/rotate_rect[1][1]#r=宽除以高
   r = max(r,1/r)
   area = rotate_rect[1][0]*rotate_rect[1][1]#area为实际面积
   if area>min_area and area<max_area and r>min_aspect and r<max_aspect:#如果实际面积大于最小面积且小于最大面积，并且2.4<r<6.4
       # 矩形的倾斜角度不超过theta
       if ((rotate_rect[1][0] < rotate_rect[1][1] and rotate_rect[2] >= -90 and rotate_rect[2] < -(90 - theta)) or#旋转角度在[-90,-60)
               (rotate_rect[1][1] < rotate_rect[1][0] and rotate_rect[2] > -theta and rotate_rect[2] <= 0)):#旋转角度在(-30,0]
           return True
   return False

def img_Transform(car_rect,image):#传入填充掩膜后的最小矩形，原图
    img_h,img_w = image.shape[:2]
    rect_w,rect_h = car_rect[1][0],car_rect[1][1]
    angle = car_rect[2]

    return_flag = False
    if car_rect[2]==0:#旋转角度为0
        return_flag = True
    if car_rect[2]==-90 and rect_w<rect_h:#旋转角度=-90并且矩形的宽<高
        rect_w, rect_h = rect_h, rect_w
        return_flag = True
    if return_flag:
        car_img = image[int(car_rect[0][1]-rect_h/2):int(car_rect[0][1]+rect_h/2),#当车牌旋转角度为0或者90度时，取整个车牌
                  int(car_rect[0][0]-rect_w/2):int(car_rect[0][0]+rect_w/2)]
        return car_img

    car_rect = (car_rect[0],(rect_w,rect_h),angle)
    box = cv2.boxPoints(car_rect)#获取矩形的四个顶点坐标
    print('img_Transform_box',box)
    print('img_Transform_car_rect',car_rect[0])

    heigth_point = right_point = [0,0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]#矩形中心点坐标(x,y)
    for point in box:#找出四个顶点对应的坐标
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if heigth_point[1] < point[1]:
            heigth_point = point
        if right_point[0] < point[0]:
            right_point = point

    if left_point[1] <= right_point[1]:  # 正角度
        new_right_point = [right_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
        M = cv2.getAffineTransform(pts1, pts2)
        print('Mat1',M)
        print('pts1_1',pts1)
        print('pts1_2',pts2)
        '''
        仿射变换，其实是将图形在2D平面内做变换，变换前后图片中原来平行的线仍会保持平行，可以想象是将长方形变换为平行四边形
        M=cv2.getAffineTransform(pos1,pos2),其中两个位置就是变换前后的对应位置关系。输出的就是仿射矩阵M,shape为[2,3]
        cv.getAffineTransform将创建一个2x3矩阵，该矩阵将传递给cv.warpAffine。
        '''
        dst = cv2.warpAffine(image, M, (round(img_w*2), round(img_h*2)))
        '''
        cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
                       dsize为输出图像的大小;
                       flags表示插值方式，默认为 flags=cv2.INTER_LINEAR，表示线性插值，此外还有：cv2.INTER_NEAREST(最近邻插值)、cv2.INTER_AREA(区域插值)、cv2.INTER_CUBIC(三次样条插值)、cv2.INTER_LANCZOS4(Lanczos插值)
                       borderMode - 边界像素模式
                       borderValue - 边界填充值; 默认情况下，它为0
        round() 方法返回浮点数x的四舍五入值。round(x,n) 返回浮点数x的四舍五入的小数点后的n位数值
        '''
        car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]

    elif left_point[1] > right_point[1]:  # 负角度
        new_left_point = [left_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
        print('pts2_1',pts1)
        print('pts2_2',pts2)
        M = cv2.getAffineTransform(pts1, pts2)
        print('Mat2',M)
        dst = cv2.warpAffine(image, M, (round(img_w*2), round(img_h*2)))
        car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]

    return car_img

def pre_process(orig_img):

    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)    #将原图转换为灰度图
    cv2.imwrite('./carIdentityData/opencv_output/gray_img.jpg', gray_img)
    #cv2.imshow('gray_img', gray_img)

    blur_img = cv2.blur(gray_img, (3, 3))   #均值滤波
    cv2.imwrite('./carIdentityData/opencv_output/blur.jpg', blur_img)
    #cv2.imshow('blur', blur_img)

    sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)   #沿x轴求导，找边缘
    sobel_img = cv2.convertScaleAbs(sobel_img)   #转换图片格式
    cv2.imwrite('./carIdentityData/opencv_output/sobel.jpg', sobel_img)
    #cv2.imshow('sobel', sobel_img)

    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)   #将原图的颜色域转到HSV域，色调、饱和度、亮度
    cv2.imwrite('./carIdentityData/opencv_output/hsv_pic.jpg', hsv_img)
    #cv2.imshow('hsv_pic',hsv_img)

    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]   #h，s，v分别为hsv_img的三个通道
    # 黄色色调区间[26，34],蓝色色调区间:[100,124]，饱和度和亮度均需要高于70
    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')  #将blue_img格式转换为浮点型32位
    cv2.imwrite('./carIdentityData/opencv_output/blue&yellow.jpg', blue_img)
    cv2.imshow('blue&yellow',blue_img)

    mix_img = np.multiply(sobel_img, blue_img)    #两个数组或矩阵相乘，对应位置直接相乘
    cv2.imwrite('./carIdentityData/opencv_output/mix.jpg', mix_img)
    #cv2.imshow('mix', mix_img)

    mix_img = mix_img.astype(np.uint8)

    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  
    #ret, binary_img = cv2.threshold(mix_img, 50, 255, cv2.THRESH_BINARY)    
    '''
    使用最大类间方差法将图像二值化，cv2.THRESH_OTSU自适应找出最合适的阈值
    cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst
                  src：表示的是图片源
                  thresh：表示的是阈值（起始值）
                  maxval：表示的是最大值
                  type：表示的是这里划分的时候使用的是什么类型的算法，常用值为0（cv2.THRESH_BINARY）,超过阈值的设置为最大值255，其他设置为0
    返回值：
          ret ：cv2.THRESH_OTSU 求解出的阈值
          binary_img ：二值图像
    '''
    print('ret',ret)
    cv2.imwrite('./carIdentityData/opencv_output/binary.jpg', binary_img)
    #cv2.imshow('binary',binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,5))     #获得结构元素
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)  #闭操作，先膨胀再腐蚀，使图像轮廓更光滑（need more）,将车牌垂直的边缘连成一个整体
    cv2.imwrite('./carIdentityData/opencv_output/close.jpg', close_img)
    #cv2.imshow('close', close_img)

    return close_img

# 给候选车牌区域做漫水填充算法，一方面补全上一步求轮廓可能存在轮廓歪曲的问题，
# 另一方面也可以将非车牌区排除掉
def verify_color(rotate_rect,src_image):
    img_h,img_w = src_image.shape[:2]#shape[0],shape[1]
    mask = np.zeros(shape=[img_h+2,img_w+2],dtype=np.uint8)#整个掩膜区域比原图要大两个像素值，并且像素值全为0，在全黑的掩膜上将种子区域的领域填充成白色，其他仍为黑色
    #cv2.imshow('flood_mask',mask)
    connectivity = 4#0100
    #种子点上下左右4邻域与种子颜色值在[loDiff,upDiff]的被涂成new_value，也可设置8邻域，
    #如果设为4，表示填充算法只考虑当前像素水平方向和垂直方向的相邻点；如果设为 8，除上述相邻点外，还会包含对角线方向的相邻点。
    loDiff,upDiff = 30,30#负差最大值，正差最大值.loDiff表示当前观察像素值与其部件邻域像素值或者待加入该部件的种子像素之间的亮度或颜色之负差（lower brightness/color difference）的最大值。 
    new_value = 255
    flags = connectivity#0100
    print('flags1',flags)
    flags |= cv2.FLOODFILL_FIXED_RANGE  #按位或，FLOODFILL_FIXED_RANGE=2**16=65536.考虑当前像素与种子象素之间的差，不设置的话则和邻域像素比较,运算结果为01 0000 0000 0000 0100，十进制为65540
    print('flags2',flags)
    print('cv2.FLOODFILL_FIXED_RANGE',cv2.FLOODFILL_FIXED_RANGE)
    '''
    flags = flags | cv2.FLOODFILL_FIXED_RANGE
    cv.FLOODFILL_FIXED_RANGE： 指定颜色填充，二进制为 01 0000 0000 0000 0000.填充时的判断标准是：src(seed.x’, seed.y’) - loDiff <= src(x, y) <= src(seed.x’, seed.y’) +upDiff，此范围内被填充指定的颜色
    cv.FLOODFILL_MASK_ONLY:    指定位置填充，二进制为 10 0000 0000 0000 0000
    '''
    flags |= new_value << 8  
    #<<左移动运算符：运算数的各二进位全部左移若干位，由 << 右边的数字指定了移动的位数，高位丢弃，低位补0.  255左移8位是1111111100000000，运算结果为01 1111 1111 0000 0100，十进制为130820
    print('flags3',flags)
    flags |= cv2.FLOODFILL_MASK_ONLY 
    #FLOODFILL_MASK_ONLY=2**17=131072.设置这个标识符则不会去填充改变原始图像，而是去填充掩模图像（mask），运算结果为11 1111 1111 0000 0100，十进制为261892
    print('flags4',flags)
    print('FLOODFILL_MASK_ONLY',cv2.FLOODFILL_MASK_ONLY)
    '''
    相当于flags = 4 | cv2.FLOODFILL_FIXED_RANGE | 255 << 8 | cv2.FLOODFILL_MASK_ONLY
    通俗来讲，就是用4邻域填充，并填充固定像素值范围，填充掩码而不是填充源图像，以及设填充值为255
    标识符的0-7位为connectivity,8-15位为new_value左移8位的值，16-23位为cv2.FLOODFILL_FIXED_RANGE\cv2.FLOODFILL_MASK_ONLY或者0
        1.低8位用于控制算法的连通性，可取4（填充算法只考虑当前享受水平方向和垂直方向）/8（还考虑对角线方向）
        2.高8位可为0/FLOODFILL_FIXED_RANGE（考虑当前像素与种子像素之间的差）/FLOODFILL_MASK_ONLY(不填充改变原始图像，去填充掩模图像)
        3.中间8位制定填充掩码图像的值
    最终得到的flags为11 1111 1111 0000 0100，十进制为261892
    '''
    rand_seed_num = 5000 #生成多个随机种子
    valid_seed_num = 200 #从rand_seed_num中随机挑选valid_seed_num个有效种子
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)
    '''
    cv2.boxPoints根据minAreaRect的返回值rotate_rect计算矩形的四个点
    旋转的边界矩形,这个边界矩形是面积最小的，因为它考虑了对象的旋转。用到的函数为cv2.minAreaRect()。返回的是一个Box2D结构，
        其中包含矩形左上角角点的坐标（x，y）,矩形的宽和高（w，h），以及旋转角度。但是要绘制这个矩形需要矩形的4个角点，可以通过函数cv2.boxPoints()获得。
        返回形式[ [x0,y0], [x1,y1], [x2,y2], [x3,y3] ]
    '''
    box_points_x = [n[0] for n in box_points]#每一个坐标点的x值
    print('box_points_x1',box_points_x)
    box_points_x.sort(reverse=False)#list.sort( key=None, reverse=False)，reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）
    print('box_points_x2',box_points_x)
    adjust_x = int((box_points_x[2]-box_points_x[1])*adjust_param)#=(第三个x-第二个x*0.1)，对角点
    print('adjust_x',adjust_x)
    col_range = [box_points_x[1]+adjust_x,box_points_x[2]-adjust_x]
    print('col_range',col_range)
    box_points_y = [n[1] for n in box_points]#每一个坐标点的y值
    print('box_points_y1',box_points_y)
    box_points_y.sort(reverse=False)
    print('box_points_y2',box_points_y)
    adjust_y = int((box_points_y[2]-box_points_y[1])*adjust_param)
    print('adjust_y',adjust_y)
    row_range = [box_points_y[1]+adjust_y, box_points_y[2]-adjust_y]
    print('row_range',row_range)
    # 如果以上方法种子点在水平或垂直方向可移动的范围很小，则采用旋转矩阵对角线来设置随机种子点
    if (col_range[1]-col_range[0])/(box_points_x[3]-box_points_x[0])<0.4\
        or (row_range[1]-row_range[0])/(box_points_y[3]-box_points_y[0])<0.4:#小于0.4时重新定义
        points_row = []
        points_col = []
        for i in range(2):
            pt1,pt2 = box_points[i],box_points[i+2]#第一个和第三个坐标点，第二个和第四个坐标点，对角线
            x_adjust,y_adjust = int(adjust_param*(abs(pt1[0]-pt2[0]))),int(adjust_param*(abs(pt1[1]-pt2[1])))
            if (pt1[0] <= pt2[0]):
                pt1[0], pt2[0] = pt1[0] + x_adjust, pt2[0] - x_adjust
            else:
                pt1[0], pt2[0] = pt1[0] - x_adjust, pt2[0] + x_adjust
            if (pt1[1] <= pt2[1]):
                pt1[1], pt2[1] = pt1[1] + adjust_y, pt2[1] - adjust_y
            else:
                pt1[1], pt2[1] = pt1[1] - y_adjust, pt2[1] + y_adjust
            temp_list_x = [int(x) for x in np.linspace(pt1[0],pt2[0],int(rand_seed_num /2))]
            temp_list_y = [int(y) for y in np.linspace(pt1[1],pt2[1],int(rand_seed_num /2))]
            points_col.extend(temp_list_x)
            points_row.extend(temp_list_y)
        print('in for')
    else:
        points_row = np.random.randint(row_range[0],row_range[1],size=rand_seed_num)
        '''
        numpy.random.randint(low, high=None, size=None, dtype='l')返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。如果没有写参数high的值，则返回[0,low)的值。
        定义rand_seed_num = 5000。size为输出随机数的尺寸，这里输出5000个随机数
        '''
        points_col = np.linspace(col_range[0],col_range[1],num=rand_seed_num).astype(np.int)
        '''
        np.linspace主要用来创建等差数列。np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None),在start和stop之间返回均匀间隔的数据
                                                     endpoint：True则包含stop；False则不包含stop; retstep如果为True则结果会给出数据间隔
                                         在[col_range[0],col_range[1]]之间输出包含5000个数据的等差数列，并将其修改格式为整型
        '''
        print('in else')

    points_row = np.array(points_row)
    points_col = np.array(points_col)
    hsv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    h,s,v = hsv_img[:,:,0],hsv_img[:,:,1],hsv_img[:,:,2]
    # 将随机生成的多个种子依次做漫水填充,理想情况是整个车牌被填充
    flood_img = src_image.copy()
    seed_cnt = 0
    for i in range(rand_seed_num):
        rand_index = np.random.choice(rand_seed_num,1,replace=False)#从[0,5000)之间随机抽取一个数，且不能重复
        row,col = points_row[rand_index],points_col[rand_index]
        # 限制随机种子必须是车牌背景色，黄色色调区间[26，34],蓝色色调区间:[100,124]
        if (((h[row,col]>26)&(h[row,col]<34))|((h[row,col]>100)&(h[row,col]<124)))&(s[row,col]>70)&(v[row,col]>70):
            cv2.floodFill(src_image, mask, (col,row), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
            '''
            填充的起始像素值位于车牌区域内，将车牌区域中蓝色和黄色的点填充成白色
            floodFill(image, mask, seedPoint, newVal, loDiff=None, upDiff=None, flags=None)
            floodFill( 1.操作的图像, 2.掩模, 3.起始像素值，4.填充的颜色, 5.填充颜色的低值， 6.填充颜色的高值 ,7.填充的方法)   (255, 255, 255)是白色
                      mask = np.zeros(shape=[img_h+2,img_w+2],dtype=np.uint8)
            loDiff,upDiff = 30,30;(loDiff,) * 3=(loDiff,loDiff,loDiff)，是相对于seed种子点像素可以往下的像素值，即seed(B0,G0,R0)，泛洪(floodFill)区域下界为（B0-loDiff1,G0-loDiff2,R0-loDiff3）
            (upDiff1,upDiff2,upDiff3)：是相对于seed种子点像素可以往上的像素值，即seed(B0,G0,R0)，泛洪区域上界为（B0+upDiff1,G0+upDiff2,R0+upDiff3）
            漫水填充法是一种用特定的颜色填充联通区域，通过设置可连通像素的上下限以及连通方式来达到不同的填充效果的方法。
            漫水填充经常被用来标记或分离图像的一部分以便对其进行进一步处理或分析，也可以用来从输入图像获取掩码区域，掩码会加速处理过程，或只处理掩码指定的像素点，操作的结果总是某个连续的区域。
                                                                                
            所谓漫水填充，简单来说，自动选中了和种子点相连的区域，接着将该区域替换成指定的颜色。
            做漫水填充的目的有两个，第一个是预处理的时候车牌轮廓可能有残缺，做完漫水填充后可以把剩余的部分补全，第二个目的是进一步排除非车牌区域。
            掩膜mask，就是用于进一步控制哪些区域将被填充颜色.用于覆盖的特定图像或物体称为掩模或模板,数字图像处理中,掩模为二维矩阵数组,有时也用多值图像。
            
                    提取感兴趣区,用预先制作的感兴趣区掩模与待处理图像相乘,得到感兴趣区图像,感兴趣区内图像值保持不变,而区外图像值都为0。
            当邻近像素点位于给定的范围（从loDiff到upDiff）内或在原始seedPoint像素值范围内时，FloodFill函数就会为这个点涂上颜色。
            flags为11 1111 1111 0000 0100，十进制为261892
            (255, 255, 255)这里的顺序是(B,G,R)
                      
                      
            '''
            cv2.circle(flood_img,center=(col,row),radius=2,color=(0,0,255),thickness=2)
            '''
            cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]),根据给定的圆心和半径等画圆
            center:圆心位置；radius：圆的半径；color：圆的颜色；thickness：圆形轮廓的粗细（如果为正），负厚度表示要绘制实心圆；lineType：圆边界的类型；shift：中心坐标和半径值中的小数位数。
            '''
            #cv2.imwrite('./carIdentityData/opencv_output/floodfill%d.jpg'%(i), flood_img)
            #cv2.imwrite('./carIdentityData/opencv_output/flood_mask%d.jpg'%(i), mask)
            #cv2.imwrite('./carIdentityData/opencv_output/mask_image%d.jpg'%(i), src_image)
            seed_cnt += 1
            if seed_cnt >= valid_seed_num:
                break
    #======================调试用======================#
    show_seed = np.random.uniform(1,100,1).astype(np.uint16)
    '''
    numpy.random.uniform(low,high,size)，从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
           low: 采样下界，float类型，默认值为0；
           high: 采样上界，float类型，默认值为1；
           size: 输出样本数目，为int或元组(tuple)类型
    返回值：ndarray类型，其形状和参数size中描述一致
    '''
    #cv2.imshow('floodfill'+str(show_seed),flood_img)
    #cv2.imshow('flood_mask'+str(show_seed),mask)
    
    #cv2.imwrite('./carIdentityData/opencv_output/floodfill%d.jpg'%(show_seed), flood_img)
    #cv2.imwrite('./carIdentityData/opencv_output/flood_mask%d.jpg'%(show_seed), mask)
    #======================调试用======================#
    # 获取掩模上被填充点的像素点，并求点集的最小外接矩形
    mask_points = []
    for row in range(1,img_h+1):
        for col in range(1,img_w+1):
            if mask[row,col] != 0:
                mask_points.append((col-1,row-1))#把不是黑色的像素点添加进mask_points，把mask被填充成白色的点集添加进去
    mask_rotateRect = cv2.minAreaRect(np.array(mask_points))#获取点集的最小矩形
    if verify_scale(mask_rotateRect):
        return True,mask_rotateRect
    else:
        return False,mask_rotateRect

# 车牌定位
def locate_carPlate(orig_img,pred_image):
    car_plate_w, car_plate_h = 136, 36 #dengjie.tkadd
    carPlate_list = []
    temp1_orig_img = orig_img.copy() #调试用
    temp2_orig_img = orig_img.copy() #调试用
    #cloneImg,contours,heriachy = cv2.findContours(pred_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours, heriachy = cv2.findContours(pred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)    #dengjie
    # RETR_EXTERNAL找最外层轮廓，CHAIN_APPROX_SIMPLE仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours向量内，拐点与拐点之间直线段上的信息点不予保留。heriachy这里没有用到
    for i,contour in enumerate(contours):     #enumerate同时获得列表或者字符串的索引和值，i是索引，contour是值
        cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 0), 2)#用绿色线宽为2的线条画出原图的所有轮廓
        # 获取轮廓最小外接矩形，返回值rotate_rect。rotate_rect是点集数组或向量（里面存放的是点的坐标），并且这个点集中的元素不定个数(中心(x,y), (宽,高), 旋转角度)
        rotate_rect = cv2.minAreaRect(contour)
        # 根据矩形面积大小和长宽比判断是否是车牌
        if verify_scale(rotate_rect):#return True
            ret,rotate_rect2 = verify_color(rotate_rect,temp2_orig_img)#返回True和mask上被填充点的像素点集的最小矩形
            if ret == False:
                continue
            # 车牌位置矫正
            car_plate = img_Transform(rotate_rect2, temp2_orig_img)#做仿射变换
            car_plate = cv2.resize(car_plate,(car_plate_w,car_plate_h)) #调整尺寸为后面CNN车牌识别做准备
            #========================调试看效果========================#
            box = cv2.boxPoints(rotate_rect2)#获取矩形顶点坐标
            for k in range(4):
                n1,n2 = k%4,(k+1)%4
                cv2.line(temp1_orig_img,(box[n1][0],box[n1][1]),(box[n2][0],box[n2][1]),(0,0,255),2)
                '''
                cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) → img
                       img:原图
                       pt1：直线起点坐标，(box[n1][0],box[n1][1])；(box[0][0],box[0][1])
                       pt2，直线终点坐标，(box[n2][0],box[n2][1])；(box[1][0],box[1][1])
                       color，当前绘画的颜色;如在BGR模式下，传递(255,0,0)表示蓝色画笔。
                       hickness，画笔的粗细，线宽。若是-1表示画封闭图像，如填充的圆。默认值是1
                       lineType，线条的类型
                '''
            #cv2.imshow('opencv_' + str(i), car_plate)
            cv2.imwrite('./carIdentityData/opencv_output/opencv_%d.jpg'%(i), car_plate)
            #========================调试看效果========================#
            carPlate_list.append(car_plate)
            #print('carPlate_list',carPlate_list)
    print('初步筛选车牌数量：',len(carPlate_list))

    cv2.imwrite('./carIdentityData/opencv_output/contour.jpg', temp1_orig_img)
    #cv2.imshow('contour', temp1_orig_img)
    return carPlate_list

# 左右切割
def horizontal_cut_chars(plate):#传入车牌二值图像中的字符部分
    char_addr_list = []
    area_left,area_right,char_left,char_right= 0,0,0,0
    img_w = plate.shape[1]

    # 获取车牌每列边缘像素点个数
    def getColSum(img,col):
        sum = 0
        for i in range(img.shape[0]):
            sum += round(img[i,col]/255)#二值图像，img[i,col]=0或255，获取每一列像素值为255的像素个数
        return sum;

    sum = 0
    for col in range(img_w):
        sum += getColSum(plate,col)#所有列白色像素点的个数总和
    
    col_limit = 0
    #col_limit = round(0.3*sum/img_w) # 每列边缘像素点必须超过均值的30%才能判断属于字符区域
    print('col_limit',sum,img_w,col_limit)#1344.0,136,6.0
    # 每个字符宽度也进行限制
    charWid_limit = [round(img_w/12),round(img_w/5)]#[11,27]
    is_char_flag = False

    for i in range(img_w):
        colValue = getColSum(plate,i)#每一列像素值为255的像素个数,ex:i=7时，colValue=3；i=8时，colValue=9；i=9时，colValue=19；i=18时，colValue=16；i=19时，colValue=0
        #print('colValue'+str(i),colValue)
        if colValue > col_limit:
            if is_char_flag == False:
                area_right = round((i+char_right)/2)#ex:i=8,area_right=4
                area_width = area_right-area_left#area_width=4
                char_width = char_right-char_left#char_width=0
                if (area_width>charWid_limit[0]) and (area_width<charWid_limit[1]):
                    char_addr_list.append((area_left,area_right,char_width))
                char_left = i#i=8
                area_left = round((char_left+char_right) / 2)#area_left=4
                is_char_flag = True
        else:
            if is_char_flag == True:
                char_right = i-1#19-1=18
                is_char_flag = False
        #print('is_char_flag'+str(i),area_left,area_right,char_right,char_left)
        #print('char_addr_list1',char_addr_list)
    # 手动结束最后未完成的字符分割
    if area_right < char_left:
        area_right,char_right = img_w,img_w#以img_w为右边界
        #area_right = round((img_w+char_right)/2)
        area_width = area_right - area_left
        char_width = char_right - char_left
        if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
            char_addr_list.append((area_left, area_right, char_width))#每一个字符区域的左右边界及字符的宽度
    print('char_addr_list',char_addr_list)#ex.char_addr_list=[(4, 20, 10), (20, 45, 14), (45, 62, 7), (62, 83, 13), (83, 96, 6), (96, 114, 14), (114, 132, 14)]
    return char_addr_list

def get_chars(car_plate):#传入车牌二值化图像
    char_w, char_h = 20, 20#dengjie.tkadd
    result_char_imgs = []
    for k in range(len(car_plate)):
        img_h,img_w = car_plate[k].shape[:2]
        h_proj_list = [] # 水平投影长度列表
        h_temp_len,v_temp_len = 0,0
        h_startIndex,h_end_index = 0,0 # 水平投影记索引
        h_proj_limit = [0.2,0.8] # 车牌在水平方向的轮廓长度少于20%或多余80%过滤掉
        char_imgs = []

    # 将二值化的车牌水平投影到Y轴，计算投影后的连续长度，连续投影长度可能不止一段
        h_count = [0 for i in range(img_h)]
        for row in range(img_h):
            temp_cnt = 0
            for col in range(img_w):
                if car_plate[k][row,col] == 255:
                    temp_cnt += 1
            h_count[row] = temp_cnt#统计每一行像素值为255的像素个数
            if temp_cnt/img_w<h_proj_limit[0] or temp_cnt/img_w>h_proj_limit[1]:
            #每一行像素值为255的像素个数/车牌宽度<0.2 或者 每一行像素值为255的像素个数/车牌宽度>0.8，过滤掉太长和太短的白线
                if h_temp_len != 0:
                    h_end_index = row-1
                    h_proj_list.append((h_startIndex,h_end_index))
                    print('h_proj_list1',h_proj_list)
                    h_temp_len = 0
                continue
            if temp_cnt > 0:
                if h_temp_len == 0:
                    h_startIndex = row#从0.2<(像素值为255的像素个数/img_w)<0.8 的行开始
                    h_temp_len = 1
                else:
                    h_temp_len += 1
            else:
                if h_temp_len > 0:
                    h_end_index = row-1
                    h_proj_list.append((h_startIndex,h_end_index))
                    print('h_proj_list2',h_proj_list)
                    h_temp_len = 0
        print('h_temp_len',h_temp_len)
    # 手动结束最后得水平投影长度累加
        if h_temp_len != 0:
            h_end_index = img_h-1
            h_proj_list.append((h_startIndex, h_end_index))#h_temp_len不等于0时再添加一对值
            print('h_proj_list',h_proj_list)#ex:[(1, 1), (7, 29), (34, 35)]或者[(2, 2), (7, 28), (34, 34)]
    # 选出最长的投影，该投影长度占整个截取车牌高度的比值必须大于0.5
        h_maxIndex,h_maxHeight = 0,0
        for i,(start,end) in enumerate(h_proj_list):
            if h_maxHeight < (end-start):
                h_maxHeight = (end-start)
                h_maxIndex = i
        if h_maxHeight/img_h < 0.5:
            return char_imgs
        chars_top,chars_bottom = h_proj_list[h_maxIndex][0],h_proj_list[h_maxIndex][1]#chars_top=h_proj_list[1][0],chars_bottom=h_proj_list[1][1]

        plates = car_plate[k][chars_top:chars_bottom+1,:]#获取车牌二值图像中的字符高度部分，plates比car_plate要窄，然后在进行字符分割
        cv2.imwrite('./carIdentityData/opencv_output/car_plate%d.jpg'%(k),car_plate[k])
        cv2.imwrite('./carIdentityData/opencv_output/plates%d.jpg'%(k), plates)
        char_addr_list = horizontal_cut_chars(plates)#ex.char_addr_list=[(4, 20, 10), (20, 45, 14), (45, 62, 7), (62, 83, 13), (83, 96, 6), (96, 114, 14), (114, 132, 14)]

        for i,addr in enumerate(char_addr_list):
            char_img = car_plate[k][chars_top:chars_bottom+1,addr[0]:addr[1]]#输出单个字符
            char_img = cv2.resize(char_img,(char_w,char_h))#resize字符
            char_imgs.append(char_img)
        #cv2.imshow('22',char_img)     #dengjie2
            cv2.imwrite('./carIdentityData/opencv_output/char_%d_%d.jpg'%(k,i),char_img)
        result_char_imgs.append(char_imgs)
    print('len(result_char_imgs)',len(result_char_imgs))
    #print('1122_result_char_imgs',result_char_imgs)
    #print('char_img',char_img)
    #return char_imgs
    return result_char_imgs

def extract_char(car_plate):#传入正确的车牌
    result_gray_plate = []
    result_binary_plate = []
    for i in range(len(car_plate)):
        gray_plate = cv2.cvtColor(car_plate[i],cv2.COLOR_BGR2GRAY)#转换成灰度图
        ret,binary_plate = cv2.threshold(gray_plate,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)#使用最大类间方差法将图像二值化，自适应找出最合适的阈值
        result_gray_plate.append(gray_plate)
        result_binary_plate.append(binary_plate)
    #cv2.imshow('extract_char_binary_plate',binary_plate)
        cv2.imwrite('./carIdentityData/opencv_output/extract_char_binary_plate%d.jpg'%(i),binary_plate)
    #char_img_list = get_chars(binary_plate)
    char_img_list = get_chars(result_binary_plate)
    #cv2.imshow('1',binary_plate)  #dengjie
    return char_img_list

def cnn_select_carPlate(plate_list,model_path):
    if len(plate_list) == 0:
        return False,plate_list
    g1 = tf.Graph()
    sess1 = tf.Session(graph=g1)
    '''
    Tensorflow中的图（tf.Graph）和会话（tf.Session）
    tf.Graph()表示实例化一个用于tensorflow计算和表示用的数据流图，不负责运行计算
    1、使用g = tf.Graph()函数创建新的计算图
    2、在with g.as_default():语句下定义属于计算图g的张量和操作
    3、在with tf.Session()中通过参数graph=xxx指定当前会话所运行的计算图
    4、如果没有显示指定张量和操作所属的计算图，则这些张量和操作属于默认计算图
    5、一个图可以在多个sess中运行，一个sess也能运行多个图
    '''
    with sess1.as_default():
        with sess1.graph.as_default():#使用此图作为默认图的上下文管理器
            model_dir = os.path.dirname(model_path)# 获取文件的完整目录，得到当前文件的绝对路径
            saver = tf.train.import_meta_graph(model_path)#用来加载训练模型meta文件中的图,以及图上定义的结点参数包括权重偏置项等需要训练的参数,也包括训练过程生成的中间参数
            saver.restore(sess1, tf.train.latest_checkpoint(model_dir))#自动找到最近保存的变量文件并载入
            graph = tf.get_default_graph()#获取当前默认的计算图
            net1_x_place = graph.get_tensor_by_name('x_place:0')#按tensor名称获取tensor信息，Tensor("x_place:0", shape=(?, 36, 136, 3), dtype=float32)
            #Once you know the name you can fetch the Tensor using <name>:0 (0 refers to endpoint which is somewhat redundant)
            #一旦知道名称，就可以使用<name>：0来获取Tensor（0表示冗余的端点）
            print('net1_x_place',net1_x_place)
            net1_keep_place = graph.get_tensor_by_name('keep_place:0')#Tensor("keep_place:0", dtype=float32)
            print('net1_keep_place',net1_keep_place)
            net1_out = graph.get_tensor_by_name('out_put:0')#Tensor("out_put:0", shape=(?, 2), dtype=float32),获取cnn_construct()的输出
            print('net1_out',net1_out)

            result_plate_list=[]
            input_x = np.array(plate_list)
            net_outs = tf.nn.softmax(net1_out)
            preds = tf.argmax(net_outs,1) #预测结果，按行取最大值对应的索引
            probs = tf.reduce_max(net_outs,reduction_indices=[1]) #结果概率值，按行取概率的最大值
            pred_list,prob_list = sess1.run([preds,probs],feed_dict={net1_x_place:input_x,net1_keep_place:1.0})
            print('pred_list',pred_list)
            print('prob_list',prob_list)
            # 选出概率最大的车牌
            result_index,result_prob = -1,0.
            for i,pred in enumerate(pred_list):
                #if pred==1 and prob_list[i]>=result_prob:
                if pred==1 and prob_list[i]>=0.9:
                    result_index,result_prob = i,prob_list[i]#0,概率
                    print('in pred')
                    print(result_index,result_prob)
                    result_plate_list.append(plate_list[i])
            if result_index == -1:
                return False,plate_list[0]#返回第一张车牌
            else:
                #return True,plate_list[result_index]#返回正确的索引对应的车牌
                return True,result_plate_list

def cnn_recongnize_char(img_list,model_path):
    g2 = tf.Graph()
    sess2 = tf.Session(graph=g2)
    result_text_list = []
    
    print('cnn_recongnize_char_len(img_list):',len(img_list))
    with sess2.as_default():
        with sess2.graph.as_default():
            for k in range(len(img_list)):
                text_list = []
                if len(img_list) == 0:
                    return text_list
                model_dir = os.path.dirname(model_path)
                saver = tf.train.import_meta_graph(model_path)
                saver.restore(sess2, tf.train.latest_checkpoint(model_dir))
                graph = tf.get_default_graph()
                net2_x_place = graph.get_tensor_by_name('x_place:0')
                net2_keep_place = graph.get_tensor_by_name('keep_place:0')
                net2_out = graph.get_tensor_by_name('out_put:0')

                data = np.array(img_list[k])
            # 数字、字母、汉字，从67维向量找到概率最大的作为预测结果
                net_out = tf.nn.softmax(net2_out)
                preds = tf.argmax(net_out,1)
                my_preds= sess2.run(preds, feed_dict={net2_x_place: data, net2_keep_place: 1.0})
                print('my_preds',my_preds)#ex.my_preds=[49 11 13  8 19  5  3]

                for i in my_preds:
                    text_list.append(char_table[i])
                    
                result_text_list.append(text_list)
            #return text_list
            print('cnn_recongnize_char_len(result_text_list):',len(result_text_list))
            #print('11_result_text_list',result_text_list)
            return result_text_list

if __name__ == '__main__':
    cur_dir = sys.path[0]
    car_plate_w,car_plate_h = 136,36
    char_w,char_h = 20,20
    plate_model_path = os.path.join(cur_dir, './carIdentityData/model/plate_recongnize/model.ckpt-1020.meta')
    char_model_path = os.path.join(cur_dir,'./carIdentityData/model/char_recongnize/model.ckpt-1030.meta')
    img = cv2.imread('./plate_pic/4.jpg')

    # 预处理
    pred_img = pre_process(img)

    # 车牌定位
    car_plate_list = locate_carPlate(img,pred_img)

    # CNN车牌过滤
    ret,car_plate = cnn_select_carPlate(car_plate_list,plate_model_path)#True,正确的车牌
    if ret == False:
        print("未检测到车牌")
        sys.exit(-1)#sys.exit(-1)告诉程序退出。它基本上只是停止继续执行python代码。-1只是传入的状态码。通常0表示成功执行，其他任何数字（通常为1）表示发生故障。
    #cv2.imshow('cnn_plate',car_plate)
    print('len(car_plate)',len(car_plate))
    for i in range(len(car_plate)):
        cv2.imwrite('./carIdentityData/opencv_output/cnn_plate%d.jpg'%(i), car_plate[i])

    # 字符提取
    char_img_list = extract_char(car_plate)

    # CNN字符识别
    text = cnn_recongnize_char(char_img_list,char_model_path)
    #print('text_len',text)
    for i in range(len(text)):
        print('Recognition result_%d:'%(i+1))
        print(text[i])

    cv2.waitKey(0)
