import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import carPlateIdentity
import cv2
from PIL import Image, ImageTk
#import threading
import time
import os
# video import create_capture
import sys
import getopt


class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    color_transform = {"green":("绿牌","#55FF55"), "yello":("黄牌","#FFFF00"), "blue":("蓝牌","#6666FF")}

    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        style=ttk.Style()
        #style.configure("BW.Tlable",foreground="blue",background="blue")
        style.configure("TButton",font=("Times",12),foreground="black",background="green")
        win.title("车牌识别")
        #win.state("zoomed")
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        #fill=tk.BOTH:水平和竖直方向填充;expand=tk.YES:扩展整个空白区;padx:x方向的外边距;pady:y方向的外边距
        frame_left.pack(side=LEFT,expand=1,fill=BOTH)
        #side=LEFT:按扭停靠在窗口的左侧
        frame_right1.pack(side=TOP,expand=1,fill=tk.Y)
        frame_right2.pack(side=RIGHT,expand=1)
        ttk.Label(frame_left, text='Original pic：',font=("Times",12)).pack(anchor="nw") #nw表示位置在上左，n是north，w是west
        ttk.Label(frame_right1, text='Plate Location：',font=("Times",12)).grid(column=0, row=0, sticky=tk.W)
        #位置在上面
        from_vedio_ctl = ttk.Button(frame_right2, text="Open camera", width=20, style="TButton",command=self.from_vedio)
        from_pic_ctl = ttk.Button(frame_right2, text="Open picture",width=20, style="TButton",command=self.from_pic)
        from_img_pre = ttk.Button(frame_right2, text="Show pre_img",width=20, style="TButton",command=self.show_img_pre)

        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        self.roi_ctl = ttk.Label(frame_right1)#车牌
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='Recognition result：',font=("Times",12)).grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="",font=("Times",12))#字符
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        self.color_ctl = ttk.Label(frame_right1, text="", font=("Times",12),width="20")
        self.color_ctl.grid(column=0, row=4, sticky=tk.W)
        from_pic_ctl.pack(anchor="se", pady="5")
        from_vedio_ctl.pack(anchor="se", pady="5")
        from_img_pre.pack(anchor="se", pady="5")


    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)#array转换成image图片
        imgtk = ImageTk.PhotoImage(image=im)#显示图片
        wide = imgtk.width()#图片的宽
        high = imgtk.height()#图片的高
        #print('wide',wide)
        #print('high',high)
        if wide > self.viewwide or high > self.viewhigh:#前面有定义，viewwide和viewhigh都为600
            wide_factor = self.viewwide / wide#viewwide除以图片的宽
            high_factor = self.viewhigh / high#viewhigh除以图片的高
            factor = min(wide_factor, high_factor)#取两者的最小值
            wide = int(wide * factor)
            if wide <= 0 : wide = 1#如果wide<=0,则令wide=1
            high = int(high * factor)
            if high <= 0 : high = 1
            im=im.resize((wide, high), Image.ANTIALIAS)#Image.ANTIALIAS：PIL高质量、抗锯齿
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk
        #此方法得到resize后且高质量的图片

    def show_roi(self, r, roi,color):#传入字符r与车牌图像roi
        if r :
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            self.r_ctl.configure(text=str(r))
            self.update_time = time.time()
                    
            try:
                c = self.color_transform[color]
                self.color_ctl.configure(text=c[0], background=c[1], state='enable')
            except:
                self.color_ctl.configure(state='disabled')
            
        if self.update_time + 8 < time.time():
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")
            self.color_ctl.configure(state='disabled')
    
    def show_img_pre(self):
       pre_img1=cv2.imread('/home/dengjie/dengjie/project/car_recognition/CarPlateIdentity/code/carIdentityData/opencv_output/blur.jpg')
       pre_img2=cv2.imread('/home/dengjie/dengjie/project/car_recognition/CarPlateIdentity/code/carIdentityData/opencv_output/sobel.jpg')
       pre_img3=cv2.imread('/home/dengjie/dengjie/project/car_recognition/CarPlateIdentity/code/carIdentityData/opencv_output/hsv_pic.jpg')
       pre_img4=cv2.imread('/home/dengjie/dengjie/project/car_recognition/CarPlateIdentity/code/carIdentityData/opencv_output/contour.jpg')
       pre_img5=cv2.imread('/home/dengjie/dengjie/project/car_recognition/CarPlateIdentity/code/carIdentityData/opencv_output/floodfill.jpg')
       pre_img6=cv2.imread('/home/dengjie/dengjie/project/car_recognition/CarPlateIdentity/code/carIdentityData/opencv_output/plates.jpg')
       pre_img7=cv2.imread('/home/dengjie/dengjie/project/car_recognition/CarPlateIdentity/code/carIdentityData/opencv_output/cnn_plate.jpg')
       
       cv2.imshow('blur',pre_img1)
       cv2.imshow('sobel',pre_img2)
       cv2.imshow('hsv_pic',pre_img3)
       cv2.imshow('contour',pre_img4)
       cv2.imshow('floodfill',pre_img5)
       cv2.imshow('plates',pre_img6)
       cv2.imshow('cnn_plate',pre_img7)
       
       while True:
           ch = cv2.waitKey(1)
           if ch == 27:
                break
       cv2.destroyAllWindows()
       
       
    def from_vedio(self):
       video=[0,"http://admin:admin@192.168.0.13:8081","http://admin:admin@iPhone.local:8081","http://admin:admin@10.119.223.51:8081"]
       '''
       默认情况下用户名和密码都是admin,客户端与IP摄像机服务器需处于同一局域网下,wifi
       参数为0表示打开内置摄像头，参数是视频文件路径则打开视频
       video="http://admin:admin@192.168.0.13:8081"  和  video="http://admin:admin@iPhone.local:8081"是使用wifi
       video="http://admin:admin@10.119.223.51:8081"使用流量，连接超时
       '''
       video = video[0]
       capture =cv2.VideoCapture(video)

       # 建个窗口并命名
       cv2.namedWindow("camera",1)
       num = 0

       # 用于循环显示图片，达到显示视频的效果
       while True:
            ret, frame = capture.read()
    
            # 在frame上显示test字符
            image1=cv2.putText(frame,'test', (50,100), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
                        thickness = 2, lineType = 2)
                
            cv2.imshow('camera',frame)
    
            # 不加waitkey（） 则会图片显示后窗口直接关掉
            key = cv2.waitKey(1)
    
            if key == 27:
                #esc键退出
                print("esc break...")
                break

            if key == ord(' '):
                # 保存一张图像
                num = num+1
                filename = "frames_%s.jpg" % num
                print('已保存图片：%s.jpg' % num)
                cv2.imwrite(filename,frame)
       cv2.destroyAllWindows()

       '''
        if self.thread_run:
            return
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                mBox.showwarning('警告', '摄像头打开失败！')
                #mBox.showwarning('warnning', 'open camera failed！')
                self.camera = None
                return
        #self.thread = threading.Thread(target=self.vedio_thread, args=(self,))
        self.thread = threading.Thread(target=self.vedio_thread)
        self.thread.setDaemon(True)
        self.thread.start()
        self.thread_run = True
       '''
    def from_pic(self):
        self.thread_run = False
        self.pic_path = askopenfilename(title="选择图片", filetypes=[("jpg", "*.jpg")])
        cur_dir = sys.path[0]
        plate_model_path = os.path.join(cur_dir, './carIdentityData/model/plate_recongnize/model.ckpt-1020.meta')
        char_model_path = os.path.join(cur_dir,'./carIdentityData/model/char_recongnize/model.ckpt-1030.meta')

        if self.pic_path:
            img_bgr = cv2.imread(self.pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            #预处理
            pred_img = carPlateIdentity.pre_process(img_bgr)
            # 车牌定位
            car_plate_list = carPlateIdentity.locate_carPlate(img_bgr,pred_img)
            # CNN车牌过滤
            ret,car_plate,color = carPlateIdentity.cnn_select_carPlate(car_plate_list,plate_model_path)
            cv2.imwrite('./carIdentityData/opencv_output/cnn_plate.jpg', car_plate)
            # 字符提取
            char_img_list = carPlateIdentity.extract_char(car_plate)
            # CNN字符识别
            text = carPlateIdentity.cnn_recongnize_char(char_img_list, char_model_path)
            print('result:', text)
            #r, roi = self.predictor.predict(img_bgr)#识别到的字符、定位的车牌图像
            self.show_roi(text, car_plate,color)

    '''
    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            _, img_bgr = self.camera.read()
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            #if time.time() - predict_time > 2:
                #self.show_roi(text, car_plate)
               # predict_time = time.time()
        print("run end")
    '''

def close_window():
    print("destroy")
    if surface.thread_run :
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()
    win.geometry('950x500')

    surface = Surface(win)
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()#进入消息循环

