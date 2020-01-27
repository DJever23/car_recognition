import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']#10
alphbets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']#26
chinese = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
                'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
                'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
                'zh_zang', 'zh_zhe']#31

class char_cnn_net:
    def __init__(self):
        self.dataset = numbers + alphbets + chinese
        self.dataset_len = len(self.dataset)#67类
        self.img_size = 20
        self.y_size = len(self.dataset)#类别，67，多分类问题
        self.batch_size = 100

        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size], name='x_place')#None 指 batch_size 的大小
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.y_size], name='y_place')#None 指 batch_size 的大小
        self.keep_place = tf.placeholder(dtype=tf.float32, name='keep_place')

    def cnn_construct(self):
        x_input = tf.reshape(self.x_place, shape=[-1, 20, 20, 1])    #shape为 [ batch, in_height, in_weight, in_channel ],-1表示此维度由函数自行计算

        cw1 = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01), dtype=tf.float32)
        '''
        tf.Variable定义变量
        tf.random_normal()函数用于从“服从指定正态分布的序列”中随机取出指定个数的值。tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
                          shape: 输出张量的形状;mean: 正态分布的均值，默认为0;stddev: 正态分布的标准差，默认为1.0;dtype: 输出的类型，默认为tf.float32;
                          seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样; name: 操作的名称
        cw1 shape为 [ filter_height, filter_weight, in_channel, out_channels ],in_channel是图像通道数,和input的in_channel要保持一致,out_channel是卷积核数量。
        '''
        cb1 = tf.Variable(tf.random_normal(shape=[32]), dtype=tf.float32)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input,filter=cw1,strides=[1,1,1,1],padding='SAME'),cb1))
        '''
        tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
                      input : 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，
                              其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。
                      filter：卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，
                              filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
                      strides：卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
                      padding：string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
                               若padding设置的值是SAME，输出大小等于输入大小除以步长后向上取整；若padding设置的值是VALID，输出大小等于输入大小减去滤波器大小加上1，最后再除以步长后向上取整
                      use_cudnn_on_gpu：bool类型，是否使用cudnn加速，默认为true
        tf.nn.conv2d输出为一个张量
        tf.nn.bias_add(value,bias,data_format=None,name=None)
                       value：一个Tensor,类型为float,double,int64,int32,uint8,int16,int8,complex64,或complex128
                       bias：一个 1-D Tensor,其大小与value的最后一个维度匹配；必须和value是相同的类型,除非value是量化类型,在这种情况下可以使用不同的量化类型.
                       将bias添加到value，返回与value具有相同类型的Tensor.
        tf.nn.relu(features, name = None)
                  这个函数的作用是计算激活函数 relu，即 max(features, 0)。将大于0的保持不变，小于0的数置为0。
        此行代码包含了卷积，计算(偏置项)，激活，最终输出tensor(feature)的shape为(-1,20,20,32)               
        '''
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        '''
        tf.nn.max_pool(value, ksize, strides, padding, name=None)
                       value:需要池化的输入,通常是feature map，shape依然是[batch, height, width, channels]
                       ksize:池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
                       strides:和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
                       padding：和卷积类似，可以取'VALID' 或者'SAME'
        输出tensor的shape为:(-1,10,10,32)
        '''
        conv1 = tf.nn.dropout(conv1, self.keep_place)
        '''
        tf.nn.dropout()是tensorflow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层.Dropout就是在不同的训练过程中随机扔掉一部分神经元。
                      也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
        tf.nn.dropout(x,keep_prob,noise_shape=None,seed=None,name=None)
                      x：指输入，输入tensor
                      keep_prob: float类型，每个元素被保留下来的概率，设置神经元被选中的概率,在初始化时keep_prob是一个占位符, 
                                 keep_prob = tf.placeholder(tf.float32).tensorflow在run时设置keep_prob具体的值，例如keep_prob: 0.75
                      noise_shape: 一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape。
        dropout必须设置概率keep_prob，并且keep_prob也是一个占位符，跟输入是一样的.train的时候才是dropout起作用的时候，test的时候不应该让dropout起作用
               这里是self.keep_place，在训练时会给出参数
        '''

        cw2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01), dtype=tf.float32)
        #因为第一层的输出为(-1,10,10,32),所以第二层的卷积核的in_channel是32，有64个卷积核
        cb2 = tf.Variable(tf.random_normal(shape=[64]), dtype=tf.float32)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,filter=cw2,strides=[1,1,1,1],padding='SAME'),cb2))
        #输出tensor的shape为(-1,10,10,64)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #输出tensor的shape为(-1,5,5,64)
        conv2 = tf.nn.dropout(conv2, self.keep_place)

        cw3 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01), dtype=tf.float32)
        #第三层的卷积核的in_channel是64，有128个卷积核
        cb3 = tf.Variable(tf.random_normal(shape=[128]), dtype=tf.float32)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,filter=cw3,strides=[1,1,1,1],padding='SAME'),cb3))
        #输出tensor的shape为(-1,5,5,128)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #输出tensor的shape为(-1,3,3,128)
        conv3 = tf.nn.dropout(conv3, self.keep_place)

        conv_out = tf.reshape(conv3, shape=[-1, 3 * 3 * 128])#输出的tensor形状是(-1,3,3,128),这里将其reshape成(-1,3*3*128)

        fw1 = tf.Variable(tf.random_normal(shape=[3 * 3 * 128, 1024], stddev=0.01), dtype=tf.float32)
        fb1 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully1 = tf.nn.relu(tf.add(tf.matmul(conv_out, fw1), fb1))
        #tf.matmul为两个矩阵相乘，矩阵乘法，conv_out为(-1,3*3*128),fw1为(3*3*128,1024),相乘后输出为(-1,1024)
        fully1 = tf.nn.dropout(fully1, self.keep_place)

        fw2 = tf.Variable(tf.random_normal(shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fb2 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully2 = tf.nn.relu(tf.add(tf.matmul(fully1, fw2), fb2))
        fully2 = tf.nn.dropout(fully2, self.keep_place)

        fw3 = tf.Variable(tf.random_normal(shape=[1024, self.dataset_len], stddev=0.01), dtype=tf.float32)
        fb3 = tf.Variable(tf.random_normal(shape=[self.dataset_len]), dtype=tf.float32)
        fully3 = tf.add(tf.matmul(fully2, fw3), fb3, name='out_put')
        #输出一个(-1,self.dataset_len)的矩阵，-1表示batch_size个样本，（-1,67）

        return fully3

    def train(self,data_dir,save_model_path):
        print('ready load train dataset')
        X, y = self.init_data(data_dir)#X为一个数组，有图片个数个元素，每个元素是一个灰色图像数据；y是一个二维矩阵，有图片个数行，67列
        print('success load' + str(len(y)) + 'datas')
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
        '''
        from sklearn.model_selection import train_test_split
        train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
        train_x, test_x, train_y, test_y =sklearn.model_selection.train_test_split(train_data,train_target,test_size=0.4, random_state=0,stratify=y_train)
        train_data：所要划分的样本特征集;train_target：所要划分的样本结果； test_size：样本占比，如果是整数的话就是样本的数量；
        random_state：是随机数的种子。随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
                      比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。设置random_state=0得到的随机数是一样的
        '''

        out_put = self.cnn_construct()
        predicts = tf.nn.softmax(out_put)
        '''
        tf.nn.softmax(logits,axis=None,name=None,dim=None)
                      logits：一个非空的Tensor。必须是下列类型之一:half，float32，float64
                      axis：将在其上执行维度softmax。默认值为-1，表示最后一个维度
                      默认按行计算softmax，输出67类每一类的概率
                      shape为(-1,67)
        '''
        predicts = tf.argmax(predicts, axis=1)
        '''
        #tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引
                   axis=0时比较每一列的元素，将每一列最大元素所在的索引记录下来，最后输出每一列最大元素所在的索引数组。
                   axis=1的时候，将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组。
                   数量为batch_size个索引,1*batch_size的向量
        '''
        actual_y = tf.argmax(self.y_place, axis=1)#self.y_place的shape=[None, 67],输出为(1,None)的向量,也就是1*batch_size
        #print('aa',self.y_place)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, actual_y), dtype=tf.float32))
        '''
        tf.equal逐个元素进行判断，如果相等返回True，不相等返回False。
        tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。
        tf.reduce_mean函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。如果不指定轴，则计算所有元素的均值
        预测值predicts与真值actual_y逐个元素判断并将其转换为float32格式，也就是0,1，再全部相加求均值即为精度accuracy

        '''
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_put, labels=self.y_place))
        '''
        tf.nn.softmax_cross_entropy_with_logits(logits=out_put, labels=self.y_place)交叉熵损失函数
                     logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes
                     labels：实际的标签，大小同上
        
        '''
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        #tf.train.AdamOptimizer()函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。learning_rate：张量或浮点值。学习速率，值越大则表示权值调整动作越大
        train_step = opt.minimize(cost)#用adam优化器求解loss的最小值
        #train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        #print('fff',train_step)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()#初始化创建的变量
            sess.run(init)#一般调用tf.Session().run()方法来启动计算图
            step = 0#迭代训练的计数器
            saver = tf.train.Saver()#实例化对象
            while True:
                train_index = np.random.choice(len(train_x), self.batch_size, replace=False)
                '''
                numpy.random.choice(a, size=None, replace=True, p=None)
                            从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
                            replace:True表示可以取相同数字，False表示不可以取相同数字
                            数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
                从[0,len(train_x))中随机抽取batch_size个数组成一个1*batch_size的一维数组,1*100的数组
                '''
                train_randx = train_x[train_index]#这里应该是输出100个索引对应的图片样本数据
                train_randy = train_y[train_index]#输出100个索引对应的训练标签
                _, loss = sess.run([train_step, cost],
                                   feed_dict={self.x_place:train_randx,self.y_place:train_randy,self.keep_place:0.75})#dropout率为0.25，以25%的概率舍弃神经元
                step += 1

                if step % 10 == 0:
                    test_index = np.random.choice(len(test_x), self.batch_size, replace=False)
                    test_randx = test_x[test_index]
                    test_randy = test_y[test_index]
                    acc = sess.run(accuracy,feed_dict={self.x_place : test_randx, self.y_place : test_randy,
                                                       self.keep_place : 1.0})#不舍弃神经元，计算精确度
                    print(step, loss)
                    #每训练10次就在[0,len(test_x))中随机抽取batch_size个数字组成一个1*batch_size的一维数组，并且不能重复选取元素，然后计算并输出损失函数
                    if step % 50 == 0:
                        print('accuracy:' + str(acc))#每训练50次就输出精确度accuracy
                    if step % 500 == 0:
                        saver.save(sess, save_model_path, global_step=step)#训练500次时保存模型
                        '''
                        保存模型,一次saver.save()后可以在文件夹中看到新增的四个文件
                        saver.save(sess=sess, save_path=model_save_path, global_step=step)
                                   第一个参数sess=sess, 会话名字；
                                   第二个参数save_path=model_save_path, 设定权重参数保存到的路径和文件名；
                                   第三个参数global_step=step, 将训练的次数作为后缀加入到模型名字中。
                        '''
                    if acc > 0.99 and step > 1000:
                        saver.save(sess, save_model_path, global_step=step)#当精度大于0.99或者训练次数超过500次时保存模型
                        break

    def test(self,x_images,model_path):#传入测试图片和训练好的模型
        text_list = []
        out_put = self.cnn_construct()#x_images*67
        predicts = tf.nn.softmax(out_put)#每一个样本属于每一类的概率,x_images*67
        predicts = tf.argmax(predicts, axis=1)#每一个样本概率最大的值对应的索引,1*x_images
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            '''
            saver.restore(sess, model_path)重载模型的参数，继续训练或用于测试数据
                         第一个参数sess=sess, 会话名字
                         第二个参数save_path=model_save_path, 权重参数的保存路径和文件名
            '''
            preds = sess.run(predicts, feed_dict={self.x_place: x_images, self.keep_place: 1.0})#preds长度为x_images，比如有5张测试图片就是1*5
            for i in range(len(preds)):
                pred = preds[i].astype(int)#将索引值转换为整型
                text_list.append(self.dataset[pred])#将self.dataset列表中对应的pred索引的值也就是标签逐个添加到text_list列表中
            return text_list
            #输出测试图片的预测标签列表
    def list_all_files(self,root):
        files = []
        list = os.listdir(root)
        #os.listdir()方法：返回指定文件夹包含的文件或子文件夹名字的列表。该列表顺序以字母排序。此处root就是data_dir，返回的列表是'carIdentityData/cnn_char_train'下所有子文件夹的名字
        for i in range(len(list)):
            element = os.path.join(root, list[i])
            #需要先使用python路径拼接os.path.join()函数，将os.listdir()返回的名称拼接成文件或目录的绝对路径再传入os.path.isdir()和os.path.isfile().
            if os.path.isdir(element):#os.path.isdir()用于判断某一对象(需提供绝对路径)是否为目录
                temp_dir = os.path.split(element)[-1]
                #os.path.split分割文件名与路径,分割为data_dir和此路径下的文件名，[-1]表示只取data_dir下的文件名
                if temp_dir in self.dataset:
                    files.extend(self.list_all_files(element))
                    '''
                    extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。该方法没有返回值，但会在已存在的列表中添加新的列表内容。
                    第一次遍历carIdentityData/cnn_char_train下所有的目录，如果目录在dataset中，则重新调用self.list_all_files方法，传入element，第二次遍历每个文件夹下的图片并赋给element
                    '''
            elif os.path.isfile(element):
                files.append(element)
        #print('2',files)        
        return files
        #list_all_files方法将carIdentityData/cnn_char_train下所有类别下的图片(包含完整的目录)都写进files列表里
    def init_data(self,dir):
        X = []
        y = []
        if not os.path.exists(data_dir):
            raise ValueError('没有找到文件夹')
            #当程序出现错误，python会自动引发异常，也可以通过raise显示地引发异常。一旦执行了raise语句，raise后面的语句将不能执行。
        files = self.list_all_files(dir)
        #files是一个包含carIdentityData/cnn_char_train下所有类别下的图片(包含完整的目录)的完整列表
        for file in files:
            #访问列表
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if src_img.ndim == 3:
                continue
                #ndim返回的是数组的维度，返回的只有一个数，该数即表示数组的维度。
            resize_img = cv2.resize(src_img, (20, 20))
            X.append(resize_img)#将resize后的图片逐个添加到X空列表中
            dir = os.path.dirname(file)# 获取图片文件的完整目录，得到当前文件的绝对路径
            dir_name = os.path.split(dir)[-1]# 获取图片文件上一级目录名,[-1]表示只取carIdentityData/cnn_char_train下的文件名,即1,2,3...A,B,C...zh_cuan,zh_e...
            vector_y = [0 for i in range(len(self.dataset))]
            '''
            列表解析是Python迭代机制的一种应用，它常用于实现创建新的列表，因此用在[]中
            语法：[expression for iter_val in iterable if cond_expr]，expression为要输出的表达式，这里的输出为67个0的列表[0,0...0]
            '''
            index_y = self.dataset.index(dir_name)
            '''
            Python index()方法检测字符串中是否包含子字符串str，如果指定beg(开始)和end(结束)范围，则检查是否包含在指定范围内，如果包含子字符串返回开始的索引值，否则抛出异常。
            这里检查dataset中是否包含dir_name，若包含则返回索引值
            '''
            vector_y[index_y] = 1#索引处赋值为1，其他仍为0
            y.append(vector_y)#将vector_y列表逐个添加到y空列表中

        X = np.array(X)
        y = np.array(y).reshape(-1, self.dataset_len)# -1表示行数为图片个数，函数自行计算得出，67列
        return X, y
        #返回X为图片数据，y为所有训练文件夹下图片的标签

    def init_testData(self,dir):
        test_X = []
        if not os.path.exists(test_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(test_dir)
        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            test_X.append(resize_img)
        test_X = np.array(test_X)
        return test_X
        #init_testData方法将carIdentityData/cnn_char_test目录下所有的图片(包含完整的目录)写进files列表里，然后将图片转换为灰度图并resize成20*20，最后将其逐个添加到test_X列表中
        #最终返回test_X列表


if __name__ == '__main__':
    cur_dir = sys.path[0]
    #sys.path是python的搜索模块的路径集，返回的结果是一个list,path[0],在程序启动时初始化，是包含用来调用Python解释器的脚本的目录,其实就是存放需要运行的代码的路径
    data_dir = os.path.join(cur_dir, 'carIdentityData/cnn_char_train')
    #os.path.join()函数：连接两个或更多的路径名组件
    test_dir = os.path.join(cur_dir, 'carIdentityData/cnn_char_test')
    train_model_path = os.path.join(cur_dir, './carIdentityData/model/char_recongnize/model.ckpt')
    model_path = os.path.join(cur_dir,'./carIdentityData/model/char_recongnize/model.ckpt-530')

    #train_flag = 0
    train_flag =1
    net = char_cnn_net()

    if train_flag == 1:
        # 训练模型
        net.train(data_dir,train_model_path)
    else:
        # 测试部分
        test_X = net.init_testData(test_dir)
        text = net.test(test_X,model_path)
        print(text)
