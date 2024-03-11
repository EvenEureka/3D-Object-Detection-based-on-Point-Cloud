# python show_pkl.py

import pickle

path = '../result.pkl'

f = open(path, 'rb')
data = pickle.load(f)

# print(data[0].keys())
#dict_keys(['name', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score', 'boxes_lidar', 'frame_id'])
#truncated目标被截断的程度，0-1之间的浮点数，表示目标距离图像边界的程度
#occluded目标遮挡程度，0-3之间的整数，0完全可见，1部分遮挡，2大部分遮挡，3未知
#alpha目标观测角[-pi,pi]
#bbox目标2D检测框位置，左上和右下顶点的像素坐标，4维
#dimensions，3D目标尺寸，高宽长
#location目标3D框底面坐标，xyz相机坐标系
#rotation_y目标转向角[-pi,pi]

#要提交的目标为txt文件，每一行代表一个目标的信息，每一列的含义为：
# 类别 x y z w l h ry 置信度 (x y z w l h的单位为米，ry为弧度值)
# （注：xyz为中心点坐标，wlh为宽长高，ry为航向角，输出的航向角参考系需和标注文件一致，
# 类别和数字的对应关系为：行人-1、两轮车-2、三轮车-3、小汽车-4、半挂牵引车-5、卡车-6、巴士-7）

result='../result_20/'

for i in range(len(data)):
    frame_id=data[i].get('frame_id')
    frame_id=frame_id+'.txt'
    path= result + frame_id
    with open(path,'w')as f:
        for j in range(len(data[i].get('name'))):
            if(data[i].get('name')[j]=='car'):
                f.write("4")
                f.write(" ")
            f.write(str(data[i].get('location')[j][0]))
            f.write(" ")
            f.write(str(data[i].get('location')[j][1]))
            f.write(" ")
            f.write(str(data[i].get('location')[j][2]))
            f.write(" ")
            f.write(str(data[i].get('dimensions')[j][1]))
            f.write(" ")
            f.write(str(data[i].get('dimensions')[j][2]))
            f.write(" ")
            f.write(str(data[i].get('dimensions')[j][0]))
            f.write(" ")
            f.write(str(data[i].get('rotation_y')[j]))
            f.write(" ")
            f.write(str(data[i].get('score')[j]))
            f.write("\n")



