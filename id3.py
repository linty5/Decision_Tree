import numpy as np
import time
import math
def my_select(my_train_matrix,train_row,train_col):       #将经过处理的列表转换为各式矩阵
    train_y = my_train_matrix[:,-1]
    de1 = np.sum(train_y == 1)
    de_1 = np.sum(train_y == -1)
    HD = -(de1/train_row)*math.log(de1/train_row,2)-(de_1/train_row)*math.log(de_1/train_row,2)
    HDA = [0] * (train_col - 1)
    gDA = [0] * (train_col - 1)
    for decol in range(0,train_col-1):
        A_matrix = np.zeros((3, train_row))
        for j in range(0, train_row):
            A_matrix[0][j] = -1
        geshu = 0
        for derow in range(0, train_row):
            if my_train_matrix[derow][decol] in A_matrix[0]:
                A_matrix[1][np.where(A_matrix[0] == my_train_matrix[derow][decol])]+=1
                if my_train_matrix[derow][train_col-1] == 1:
                    A_matrix[2][np.where(A_matrix[0] == my_train_matrix[derow][decol])]+=1
            if my_train_matrix[derow][decol] not in A_matrix[0]:
                A_matrix[0][geshu] = my_train_matrix[derow][decol]
                A_matrix[1][geshu] += 1
                if my_train_matrix[derow][train_col-1] == 1:
                    A_matrix[2][geshu]+=1
                geshu += 1
        for de_te in range(0, geshu):
            shi_1 = A_matrix[2][de_te] / A_matrix[1][de_te]
            if shi_1 != 0:
                HDA[decol] += (A_matrix[1][de_te] / train_row)*(-(shi_1)*math.log(shi_1,2)-(1-shi_1)*math.log((1-shi_1),2))
            #print(A_matrix[1][de_te] / train_row)
        #print(HDA[decol])
    #print(HD)
    for k in range(0, train_col - 1):
        gDA[k] = HD - HDA[k]
        print("第",k+1,"个特征信息增益为： ",gDA[k])
    print("信息增益最大的特征位于：",gDA.index(max(gDA))+1," 值为: ",max(gDA))

if __name__ == '__main__':
    start_time = time.time()
    train_set = 'E:/B,B,B,BBox/大三上/人工智能/lab4_Decision_Tree/train.csv'
    pathtwo = 'E:/B,B,B,BBox/大三上/人工智能/lab4_Decision_Tree/test.csv'
    pathone = 'E:/B,B,B,BBox/大三上/人工智能/lab4_Decision_Tree/answer.txt'
    my_train_matrix = np.loadtxt(train_set,delimiter=",")
    train_row = my_train_matrix.shape[0]
    train_col = my_train_matrix.shape[1]
    my_select(my_train_matrix, train_row,train_col)
    end_time = time.time()
    print("程序运行时间为: ",end_time - start_time)
