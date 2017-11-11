import numpy as np
import time
import math
import treePlotter
from collections import Counter

def my_select(my_train_matrix,train_row,train_col):       #将经过处理的列表转换为各式矩阵
    train_y = my_train_matrix[:,-1]
    de1 = np.sum(train_y == 1)
    de_1 = np.sum(train_y == -1)
    HD = -(de1/train_row)*math.log(de1/train_row,2)-(de_1/train_row)*math.log(de_1/train_row,2)
    HDA = [0] * (train_col - 1)
    gDA = [0] * (train_col - 1)
    for decol in range(0,train_col-1) :
        if 1:
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
                    #print("A_matrix:",A_matrix[0][geshu])
                    #print("my_train_matrix:",my_train_matrix[derow][decol])
                    A_matrix[0][geshu] = my_train_matrix[derow][decol]
                    A_matrix[1][geshu] += 1
                    if my_train_matrix[derow][train_col-1] == 1:
                        A_matrix[2][geshu]+=1
                    geshu += 1
            for de_te in range(0, geshu):
                shi_1 = A_matrix[2][de_te] / A_matrix[1][de_te]
                if shi_1 > 0 and shi_1<1:
                    HDA[decol] += (A_matrix[1][de_te] / train_row)*(-(shi_1)*math.log(shi_1,2)-(1-shi_1)*math.log((1-shi_1),2))
                #print(A_matrix[1][de_te] / train_row)
            #print(HDA[decol])
    #print(HD)
    for k in range(0, train_col - 1):
        gDA[k] = HD - HDA[k]
        #print("第",k+1,"个特征信息增益为： ",gDA[k])
    #print("信息增益最大的特征位于：",gDA.index(max(gDA))+1," 值为: ",max(gDA))
    #print(labels)
    #print(my_train_matrix.shape[1])
    return gDA.index(max(gDA))


def my_tree(dataSet):
    classList = dataSet[:,-1]
    #if len(classList)>1:
    #    classList = classList[1:]
    #print(classList)
    if np.sum(classList == classList[0]) == len(classList):
        # claslist 所有元素都相等，即类别完全相同，停止划分
        #print("*****************enough**********************",classList[0])
        return classList[0]#splitDat aSet (dataSet,0,0)此时全是N，返回N
    if len(dataSet[0]) == 1:      #[0,0,0，0，'N]
        # 遍历完所有特征时返回出现欠数最多的
        #print("*****************nomore**********************",(Counter(classList).most_common(1))[0][0])
        #print((Counter(classList).most_common(1)))
        return (Counter(classList).most_common(1))[0][0]
    bestFeat = my_select (dataSet, dataSet.shape[0], dataSet.shape[1])#0- > 2
        # 选择最大的gain ratio对应的feature
    myTree = {bestFeat: {} }
        #多重字典构建树{outlook': {0:'I"
    #labels[bestFeat] = 0  #['temperature',' humidity’,'windy']-> [' temperature' 。’humidity' ]
    featValues = [de_row[bestFeat] for de_row in dataSet]#[0,0,1,2,2,2,1]
    #print(bestFeat,featValues)
    uniqueVals = set (featValues)
    #print(bestFeat, uniqueVals)
    #dataSet = np.delete(dataSet, bestFeat, axis=1)
    for value in uniqueVals:
        myTree[bestFeat][value] = my_tree(splitDataSet(dataSet,bestFeat,value))
        # 划分数据，为下一层计算准备
    #treePlotter.createPlot(myTree)
    return myTree

def splitDataSet(dataSet,axis,value) :
    de_count = 0
    for featVec in dataSet :
        if featVec[axis]== value: #只看当第i列的值= value时的item
            if de_count == 0:
                #print("featVec",featVec)
                reduceFeatVec = featVec[:axis]  # featVed的第i列给除去
                reduceFeatVec_list = list(reduceFeatVec)
                reduceFeatVec_list.extend(featVec[axis + 1:])
                #print("reduceFeatVec_list", reduceFeatVec_list)
                reduceFeatVec = np.array(reduceFeatVec_list)
                #print("reduceFeatVec", reduceFeatVec)
                retDataSet = np.ones((1,len(reduceFeatVec)))
                retDataSet[0] = reduceFeatVec
                #print("retDataSet[0]",retDataSet[0])
                #retDataSet = np.asmatrix(reduceFeatVec)
                #print(retDataSet)
                #print(retDataSet[0][1])
                #de_count += 1
                #print("####################HELLO#####################")
            if de_count > 0:
                #print("^^^^^^^^^^^^^^^^^^^^HELLO^^^^^^^^^^^^^^^^^^^^^")
                reduceFeatVec = featVec[:axis]  # featVed的第i列给除去
                reduceFeatVec_list = list(reduceFeatVec)
                reduceFeatVec_list.extend(featVec[axis + 1:])
                reduceFeatVec = np.array(reduceFeatVec_list)
                #print(reduceFeatVec)
                retDataSet = np.row_stack((retDataSet, reduceFeatVec))
                #print(retDataSet)
            de_count += 1
    #print(retDataSet)
    return retDataSet

def my_choose(de_tree,val_list):
    global de_count
    global my_answer
    if de_count == val_col-1:
        return my_answer
    for i in range(0,len(val_list)-1):
        if isinstance(de_tree,(float,str)):
            return de_tree
        if de_tree.__contains__(i):
            de_count += 1
            if de_tree[i].__contains__(val_list[i]):
                de_tree = de_tree[i][val_list[i]]
                my_answer = my_choose(de_tree, val_list)
                return my_answer
            else:
                return 1.0


def my_val(desicionTree,my_val_matrix,val_row,val_col):
    correct = 0
    for i in range(0,val_row):
        answer = my_val_matrix[i][val_col-1]
        my_answer = my_choose(desicionTree,my_val_matrix[i])
        print("answer",answer,"my_answer",my_answer)
        if answer == my_answer:
            correct += 1
    print("正确率： " , correct/val_row )


if __name__ == '__main__':
    start_time = time.time()
    train_set = 'E:/B,B,B,BBox/大三上/人工智能/lab4_Decision_Tree/train.csv'
    val_set = 'E:/B,B,B,BBox/大三上/人工智能/lab4_Decision_Tree/val.csv'
    pathtwo = 'E:/B,B,B,BBox/大三上/人工智能/lab4_Decision_Tree/test.csv'
    pathone = 'E:/B,B,B,BBox/大三上/人工智能/lab4_Decision_Tree/answer.txt'
    my_train_matrix = np.loadtxt(train_set,delimiter=",")
    train_row = my_train_matrix.shape[0]
    train_col = my_train_matrix.shape[1]
    print("train_row",train_row,"train_col",train_col)
    #for i in range (0,train_row):
    #    my_train_matrix[i][0] = int(my_train_matrix[i][0]/10)
    #print(my_train_matrix)

    my_val_matrix = np.loadtxt(train_set,delimiter=",")
    val_row = my_val_matrix.shape[0]
    val_col = my_val_matrix.shape[1]
    for i in range (0,val_row):
        my_val_matrix[i][0] = int(my_val_matrix[i][0]/10)

    desicionTree = my_tree(my_train_matrix)
    treePlotter.createPlot(desicionTree)
    global de_count
    de_count = 0
    my_val(desicionTree,my_val_matrix,val_row,val_col)

    end_time = time.time()
    print("程序运行时间为: ",end_time - start_time)
