# Stav Lidor 207299785
import numpy as np
import sys


"""
 Function Name: euclidean
 Input:p1,p2
 Output:the distance between the vectores
 Function Operation:compute the euclidean distance between the  vectores
"""
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2, ord=2)

"""
 Function Name: canberra
 Input:p1,p2
 Output:the distance between the vectores
 Function Operation:compute the canberra distance between the  vectores
"""
def canberra(p1, p2):
    sum = 0
    size = len(p1)
    for i in range(size):
        if p1[i] == 0 and p2[i] == 0:
            val = 0
        else:
            val = abs(p1[i] - p2[i]) / (abs(p1[i]) + abs(p2[i]))
        sum += val
    return sum
"""
 Function Name: most_app
 Input:list_f
 Output:the number that appears most of the time
 Function Operation:count how mach is number appear and find the numbers that appears most of the 
 times and return the number most low that appears most
"""
def most_app(list_f):
    l = []
    #count hpw mach appear for number in the list
    for x in list_f:
        l.append(list_f.count(x))
    # find the count of the number most appear
    max_app = max(l)
    size = len(list_f)
    # list_tag will be in her all the number that most appear
    list_tag = []
    for i in range(size):
        if list_f[i] not in list_tag and l[i] == max_app:
            list_tag.append(list_f[i])
    # return the most appear that minimum tag
    if 0 in list_tag:
        return 0
    if 1 in list_tag:
        return 1
    if 2 in list_tag:
        return 2

"""
 Function Name: knn
 Input:p- vactor that need to find his tag, train_x, train_y, k, dis- the fuction of the dis that use her
 Output:the tag of this vactor
 Function Operation: knn algo that find the tag according the tag most appear in k close neighbors
"""
def knn(p, train_x, train_y, k, dis):
    l = []
    size = len(train_x)
    # add list of tuple with the vector in train and is distance from the new vector 
    for x in range(size):
        l.append((x, dis(train_x[x], p)))
    # sort the list and in list_y take the k that is distance minimum
    l.sort(key=lambda x: (x[1], x[0]))
    list_y = []
    for i in range(k):
        list_y.append(train_y[l[i][0]])
    return most_app(list_y)



"""
 Function Name: to_norm_z
 Input:arrry_train, array_valid
 Output:norm train and norm train
 Function Operation: norm algo with the norm z score
"""
def to_norm_z(arrry_train, array_valid):
    size_col = 0
    size_row = len(arrry_train)
    line_valid = len(array_valid)
    col_valid = 0
    if line_valid > 0:
        col_valid = len(array_valid[0])
    if size_row > 0:
        size_col = len(arrry_train[0])
    # new arr of train and vaild
    arr_valid_new = np.zeros((line_valid, col_valid))
    arr_train_new = np.zeros((size_row, size_col))
    avg = np.mean(arrry_train, axis=0)
    std = np.std(arrry_train, axis=0)
    for i in range(size_row):
        for j in range(size_col):
            #changes the arrs with norm by avg and std of train
            arr_train_new[i][j] = (arrry_train[i][j] - avg[j]) / std[j]
            if i < line_valid and j < col_valid:
                arr_valid_new[i][j] = (array_valid[i][j] - avg[j]) / std[j]
    return np.asarray(arr_train_new), np.asarray(arr_valid_new)

"""
 Function Name: shuffle
 Input:x,y
 Output:shffule data
 Function Operation: shuffle x and y
"""
def shuffle(x, y):
    zip_xy = list(zip(x, y))
    np.random.shuffle(zip_xy)
    return list(zip(*zip_xy))

"""
 Function Name: svm
 Input:x, y, parameter
 Output:w matrix
 Function Operation: compute w by svm algo
"""
def svm(x, y, parameter):
    #take all the parmaters
    seed=parameter[0]
    iteration = parameter[1]
    r = parameter[2]
    lamda = parameter[3]
    col=x.shape[1]
    size = x.shape[0]
    #add bias to the train
    x = np.hstack((np.ones((size, 1)), x))
    #initialize w in zeros
    list_w = np.zeros((3, x.shape[1]))
    for j in range(iteration):
        #suffle by the seed
        np.random.seed(seed)
        shuffle_xy = shuffle(x, y)
        x = shuffle_xy[0]
        y = shuffle_xy[1]
        # move on the train 
        for i in range(size):
            #find the max that is not the real tag
            dis = np.dot(list_w, x[i])
            dis = np.delete(dis,y[i])
            y_tag = np.argmax(dis)
            if y_tag >= y[i]:
                y_tag+=1
            #updth all wi
            for k in range(3):
                list_w[k] = list_w[k] * (1 - lamda * r)
            # updth only if its not stand the role
            if max(0, 1 - np.dot(list_w[y[i]], x[i]) + np.dot(list_w[y_tag], x[i])) > 0:
                list_w[y[i]] += r * x[i]
                list_w[y_tag] -= r * x[i]
    return np.asarray(list_w)

"""
 Function Name: perceptron
 Input:train_x, train_y, parameter
 Output:w matrix
 Function Operation: compute w by perceptron algo
"""
def perceptron(train_x, train_y, parameter):
    #take all the parmaters
    iteration = parameter[1]
    seed = parameter[0]
    r = parameter[2]
    size = len(train_x)
    #add bais
    train_x = np.hstack((np.ones((size, 1)), train_x))
    #initialize w in zeros
    list_w = np.zeros((3, train_x.shape[1]))
    for j in range(iteration):
        #suffle data with seed
        np.random.seed(seed)
        shuffle_xy = shuffle(train_x, train_y)
        train_x = shuffle_xy[0]
        train_y = shuffle_xy[1]
        #move all the data
        for i in range(size):
            y_tag = np.argmax(np.dot(list_w, train_x[i]))
            #updth w whare its not real tag is answer
            if y_tag != train_y[i]:
                list_w[train_y[i]] = list_w[train_y[i]] + r * train_x[i]
                list_w[y_tag] = list_w[y_tag] - 1 * r * train_x[i]
    return np.asarray(list_w)

"""
 Function Name: pa
 Input:train_x, train_y, parameter
 Output:w matrix
 Function Operation: compute w by pa algo
"""
def pa(train_x, train_y, parameter):
    #take all the parameters
    seed=parameter[0]
    iteration = parameter[1]
    train_x = np.hstack((np.ones((train_x.shape[0], 1)), train_x))
    #initialize w with zeros
    list_w = np.zeros((3, train_x.shape[1]))
    size = len(train_x)
    y_e = np.zeros((train_x.shape[1]))
    for j in range(iteration):
        np.random.seed(seed)
        shuffle_xy = shuffle(train_x, train_y)
        train_x = shuffle_xy[0]
        train_y = shuffle_xy[1]
        for i in range(size):
            #like svm 
            dis = np.dot(list_w, train_x[i])
            dis = np.delete(dis, train_y[i])
            y_tag = np.argmax(dis)
            if y_tag >= train_y[i]:
                y_tag += 1
            up = max(0, 1 - np.dot(list_w[train_y[i]], train_x[i]) + np.dot(list_w[y_tag], train_x[i]))
            #for updth 
            r = up / (2 * (euclidean(train_x[i], y_e) ** 2))
            if up > 0:
                list_w[train_y[i]] = list_w[train_y[i]] + r * train_x[i]
                list_w[y_tag] = list_w[y_tag] - 1 * r * train_x[i]
    return np.asarray(list_w)

"""
 Function Name: knn_run
 Input:norml_train_x, y,  norml_test_x, dis - distance function for knn
 Output:array of tag
 Function Operation: return y tag that compute by knn algo
"""
def knn_run(norml_train_x, y,  norml_test_x, dis):
    knn_yhat = []
    size = norml_test_x.shape[0]
    #for all data in the test compute is tag
    for i in range(size):
        knn_yhat.append(knn(norml_test_x[i], norml_train_x, y, 9, dis))
    return np.asarray(knn_yhat)
"""
 Function Name: tag_algo
 Input:x- test data ,w
 Output:array of tag
 Function Operation: return y tag that compute by w that get from some algo
"""
def tag_algo(x, w):
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    size = len(x)
    y_tag = []
    #compute all the tag
    for i in range(size):
        y_tag.append(np.argmax(np.dot(w, x[i])))
    return np.asarray(y_tag)


# take all from argv
train_x_fname, train_y_fname, test_x_fname, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
train_x = np.loadtxt(train_x_fname, delimiter=',')  # load centroids
train_y = np.loadtxt(train_y_fname, delimiter=',')
test_x = np.loadtxt(test_x_fname, delimiter=',')
y = [int(i) for i in train_y]


s = len(test_x)
test_x = np.delete(test_x, 4, axis=1)
train_x = np.delete(train_x, 4, axis=1)
#normlize data by z score
norml_train_x, norml_test_x = to_norm_z(train_x, test_x)
#compute the tag by knn
knn_yhat = knn_run(norml_train_x.copy(), y,norml_test_x.copy(), canberra)
#compute tag by perceptron
per=[813,63,0.0001]
w = perceptron(norml_train_x, y, per)
per_y = tag_algo(norml_test_x, w)
#compute tag by pa
per=[865,21]       
w = pa(norml_train_x, y, per)
pa_y = tag_algo(norml_test_x, w)
#compute tag by svm
per=[7553,32,0.1,0.01]
w = svm(norml_train_x, y, per)
svm_y = tag_algo(norml_test_x, w)
#write by formta to file
f = open(output_file, "w")
for i in range(s):
    f.write(f"knn: {knn_yhat[i]}, perceptron: {per_y[i]}, svm: {svm_y[i]}, pa: {pa_y[i]}\n")

f.close()
