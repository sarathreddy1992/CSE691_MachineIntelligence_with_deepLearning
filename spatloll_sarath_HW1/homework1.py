import numpy as np
import time
import random
# - Fill in the code below the comment Python and NumPy same as in example.
# - Follow instructions in document.
###################################################################
# Example: Create a zeros vector of size 10 and store variable tmp.
# Python
pythonStartTime = time.time()
tmp_1 = [0 for i in range(10)]
pythonEndTime = time.time()

# NumPy
numPyStartTime = time.time()
tmp_2 = np.zeros(10)
numPyEndTime = time.time()
print('Python time: {0} sec.'.format(pythonEndTime-pythonStartTime))
print('NumPy time: {0} sec.'.format(numPyEndTime-numPyStartTime))


z_1 = None
z_2 = None
########################################################################################


# 1. Create a zeros array of size (3,5) and store in variable z.
# Python
pythonStartTime=time.time()
z_1=[[0]*5 for i in range(3)]

pythonEndTime=time.time()



# NumPy

numPyStartTime=time.time()
z_2=np.zeros((3,5))

numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))
print('NumPy time:{0} sec'.format(numPyEndTime-numPyStartTime))


############################################################################################

# 2. Set all the elements in first row of z to 7.

# Python
pythonStartTime=time.time()

z_1=[[0]*5 for i in range(3)]
for i in range(3):
    for j in range(5):
      if i==0:
        z_1[i][j]=7
      else:
          z_1[i][j]=0


pythonEndTime=time.time()



# NumPy

numPyStartTime=time.time()
z_2[0,:]=7

numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))

print('NumPy time:{0} sec'.format(numPyEndTime-numPyStartTime))



#####################################################


# 3. Set all the elements in second column of z to 9.
# Python

pythonStartTime=time.time()

#z_1=[[0]*5 for i in range(3)]
for i in range(3):
    for j in range(5):
      if j==1:
        z_1[i][j]=9



pythonEndTime=time.time()


# NumPy

numPyStartTime=time.time()
z_2[:,1]=9

numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))

print('NumPy time:{0} sec'.format(numPyEndTime-numPyStartTime))

#############################################################


# 4. Set the element at (second row, third column) of z to 5.
# Python
pythonStartTime=time.time()

#z_1=[[0]*5 for i in range(3)]
for i in range(3):
    for j in range(5):
      if i==1 and j==2:
        z_1[i][j]=5



pythonEndTime=time.time()



# NumPy
numPyStartTime=time.time()
z_2[1,2]=5

numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))
print('NumPy time:{0} sec'.format(numPyEndTime-numPyStartTime))

##############
print(z_1)
print(z_2)
##############


x_1 = None
x_2 = None
##########################################################################################
# 5. Create a vector of size 50 with values ranging from 50 to 99 and store in variable x.
# Python
pythonStartTime=time.time()

x_1=[50 for i in range(50)]
for i in range(50):
    x_1[i]=x_1[i]+i

pythonEndTime=time.time()




# NumPy
numPyStartTime=time.time()
x_2=np.linspace(50,99)

numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))
print('NumPy time:{0} sec'.format(numPyEndTime-numPyStartTime))

##############
print(x_1)
print(x_2)
##############


y_1 = None
y_2 = None
##################################################################################
# 6. Create a 4x4 matrix with values ranging from 0 to 15 and store in variable y.
# Python
pythonStartTime=time.time()

y_1=[[0]*4 for i in range(4)]
t = 0
for i in range(4):
    for j in range(4):
        y_1[i][j] = t
        t = t + 1
pythonEndTime = time.time()


pythonEndTime=time.time()



# NumPy
numPyStartTime=time.time()
y_2=np.arange(16).reshape(4,4)

numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))
print('NumPy time is {0} sec'.format(numPyEndTime-numPyStartTime))


##############
print(y_1)
print(y_2)
##############


tmp_1 = None
tmp_2 = None
####################################################################################
# 7. Create a 5x5 array with 1 on the border and 0 inside amd store in variable tmp.


# Python
pythonStartTime=time.time()
tmp_1=[[0]*5 for i in range(5)]
for i in range(5):
    for j in range (5):
        if i==0 or i==4 and j==0 or j==4:
            tmp_1[i][j]=1
        else:
            tmp_1[i][j]=0


pythonEndTime=time.time()





# NumPy
numPyStartTime=time.time()
tmp_2=np.ones((5,5))
tmp_2[1:-1,1:-1]=0

numPyEndTime=time.time()

print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))
print('NumPy time is {0} sec'.format(numPyEndTime-numPyStartTime))
##############
print(tmp_1)
print(tmp_2)
##############


a_1 = None; a_2 = None
b_1 = None; b_2 = None
c_1 = None; c_2 = None
#############################################################################################
# 8. Generate a 50x100 array of integer between 0 and 5,000 and store in variable a.
# Python

pythonStartTime=time.time()

a_1=[[0]*100 for i in range(50)]
t=0
for i in range(50):
    for j in range(100):
        a_1[i][j]=t
        t=t+1

pythonEndTime=time.time()



# NumPy
numPyStartTime=time.time()
a_2=np.arange(5000).reshape(50,100)

numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))
print('NumPy time is {0}sec'.format(numPyEndTime-numPyStartTime))





###############################################################################################
# 9. Generate a 100x200 array of integer between 0 and 20,000 and store in variable b.
# Python
pythonStartTime = time.time()
b_1 = [[0 for i in range(200)] for j in range(100)]
t = 0
for j in range(100):
    for i in range(200):
        b_1[j][i] = t
        t = t + 1

pythonEndTime=time.time()




# NumPy
numPyStartTime=time.time()
b_2=np.arange(20000).reshape(100,200)
print(b_2)
numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))
print('NumPy time is {0}sec'.format(numPyEndTime-numPyStartTime))




#####################################################################################
# 10. Multiply matrix a and b together (real matrix product) and store to variable c.



# Python
pythonStartTime = time.time()
pythonStartTime = time.time()
c_1 = [[0]*200 for row in range(50)]
for row in range(50):
    for col in range(200):
        c_1[row][col] = 0
        for rcol in range(100):
            c_1[row][col] = c_1[row][col] +  a_1[row][rcol] * b_1[rcol][col]
pythonEndTime = time.time()


# NumPy
numPyStartTime=time.time()
c_2=np.dot(a_2,b_2)
print(c_2)
numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))
print('NumPy time is {0}sec'.format(numPyEndTime-numPyStartTime))




d_1 = None; d_2 = None
################################################################################
# 11. Normalize a 3x3 random matrix ((x-min)/(max-min)) and store to variable d.



# Python
pythonStartTime = time.time()
d_1 = [[0 for i in range(3)] for j in range(3)]

for j in range(3):
    for i in range(3):
        d_1[j][i] = random.random()
max_d_1 = d_1[0][0]
min_d_1 = d_1[0][0]

for row in range(3):
    for col in range(3):
        if d_1[j][i] > max_d_1 :
            max_d_1 = d_1[j][i]
        if d_1[j][i] < min_d_1 :
            min_d_1 = d_1[j][i]

for row in range(3):
    for col in range(3):
        d_1[j][i] = (d_1[j][j]-min_d_1) / (max_d_1 - min_d_1)

pythonEndTime = time.time()


# NumPy

numPyStartTime=time.time()
x=np.random.random((3,3))
print(x)
d_2=(x-x.min())/(x.max()-x.min())
print(d_2)
numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))

print('NumPy time is {0} sec'.format(numPyEndTime-numPyStartTime))



##########
print(d_1)
print(d_2)
#########


################################################
# 12. Subtract the mean of each row of matrix a.


# Python

pythonStartTime = time.time()
column_1 = 0
a_1 = [[0 ]*100 for i in range(50)]
for i in range(50):
    mean_1 = 0
    for j in range(100):
        if j == 99:
            mean_1 = mean_1 + a_1[i][99]
            a_1[column_1][0] = mean_1/100
        if col < 99:
            mean_1 = mean_1 + a_1[i][j]
    column_1 = column_1+1

for i in range (50):
    for j in range (100):
        a_1[i][j] = a_1[i][j] - a_1[i][0]
pythonEndTime = time.time()





# NumPy
numPyStartTime = time.time()
a_2 = a_2 - a_2.mean(axis=1, keepdims= True)
numPyEndTime = time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))

print('NumPy time is {0} sec'.format(numPyEndTime-numPyStartTime))

###################################################
# 13. Subtract the mean of each column of matrix b.
# Python

pythonStartTime = time.time()

column_1 = 0
b1 = [[0]*200 for row in range(1)]
for col in range(200):
    mean_1 = 0
    for row in range(100):
        if row == 99:
            mean_1 = mean_1 + b_1[99][col]
            b1[0][column_1] = mean_1/100
        if row < 99:
            mean_1 = mean_1 + b_1[row][col]
    column_1 = column_1+1

for col in range (200):
    for row in range (100):
        b_1[row][col] = b_1[row][col] - b1[0][col]
pythonEndTime = time.time()



# NumPy
numPyStartTime = time.time()
b_2 = b_2 - b_2.mean(axis=0, keepdims= True)
numPyEndTime = time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))
print('NumPy time is {0} sec'.format(numPyEndTime-numPyStartTime))



################
print(np.sum(a_1 == a_2))
print(np.sum(b_1 == b_2))
################

e_1 = None; e_2 = None
###################################################################################
# 14. Transpose matrix c, add 5 to all elements in matrix, and store to variable e.


# Python
pythonStartTime = time.time()
e_1 = [[0]*50 for i in range(200)]
for i in range(200):
    for j in range(50):
        e_1[i][j] = c_1[j][i]+5
pythonEndTime = time.time()



# NumPy
numPyStartTime=time.time()
c_2=np.transpose(c_2)
print(c_2)
e_2=5+c_2
print(e_2)
numPyEndTime=time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))
print('NumPy time is {0} sec'.format(numPyEndTime-numPyStartTime))


##################
print (np.sum(e_1 == e_2))
##################


#####################################################################################
# 15. Reshape matrix e to 1d array, store to variable f, and print shape of f matrix.


# Python

pythonStartTime = time.time()
f_1=[]
for row in range(len(e_1)):
    for col in range(len(e_1[0])):
        f_1.append(e_1[row][col])

print(len(f_1))
pythonEndTime = time.time()


# NumPy


numPyStartTime = time.time()
f_2 = e_2.reshape((np.product(e_2.shape),))
print(f_2.shape)
numPyEndTime = time.time()
print('Python time:{0} sec'.format(pythonEndTime-pythonStartTime))

print('NumPy time is {0} sec'.format(numPyEndTime-numPyStartTime))
