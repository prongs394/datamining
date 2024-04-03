from chefboost import Chefboost as chef
import pandas as pd

df = pd.read_csv("train.csv")
config = {'algorithm': 'C4.5'}
model = chef.fit(df, config = config)

#prediction = chef.predict(model, param = [1,85,66,29,0,26.6,0.351,31])
#print(prediction)

t = pd.read_csv("test.csv")
#print(t)

total = 69 #69 number of test sets
correct = 0 #number of test sets predicted correctly
cmatrix = [] #confusion matrix
tp = 0 # true positive
tn = 0 # true negetive
fp = 0 # false positive
fn = 0 # false negative
for i in range(0,69):
    this = t.iloc[i]
    prediction = chef.predict(model, param = [this[0],this[1],this[2],this[3],this[4],this[5],this[6],this[7]])

    if (prediction == this[8]):
        correct = correct+1
    if(this[8] == 1):
        if(prediction == 1):
            tp = tp +1
        elif(prediction < 1):
            fn = fn +1
    if(this[8] == 0):
        if(prediction < 1):
            tn = tn + 1
        elif(prediction == 1):
            fp = fp + 1

print("the decision tree guesses" ,correct/total,"right")

print("confusion matrix:")
cmatrix = [[tp , tn],[fp,fn]]
print("         actual","         yes","          no")
print("predicted")
print("yes                      ",tp,"          ",tn)
print("no                       ",fp,"          ",fn)


#print(this)

print("__________________")






