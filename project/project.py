import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
le = preprocessing.LabelEncoder()


#reading data
data = pd.read_csv(r"C:\Users\ASUS\Desktop\uni\8\datamining\project\adult.data")

#beacause we have "education" and "education-num" and they are the same, we can prun one
del data['education']

#we should convert categorical data to numerical
workclass = data["workclass"].tolist()
encoding_workclass = le.fit(data.loc[:,["workclass"]])
encoded_workclass=le.transform(workclass)
del data["workclass"]
data["workclass"] = encoded_workclass

maritalstatus = data["marital-status"].tolist()
encoding_maritalstatus = le.fit(data.loc[:,["marital-status"]])
encoded_maritalstatus = le.transform(maritalstatus)
del data["marital-status"]
data["marital-status"]=encoded_maritalstatus

occupation = data["occupation"].tolist()
encoding_occupation = le.fit(data.loc[:,["occupation"]])
encoded_occupation = le.transform(occupation)
del data["occupation"]
data["occupation"] = encoded_occupation

relationship = data["relationship"].tolist()
encoding_relationship = le.fit(data.loc[:,["relationship"]])
encoded_relationship = le.transform(relationship)
del data["relationship"]
data["relationship"] = encoded_relationship

race = data["race"].tolist()
encoding_race=le.fit(data.loc[:,["race"]])
encoded_race = le.transform(race)
del data["race"]
data["race"] = encoded_race

sex = data["sex"].tolist()
encoding_sex = le.fit(data.loc[:,["sex"]])
encoded_sex = le.transform(sex)
del data["sex"]
data["sex"]=encoded_sex

native_country = data["native-country"].tolist()
encoding_native_country = le.fit(data.loc[:,["native-country"]])
encoded_native_country = le.transform(native_country)
del data["native-country"]
data["native-country"] = encoded_native_country

income = data["income"].tolist()
encoding_income = le.fit(data.loc[:,["income"]])
encoded_income = le.transform(income)
del data["income"]
data["income"] = encoded_income

data.head()


#now we window the data
n = 3000   #size of each window is 3000           there are 11 windows
windowed_data = [data[i:i+n] for i in range(0,data.shape[0],n)]
#print("printing len windowed_data:" , len(windowed_data))


#we scale the attributes
scaler = StandardScaler()
scaled_data = scaler.fit_transform(windowed_data[0])


kmeans = KMeans(n_clusters=2, init='k-means++')


#windowing the data
data0 = pd.DataFrame(windowed_data[0])
data1 = pd.DataFrame(windowed_data[1])
data2 = pd.DataFrame(windowed_data[2])
data3 = pd.DataFrame(windowed_data[3])
data4 = pd.DataFrame(windowed_data[4])
data5 = pd.DataFrame(windowed_data[5])
data6 = pd.DataFrame(windowed_data[6])
data7 = pd.DataFrame(windowed_data[7])
data8 = pd.DataFrame(windowed_data[8])
data9 = pd.DataFrame(windowed_data[9])
test = pd.DataFrame(windowed_data[10])

scaled_test = scaler.fit_transform(test)

scaled_data0 = scaler.fit_transform(data0)
cluster0 = kmeans.fit_transform(scaled_data0)
predict0 = kmeans.predict(scaled_test)

scaled_data1 = scaler.fit_transform(data1)
cluster1 = kmeans.fit_transform(scaled_data1)
predict1 = kmeans.predict(scaled_test)

scaled_data2 = scaler.fit_transform(data2)
cluster2 = kmeans.fit_transform(scaled_data2)
predict2 = kmeans.predict(scaled_test)

scaled_data3 = scaler.fit_transform(data3)
cluster3 = kmeans.fit_transform(scaled_data3)
predict3 = kmeans.predict(scaled_test)

scaled_data4 = scaler.fit_transform(data4)
cluster4 = kmeans.fit_transform(scaled_data4)
predict4 = kmeans.predict(scaled_test)

scaled_data5 = scaler.fit_transform(data5)
cluster5 = kmeans.fit_transform(scaled_data5)
predict5 = kmeans.predict(scaled_test)

scaled_data6 = scaler.fit_transform(data6)
cluster6 = kmeans.fit_transform(scaled_data6)
predict6 = kmeans.predict(scaled_test)

scaled_data7 = scaler.fit_transform(data7)
cluster7 = kmeans.fit_transform(scaled_data7)
predict7 = kmeans.predict(scaled_test)

scaled_data8 = scaler.fit_transform(data8)
cluster8 = kmeans.fit_transform(scaled_data8)
predict8 = kmeans.predict(scaled_test)

scaled_data9 = scaler.fit_transform(data9)
cluster9 = kmeans.fit_transform(scaled_data9)
predict9 = kmeans.predict(scaled_test)



final_prediction=[]
#counting how many times a sample has been in group 0 and 1
for i in range(len(test)):

    zero = 0
    one = 0


    if (predict0[i] == 0):
        zero = zero + 1
    if (predict0[i] == 1):
        one = one + 1

    if (predict1[i] == 0):
        zero = zero + 1
    if (predict1[i] == 1):
        one = one + 1

    if (predict2[i] == 0):
        zero = zero + 1
    if (predict2[i] == 1):
        one = one + 1

    if (predict3[i] == 0):
        zero = zero + 1
    if (predict3[i] == 1):
        one = one + 1

    if (predict4[i] == 0):
        zero = zero + 1
    if (predict4[i] == 1):
        one = one + 1

    if (predict5[i] == 0):
        zero = zero + 1
    if (predict5[i] == 1):
        one = one + 1

    if (predict6[i] == 0):
        zero = zero + 1
    if (predict6[i] == 1):
        one = one + 1

    if (predict7[i] == 0):
        zero = zero + 1
    if (predict7[i] == 1):
        one = one + 1

    if (predict8[i] == 0):
        zero = zero + 1
    if (predict8[i] == 1):
        one = one + 1

    if (predict9[i] == 0):
        zero = zero + 1
    if (predict9[i] == 1):
        one = one + 1


    if(zero >= one):
        final_prediction.append(0)
    if(one > zero):
        final_prediction.append(1)


counter = 0
tp = 0
tn = 0
fp = 0
fn = 0
for i in range (len(test)):
    if(test['income'][30000+i] == final_prediction[i]):
        counter = counter + 1
        if(final_prediction[i] == 1):
            tn = tn+1
        if(final_prediction[i] == 0):
            tp = tp+1
    else:
        if(final_prediction[i] == 1):
            fn = fn+1
        if(final_prediction[i] == 0):
            fp = fp+1

accu = counter / len(test)
print(accu)

#print(final_prediction)
final = pd.DataFrame(final_prediction)
print("________________________")
#print(final)
#print("confusion matrix:")
#print(tp , "     ", fn)
#print(fp , "     ", tn)

print(confusion_matrix(test['income'] , final))

print(classification_report(test['income'],final))

