# libraries
import pandas as pd
from sklearn.cluster import KMeans

def train_model(df):
    # เลือก feature ที่จะดู
    X = df[['Alcohol', 'Malic_Acid']].copy()

    # KMeans แบ่งเป็น 3 กลุ่ม
    Model = (KMeans(n_clusters = 3 ,max_iter=300,  random_state= 20))

    # fit เพื่อเทรนโมเดล
    Model.fit(X)
    return Model

def predict_model(Alcohol, Malic_Acid):

    # read_csv เรียกไฟล์มาดู
    df = pd.read_csv("wine-clustering.csv", sep=",")
    print(df)

    # เรียกใช้เมธอด
    Model = train_model(df)

    # สร้างข้อมูล
    X_test = [Alcohol, Malic_Acid]
    X_test_new = pd.DataFrame([X_test],columns=['Alcohol', 'Malic_Acid'])

    # predict ทำนายข้อมูล
    result =Model.predict(X_test_new)
    print(result)
    if result==0:
        return "ไวน์ตัวนี้มีแอลกอฮอล์อยู่ในเกณฑ์ที่ดี และความเข้มข้นค่อนข้างสูง"
    elif result==1:
        return "ไวน์ตัวนี้มีแอลกอฮอล์ค่อนข้างต่ำ และความเข้มข้นค่อนข้างต่ำ"
    else:
        return "ไวน์ตัวนี้มีแอลกอฮอล์ค่อนข้างสูง แต่ความเข้มข้นค่อนข้างต่ำ"