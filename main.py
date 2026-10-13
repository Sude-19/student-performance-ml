import pandas as pd

df = pd.read_csv("StudentPerformanceFactors.csv")

# VERİ SETİNİ TANIMA
print("ilk 5 satır:")
print(df.head())

print("\n Veri setinin boyutu:")
print(df.shape)

print("\nSütun İsimleri")
print(df.columns)

print("\nVeri setinin bilgisi:")
print(df.info())

print("\nSayısal Değişkenler:")
print(df.describe())

print("\nEksik değerler:")
print(df.isnull().sum())

print("\nTekrarlanan satır sayısı")
print(df.duplicated().sum())

#KATEGORİK SÜTUNLAR
print("\n Kategorik Değişkenlerin Değerleri:")

kategorik_sutunlar=df.select_dtypes(include="str").columns

for sutun in kategorik_sutunlar:
    print(f"\n{sutun}:")
    print(df[sutun].unique())

# EKSİK DEĞERLERİ MOD İLE DOLDURMA
df_filled=df.copy()
for sutun in kategorik_sutunlar:
    mod_degeri=df_filled[sutun].mode()[0]
    df_filled[sutun]=df_filled[sutun].fillna(mod_degeri)
print(f"Eksik veriler doldurulduktan sonra: ")
print(df_filled.isnull().sum())

# AYKIRI DEĞERLERİN TESPİT EDİLMESİ
sayisal_sutunlar=[
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]
aykiri_deger_maskesi=pd.Series(False, index=df_filled.index)

for sutun in sayisal_sutunlar:
    q1=df_filled[sutun].quantile(0.25)
    q3=df_filled[sutun].quantile(0.75)
    iqr=q3-q1

    alt_sinir=q1-1.5*iqr
    ust_sinir=q3+1.5*iqr

    sutun_maskesi= (
        (df_filled[sutun]<alt_sinir) |
        (df_filled[sutun]>ust_sinir)
    )
    aykiri_deger_maskesi= (
        aykiri_deger_maskesi| sutun_maskesi
    )
    print(f"\n{sutun}")
    print(f"Q1: {q1}")
    print(f"Q3: {q3}")
    print(f"IQR: {iqr}")
    print(f"Alt sınır: {alt_sinir}")
    print(f"Üst sınır: {ust_sinir}")
    print(f"Aykırı değer sayısı: {sutun_maskesi.sum()}")

print("\nEN AZ BİR AYKIRI DEĞER İÇEREN SATIR SAYISI:")
print(aykiri_deger_maskesi.sum())



# KATEGORİK VERİLERİ SAYISALLAŞTIRMA

# hedef değişken
y=df_filled["Exam_Score"]

# Özellikler
X=df_filled.drop(columns=["Exam_Score"])

print("\nX boyutu:")
print(X.shape)

print("\ny boyutu:")
print(y.shape)

print("\n KATEGORİK SÜTUNLAR:")
print(kategorik_sutunlar)

kategorik_sutunlar = X.select_dtypes(include="str").columns

print("\nKategorik sütunlar:")
print(kategorik_sutunlar)

ordinal_sutunlar= [
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level",
    "Family_Income",
    "Teacher_Quality",
    "Parental_Education_Level",
    "Distance_from_Home"
]

ordinal_mappings = {
    "Parental_Involvement": {
        "Low": 0,
        "Medium": 1,
        "High": 2
    },

    "Access_to_Resources": {
        "Low": 0,
        "Medium": 1,
        "High": 2
    },

    "Motivation_Level": {
        "Low": 0,
        "Medium": 1,
        "High": 2
    },

    "Family_Income": {
        "Low": 0,
        "Medium": 1,
        "High": 2
    },

    "Teacher_Quality": {
        "Low": 0,
        "Medium": 1,
        "High": 2
    },

    "Parental_Education_Level": {
        "High School": 0,
        "College": 1,
        "Postgraduate": 2
    },

    "Distance_from_Home": {
        "Near": 0,
        "Moderate": 1,
        "Far": 2
    }
}

for sutun, mapping in ordinal_mappings.items():
    X[sutun] = X[sutun].map(mapping)



nominal_sutunlar = [
    "Extracurricular_Activities",
    "Internet_Access",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Gender"
]

X=pd.get_dummies(
    X,
    columns=nominal_sutunlar,
    drop_first=True,
    dtype=int
)

print("\nENCODING SONRASI İLK 5 SATIR:")
print(X.head())

print("\nENCODING SONRASI VERİ TİPLERİ:")
print(X.dtypes)

print("\nENCODING SONRASI BOYUT:")
print(X.shape)

#VERİYİ TRAIN, VALIDATION VE TEST KÜMELERİNE AYIRMA
from sklearn.model_selection import train_test_split

# önce %80 train+validation, %20 test olarak ayıralım
X_train_val, X_test, y_train_val, y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train+validation'ın %60'ı train %40'ı validation
# böylece toplamda ½60 train %20 validation, %20 test olacak
X_train, X_val, y_train, y_val=train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.4,
    random_state=42
)

print("\nVERİ SETLERİNİN BOYUTLARI:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

print(f"\ny_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")
print(f"y_test: {y_test.shape}")

#STANDARDIZATION

from sklearn.preprocessing import StandardScaler

sayisal_sutunlar= [
   "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]

standard_scaler= StandardScaler()

#Veri setlerinin kopyalarını oluşturuyoruz.
X_train_standard=X_train.copy()
X_val_standard = X_val.copy()
X_test_standard = X_test.copy()

#scaler sadece TRAIN verisi üzerinde öğreniyor
X_train_standard[sayisal_sutunlar]= (
    standard_scaler.fit_transform(
        X_train[sayisal_sutunlar]
    )
)

#validation ve test verilerinde sadece transform
X_val_standard[sayisal_sutunlar]= (
    standard_scaler.transform(
        X_val[sayisal_sutunlar]
    )
)

X_test_standard[sayisal_sutunlar]=(
    standard_scaler.transform(
        X_test[sayisal_sutunlar]
    )
)

print("\nSTANDARDIZATION SONRASI:")
print(X_train_standard.head())

print("\nTRAIN ORTALAMALARI:")
print(X_train_standard[sayisal_sutunlar].mean())

print("\nTRAIN STANDART SAPMALARI:")
print(X_train_standard[sayisal_sutunlar].std())

#NORMALIZATION
from sklearn.preprocessing import MinMaxScaler

minmax_scaler=MinMaxScaler()


# veri setlerinin kopyalarını oluşturma
X_train_normalized = X_train.copy()
X_val_normalized = X_val.copy()
X_test_normalized = X_test.copy()

# scaler sadece TRAIN verşsş üzerinde öğreniyor
X_train_normalized[sayisal_sutunlar]= (
    minmax_scaler.fit_transform(
        X_train[sayisal_sutunlar]
    )
)

# validation ve test verilerinde sadece transform
X_val_normalized[sayisal_sutunlar]=(
    minmax_scaler.transform(
        X_val[sayisal_sutunlar]
    )
)

X_test_normalized[sayisal_sutunlar]=(
    minmax_scaler.transform(
        X_test[sayisal_sutunlar]
    )
)

print("\nNORMALIZATION SONRASI:")
print(X_train_normalized.head())

print("\nTRAIN MINIMUM DEĞERLERİ:")
print(X_train_normalized[sayisal_sutunlar].min())

print("\nTRAIN MAXIMUM DEĞERLERİ:")
print(X_train_normalized[sayisal_sutunlar].max())

#LİNEAR RERESSION 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Modeli oluşturma
linear_model=LinearRegression()

#Modeli train verisiyle eğit
linear_model.fit(
    X_train_standard,
    y_train
)

# Validation verisiyle tahmin yap
y_pred_linear=linear_model.predict(
    X_val_standard
)

#Performans metriklerini hesapla
mae_linear=mean_absolute_error(
    y_val,
    y_pred_linear
)

mse_linear=mean_squared_error(
    y_val,
    y_pred_linear
)

rmse_linear=mse_linear**0.5

r2_linear=r2_score(
    y_val,
    y_pred_linear
)

print("\nLINEAR REGRESSION SONUÇLARI")
print(f"MAE: {mae_linear:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f}")
print(f"R²: {r2_linear:.4f}")

# RIDGE REGRESSION

from sklearn.linear_model import Ridge

ridge_model=Ridge(alpha=1.0)

#Modeli train verisiyle eğit
ridge_model.fit(
    X_train_standard,
    y_train
)

# Validation verisiyle tahmin
y_pred_ridge=ridge_model.predict(
    X_val_standard
)

# Performans metrikleri
mae_ridge=mean_absolute_error(
    y_val,
    y_pred_ridge
)

mse_ridge=mean_squared_error(
    y_val,
    y_pred_ridge
)

rmse_ridge=mse_ridge**0.5

r2_ridge= r2_score(
    y_val,
    y_pred_ridge
)

print("\nRIDGE REGRESSION SONUÇLARI")
print(f"MAE: {mae_ridge:.4f}")
print(f"MSE: {mse_ridge:.4f}")
print(f"RMSE: {rmse_ridge:.4f}")
print(f"R²: {r2_ridge:.4f}")

# LASSO REGRESSION
from sklearn.linear_model import Lasso

lasso_model=Lasso(alpha=0.1)

#Modeli train verisiyle eğit

lasso_model.fit(
    X_train_standard,
    y_train
)

# Validation verisiyle tahmin
y_pred_lasso=lasso_model.predict(
    X_val_standard
)

# Performans metrikleri

mae_lasso=mean_absolute_error(
    y_val,
    y_pred_lasso
)

mse_lasso=mean_squared_error(
    y_val,
    y_pred_lasso
)

rmse_lasso=mse_lasso**0.5

r2_lasso=r2_score(
    y_val,
    y_pred_lasso
)

print("\nLASSO REGRESSION SONUÇLARI")
print(f"MAE: {mae_lasso:.4f}")
print(f"MSE: {mse_lasso:.4f}")
print(f"RMSE: {rmse_lasso:.4f}")
print(f"R²: {r2_lasso:.4f}")

# LASSO FEATURE SELECTION
ozellik_isimleri=X_train_standard.columns

lasso_katsayilari=lasso_model.coef_

print("\nLASSO KATSAYILARI:")

for isim, katsayi in zip(ozellik_isimleri,lasso_katsayilari):
    print(f"{isim}: {katsayi:.4f}")


#KARAR AĞACI VE RANDOM FOREST REGRESYON
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#KARAR AĞACI MODELİ
tree_model=DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

#RANDOM FOREST MODELİ
random_forest_model=RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

#MODELLERİ EĞİTME

tree_model.fit(
    X_train,
    y_train
)

random_forest_model.fit(
    X_train,
    y_train
)

#TAHMİN
tree_y_pred=tree_model.predict(X_val)

random_forest_y_pred=random_forest_model.predict(X_val)

# KARAR AĞACI PREFORMANSI

tree_mae=mean_absolute_error(
    y_val,
    tree_y_pred
)

tree_mse=mean_squared_error(
    y_val,
    tree_y_pred
)

tree_rmse=tree_mse**0.5

tree_r2=r2_score(
    y_val,
    tree_y_pred
)

# RANDOM FOREST PERFORMANSI
rf_mae=mean_absolute_error(
    y_val,
    random_forest_y_pred
)

rf_mse=mean_squared_error(
    y_val,
    random_forest_y_pred
)

rf_rmse=rf_mse**0.5

rf_r2=r2_score(
    y_val,
    random_forest_y_pred
)

# 18. SONUÇLARI YAZDIR

print("\nKARAR AĞACI SONUÇLARI")
print(f"MAE: {tree_mae:.4f}")
print(f"MSE: {tree_mse:.4f}")
print(f"RMSE: {tree_rmse:.4f}")
print(f"R²: {tree_r2:.4f}")


print("\nRANDOM FOREST SONUÇLARI")
print(f"MAE: {rf_mae:.4f}")
print(f"MSE: {rf_mse:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R²: {rf_r2:.4f}")

# K-FLOD CROSS VALIDATION

from sklearn.model_selection import KFold, cross_val_score

# 5-Fold Cross Validation
kfold=KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

#Linear Regressşon modeli üzerinde Cross Validation
kfold_r2=cross_val_score(
    linear_model,
    X_train_standard,
    y_train,
    cv=kfold,
    scoring="r2"
)

print("\nK-FOLD CROSS VALIDATION SONUÇLARI")

print("Fold R² skorları:")

for i, score in enumerate(kfold_r2, start=1):
    print(f"Fold {i}: {score:.4f}")

print(f"\nOrtalama R²: {kfold_r2.mean():.4f}")
print(f"Standart Sapma: {kfold_r2.std():.4f}")

#FINAL MODEL-LINEAR REGRESSION

#Final modeli tüm training verisiyle eğitiyoruz
linear_model.fit(
    X_train_standard,
    y_train
)

#test verisi üzerinde tahmin
y_pred_final=linear_model.predict(
    X_test_standard
)

#final model performans metrikleri
final_mae=mean_absolute_error(
    y_test,
    y_pred_final
)

final_mse=mean_squared_error(
    y_test,
    y_pred_final
)

final_rmse=final_mse**0.5

final_r2=r2_score(
    y_test,
    y_pred_final
)

print("\nFINAL LINEAR REGRESSION SONUÇLARI")
print(f"MAE: {final_mae:.4f}")
print(f"MSE: {final_mse:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"R²: {final_r2:.4f}")

#LINEAR REGRESSION KATSAYILARI

feature_names=X_train_standard.columns

linear_coefficients=linear_model.coef_

coefficient_df=pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": linear_coefficients
}
)

#mutlak katsayı değerine göre sıralama
coefficient_df["Absolute_Coefficient"]=(
    coefficient_df["Coefficient"].abs()
)

coefficient_df=coefficient_df.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print("\nLINEAR REGRESSION KATSAYILARI")
print(coefficient_df[["Feature", "Coefficient"]])

# VERİ DAĞILIMI GÖRSELLEŞTİRME

import matplotlib.pyplot as plt
import seaborn as sns

# 1. exam score dağılımı

plt.figure(figsize=(8,5))

sns.histplot(
    df["Exam_Score"],
    bins=20,
    kde=True
)

plt.title("Exam Score Dağılımı")
plt.xlabel("Exam Score")
plt.ylabel("Öğrenci Sayısı")

plt.tight_layout()
plt.show()

#2. Çalışma süresi ve sınav puanı ilişkisi
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="Hours_Studied",
    y="Exam_Score"
)

plt.title("Çalışma Süresi ve Sınav Puanı İlişkisi")
plt.xlabel("Çalışma Saati")
plt.ylabel("Sınav Puanı")

plt.tight_layout()
plt.show()

# MODEL PERFORMANS KARŞILAŞTIRILMASI
model_names=[
    "Linear Regression",
    "Ridge",
    "Lasso",
    "Decision Tree",
    "Random Forest"
]

r2_scores=[
    0.7143,
    0.7143,
    0.6872,
    0.4731,
    0.5541
]

plt.figure(figsize=(6,3))

sns.barplot(
    x=model_names,
    y=r2_scores
)

plt.title("Modellerin R² Performans Karşılaştırması")
plt.xlabel("Model")
plt.ylabel("R² Skoru")
plt.ylim(0,1)

plt.xticks(rotation=20)

plt.tight_layout()
plt.show()

# LINEAR REGRESSION KATSAYI GRAFİĞİ

top_coefficients = coefficient_df.head(10).sort_values(
    "Coefficient"
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=top_coefficients,
    x="Coefficient",
    y="Feature"
)

plt.title("Linear Regression - En Belirgin Katsayılara Sahip Özellikler")
plt.xlabel("Katsayı")
plt.ylabel("Özellik")

plt.tight_layout()
plt.show()