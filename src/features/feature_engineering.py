#feature engineering :Baukasten

import pandas as pd

import numpy as np

from scipy.stats import chi2_contingency

import statsmodels.formula.api as smf

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler # for normalization, standardization, dealing with outliers

import matplotlib.pyplot as plt

import seaborn as sns


#loading the csv file 
import pandas as pd
pd.set_option('display.max_columns', None)
df=pd.read_csv("src/data/clean/reviews_processed.csv",sep=",")



#Test base
test=df.sample(frac=0.1,random_state=42)
test.to_csv("test_raw_copy.csv",index=False)

#Train base
train=df.drop(test.index)
train.to_csv("train_raw_copy.csv",index=False)

#ich sollte hier noch unbalanced dataset berücksichtigen, dass von den jeweiligen Klassen anteilsweise gleich viele in train und test landen.

#overview
#print(train.head())
#print(train["rating"].value_counts())
#print(train[["location","supplier_response","company","domain"]].nunique())
#print(train.info())

#start feature selection: Try to find out which variables to keep

#Datum und Rating
#print(train["date"].dtype)
train['date'] = pd.to_datetime(train['date'], errors='coerce')
#print(train["date"].dtype)
#print(train.info())
#print("fehlende datumswerte:",train[train['date'].isna()])
train = train.dropna(subset=['date'])
train = train.reset_index(drop=True)
#print(train.info())
#Hier wäre es gut, wenn die Daten schon in datetime Format wären
train["year"]=train["date"].dt.year
train["month"]=train["date"].dt.month
train["day"]=train["date"].dt.day_of_week+1
print(pd.crosstab(train["year"],train["rating"],normalize="index"))

#Chi2 Test auf Unabhängigkeit, wenn unabhängig uninteressant, wenn abhängig gehts weiter zum V_cramer Test, der die Abhängigkeitsstärke angibt


#The chi-square test is used for categorical features in a dataset. We calculate the chi-square between each feature and the target and select the
#  desired number of features with the best chi-square scores. In order to properly apply the chi-square test to test the relationship between various features
#  of the dataset and the target variable, the following conditions must be met:

#the variables must be categorical
#the variables must be independently sampled
#the values must have an expected frequency greater than 5


stat,p=chi2_contingency(pd.crosstab(train["year"],train["rating"]))[:2]
print("stat:",stat,"p:",p)

V_Cramer = np.sqrt(stat/pd.crosstab(train["year"],
                                    train["rating"]).values.sum())
print("V_Cramer:",V_Cramer)


# Welche Jahre haben besonders Einfluss auf das Rating
# Variable Year

for i in pd.get_dummies(train["year"]):

    # Chi-Square test
    
    stat, p = chi2_contingency(pd.crosstab(
        train["rating"], pd.get_dummies(train["year"])[i]))[:2]

    # Cramer's V
    
    V_Cramer = np.sqrt(
        stat/pd.crosstab(train["rating"], train["year"]).values.sum())

    # Restrict to significant variables 
    
    if (p < 0.05):

        print(i, ":\n\n V von Cramer :", V_Cramer, end="\n\n")

import matplotlib.pyplot as plt
#train.groupby("year")["rating"].mean().plot()
#plt.show()
#Kein Trend zu erkennen
#Tests (Chi2) ergeben Ratings hängen hochsignifikant vom Jahr ab, d.h. die Ratings hängen absolut nicht zufällig vom Jahr ab. 
# Aber V Cramer gibt schwachen Zusammenhang (0.14). 
#Das Jahr hat nur geringen Einfluss auf das Rating
#geringer Einfluss, vermutlich kein starker Prädiktor allein, aber evtl. in Kombination nützlich

#Encoding des Jahres als Abstand zu Startjahr statt Jahreszahl
# Variable Year



train["year"] = train["year"] - train["year"].min()  #geht auch mit neuen Daten, nicht hartcodiert
#train["year"] = train["year"].replace(
#    {2011:15,2012:14,2013:13,2014:12,2015:11,2016:10,2017:9,2018:8,2019: 7, 2020: 6, 2021: 5, 2022: 4, 2023: 3, 2024: 2, 2025: 1})


#Dependence of the number of months with rating

from statsmodels.miscmodels.ordinal_model import OrderedModel

train["number_of_months"] = (
    (train["date"].dt.year - train["date"].dt.year.min()) * 12
    + train["date"].dt.month
)

model = smf.ols("rating ~ number_of_months", data=train).fit()
print(model.summary())
#Es gibt einen statistisch signifikanten, aber sehr schwachen negativen Zeittrend in den Bewertungen.
#train.groupby("number_of_months")["rating"].mean().plot()
#plt.show()

#Monat als Kategorie (Kaufverhalten im Sommer anders als im Winter?)
train["month_cat"] = train["date"].dt.month.astype("category")
#Analyse
print(pd.crosstab(train["month_cat"], train["rating"], normalize="index"))

#November (11) & Dezember (12):
#⭐ 1-Sterne deutlich höher (~26%)
#⭐ 5-Sterne deutlich niedriger (~57–62%)
#März (3):
#⭐ 5-Sterne am höchsten (~74%)
#⭐ 1-Sterne relativ niedrig (~11%)

#Visuell
#train.groupby("month_cat")["rating"].mean().plot()
#plt.show()


#Statistisch testen
print(chi2_contingency(pd.crosstab(train["month_cat"], train["rating"])))
n = pd.crosstab(train["month_cat"], train["rating"]).values.sum()
chi2 = 224.94

V_cramer = np.sqrt(chi2 / (n * (min(12,5)-1)))
print("V_cramer:",V_cramer)
#Monate können interessant sein drin zu lassen!!!


#Untersuche nun als nächstes Textlänge und Ratings, inwiefern es dort einen Zusammenhang gibt.
print(train[["review_length", "rating"]].corr())

#moderater negativer Zusammenhang -0.44 ab 0.5 bzw -0.5 wäre es ein starker Zusammenhang
#je länger die Review, desto niedriger tendenziell das Rating
# linearer Zusammenhang, das heißt nur längere reviews und schlechtere ratings treten zusammen auf


#sns.boxplot(x="rating", y="review_length", data=train)
#plt.show()

train["review_length_log"] = np.log1p(train["review_length"])
smf.ols("rating ~ review_length_log", data=train).fit()
print("Summary des OLS reports:",model.summary())
#„Wenn Reviews länger werden, sinkt das Rating minimal“ 

#ABER:

#der Effekt ist so klein, dass er praktisch kaum nutzbar ist
#Fazit: Länge allein(!!!) ist keine gute Abschätzung für das Rating, aber wir sollten es drin lassen um es kombiniert zu testen!!!

#Untersuche als nächstes ob verified / not verified einen Einfluss auf das Rating hat, Stichwort Fake Reviews, könnte man nach Company aufteilen wo gefakt wurde
pd.crosstab(train["verified"], train["rating"], normalize="index")
ct = pd.crosstab(train["verified"], train["rating"])

chi2, p, _, _ = chi2_contingency(ct)

n = ct.values.sum()
V = np.sqrt(chi2 / (n * (min(ct.shape)-1)))

print("p:",p, "V:",V)
#Cramers V 0.28 mittel bis stark!!! Untersuchen!!
#Erste Idee: Schauen ob bestimmte company auffällig ist
ct_company = pd.crosstab(
    [train["company"], train["verified"]],
    train["rating"],
    normalize="index"
)

#print(ct_company)

top_companies = train["company"].value_counts().nlargest(10).index

ct_top = pd.crosstab(
    [train["company"], train["verified"]],
    train["rating"],
    normalize="index"
).loc[top_companies]

print(ct_top)

mean_rating = train.groupby(["company", "verified"])["rating"].mean().unstack()
print("mean_rating:",mean_rating)
mean_rating["diff"] = mean_rating[1] - mean_rating[0]

print(mean_rating.sort_values("diff", ascending=False))

#Anzahl reviews pro company prüfen
counts = train.groupby(["company", "verified"])["rating"].count().unstack()
print("counts:",counts)

summary = train.groupby(["company", "verified"])["rating"].agg(["mean", "count"]).unstack()

summary.columns = ["mean_0", "mean_1", "count_0", "count_1"]

summary["diff"] = summary["mean_1"] - summary["mean_0"]

print(summary.sort_values("diff", ascending=False))

#Filtern auf companies mit mindestens 20 reviews
filtered = summary[(summary["count_0"] >= 20) & (summary["count_1"] >= 20)]

print(filtered.sort_values("diff", ascending=False))

#es gibt massive stabile unterschiede, verified reviews fallen deutlich besser aus, außer bei raceship(Ausreßer!!!), da ist es umgekehrt und es gibt viel mehr non-verified reviews
# mit viel besseren Bewertungen als verified, sieht stark nach Fake Reviews aus.
#nicht verified mehr extrem negative reviews

#NEUES FEATURE : Model lernt bei welchen companies verified besonders wichtig ist
company_verified_effect = filtered["diff"].to_dict()

#HIER GIBT ES NOCH INKONSISTENZEN MIT DEN NAMEN!!!! deswegen ist train["verified_company_effect"] alles NaN, muss noch behoben werden!!
train["verified_company_effect"] = train["company"].map(company_verified_effect)
print(train["company"].unique()[:20])
print(list(company_verified_effect.keys())[:20])
print(train["verified_company_effect"].head(10))
#Hier kann schön weiter untersucht werden, z.B. kombinieren mit has_response

#Untersuche zunächst Zusammenhang has_response und rating:

#Untersuche Location und Rating
#Hier muss erstmal noch aufgearbeitet werden, brauche die Länder mit genügend reviews 
print("Untersuche Location und Rating:",pd.crosstab(
    train["location"],
    train["rating"],
    normalize="index"
).head(20))

#top 15 locations
top_countries = train["location"].value_counts().nlargest(15).index

country_ct = pd.crosstab(
    train["location"],
    train["rating"],
    normalize="index"
).loc[top_countries]

print(country_ct)

location_mean = train.groupby("location")["rating"].mean().sort_values()
print("location_mean:",location_mean)

location_counts = train["location"].value_counts()
print(location_counts.describe())
valid = train["country"].value_counts()
valid = valid[valid >= 30].index
filtered = train[train["company"].isin(valid)]

#Kombination von features
#Unfertig, erstmal weiter zu kombinierten features
#Welche sinnvoll zu kombinieren?
#1)



#Feature engineering auf den review_texts
train["exclamations"] = train["review_text"].str.count("!")
train["questions"] = train["review_text"].str.count("\\?")
train["has_not"] = train["review"].str.contains("not|no|never", case=False)

#train["length_bin"] = pd.qcut(train["review_length"], 4)

#train.groupby("length_bin")["rating"].mean()
#Normalization between 0 and 1 : if distribution of variable does not follow a normal distribution
# formula: (xi-min(x))/max(x)-min(x)

#Standardization : If distribution of variable follows a normal distribution
#formula: (xi-mean(x))/std(x)

#Outliers: formula: (xi-median(x))/(q3(x)-q1(x))

#Correlation in the dataset
#Idee ist Variablen zu finden die miteinander korreliert sind und daher überflüsssig, dies vereinfacht das Modell
#Spaltenanzahl überprüfen, 0 te Zeile ist target Rating, wir haben 8 features, also 1 bis 9(exclusive),
#  lasse es erstmal so auch wenn da unsinnige features bei sind, ist erstmal rohversion kommentare etc sind natürlich nicht kategorisch
#detects only linear relationships!! mutual information is more useful
cor = train.iloc[:, 1:9].corr()

plt.figure(figsize = (10, 6))
sns.heatmap(cor, annot = True)

#We need to set an absolute value, for example 0.5, as the threshold for variable selection. 
# If we find that the predictor variables are correlated, we can drop the variable whose correlation coefficient value
#  is lower than that of the target variable. We can also calculate multiple correlation coefficients to check if more than two variables are correlated.
#  This phenomenon is known as multicollinearity.

X=train.drop(columns="rating")
y=train["rating"]
#Select the three best features according to the chi square test
from sklearn.feature_selection import SelectKBest,chi2
selector=SelectKBest(score_func=chi2,k=3)
X_kbest_features=selector.fit_transform(X,y)
print(X.shape[1])
print(X_kbest_features.shape[1])
selector.get_feature_names_out()
#We have successfully retained the top 3 variables according to our analysis with the chi-square test.


#Mutual information is much like correlation in that it measures a relationship between two quantities.
#  The advantage of mutual information is that it can detect any type of relationship, whereas correlation only detects linear relationships.
#  It calculates the reduction in entropy resulting from the transformation of a dataset and can be used for feature selection 
# by evaluating the information gain of each variable in the context of the target variable.
from sklearn.feature_selection import mutual_info_classif

importances = mutual_info_classif(X, y)
feat_importances = pd.Series(importances, train.columns[0:len(train.columns)-1])
feat_importances.plot(kind = 'barh')
plt.show()

#To keep in mind:

#Anova test

#Variance threshold test

#The Fisher score is one of the most commonly used supervised feature selection methods. The algorithm we are going to use returns the ranks
#  of the variables based on the Fisher score in descending order. We can then select the variables based on the case

#Mean Absolute Difference (MAD)

#The Kruskal-Wallis test is a non-parametric method used to determine if there are significant differences between the means of three or more groups in a dataset.
#  It is used as an alternative to ANOVA when the data does not meet the assumptions for ANOVA, particularly when the data is not normally distributed
#  or when heteroscedasticity is present.

#WRAPPER METHODs brute force!!! achieve better results than filtering methods! need to test that

#IDEAS FOR DATA ENRICHMENT:

#