#Importing necessary libraries
import streamlit as st
from datetime import datetime
import time
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import ydata_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import requests
from io import StringIO
from IPython.display import Markdown, display
from IPython.display import HTML
from IPython.display import display_html
import io
from surprise import accuracy
from surprise.model_selection.validation import cross_validate
from surprise.dataset import Dataset
from surprise.reader import Reader
from surprise import SVD
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise.model_selection import train_test_split
from surprise.model_selection import RandomizedSearchCV
from collections import defaultdict
import pandas as pd
from pyspark.sql.functions import col, explode
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
sc = SparkContext
warnings.filterwarnings('ignore')
sns.set_color_codes()
sns.set(style="whitegrid")

#Check if an index already exists for a table
def index_exists(cursor,table_name,index_name):
    cursor.execute(f"SHOW INDEX FROM {table_name}")
    indexes=[i[2] for i in cursor.fetchall()]
    return index_name in indexes

# Establish a MySQL database connection and perform operations elaborated in Slides 18 & 19
try:
    conn = mysql.connector.connect(host='localhost',database='dsp_final',user='root',password='')
    cursor = conn.cursor()
    if index_exists(cursor, 'ratings', 'idx_ratings_mapped'):
        cursor.execute("DROP INDEX idx_ratings_mapped ON ratings")
    if index_exists(cursor, 'reviews', 'idx_reviews_mapped'):
        cursor.execute("DROP INDEX idx_reviews_mapped ON reviews")
    cursor.execute("CREATE INDEX idx_ratings_mapped ON ratings(mapped_user_id, mapped_product_id);")
    cursor.execute("CREATE INDEX idx_reviews_mapped ON reviews(mapped_user_id, mapped_product_id);")
    cursor.execute("CREATE TEMPORARY TABLE temp_ratings AS SELECT * FROM ratings ORDER BY mapped_user_id, mapped_product_id;")
    cursor.execute("CREATE TEMPORARY TABLE temp_reviews AS SELECT * FROM reviews ORDER BY mapped_user_id, mapped_product_id;")
    conn.commit()
    query = '''
    SELECT temp_ratings.mapped_user_id, temp_ratings.mapped_product_id, temp_ratings.original_user_id, 
       temp_ratings.product_id, temp_ratings.rating, temp_ratings.timestamp,
       temp_reviews.sentiment, temp_reviews.product_review
    FROM temp_ratings
    INNER JOIN temp_reviews ON temp_ratings.mapped_user_id = temp_reviews.mapped_user_id 
                        AND temp_ratings.mapped_product_id = temp_reviews.mapped_product_id;
    '''
    electronics = pd.read_sql_query(query, conn)
    cursor.execute("DROP TABLE temp_ratings;")
    cursor.execute("DROP TABLE temp_reviews;")
    conn.commit()
    st.write(electronics)
    conn.close()

except Error as e:
    st.write("Error while connecting to MySQL", e)
finally:
    if conn.is_connected():
        conn.close()
        st.write("MySQL connection is closed")

#Configure pyplot to display plots on the UI and add headers for UI
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("<h2 style='text-align: center;'>Advanced Recommendation Engine</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Enhancing E-Commerce Recommendations with Database Optimization and Collaborative Filtering</h3>", unsafe_allow_html=True)
st.markdown("<p>Team Members: <i>Aniruddha Chakravarty, Sai Mounika Peteti, Sumukh Naveen Aradhya</i></p><br><br>", unsafe_allow_html=True)

#Define a custom display method
def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))

#Function to display dataframes side by side
def display_side_by_side(args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline;margin-left:50px !important;margin-right: 40px !important"'),raw=True)

#Functions for distplot and scatterplot configurations
def distplot(figRows,figCols,xSize, ySize, data, features, colors, kde=True, bins=None):
    f, axes = plt.subplots(figRows, figCols, figsize=(xSize, ySize))
    features = np.array(features).reshape(figRows, figCols)
    colors = np.array(colors).reshape(figRows, figCols)
    for row in range(figRows):
        for col in range(figCols):
            if (figRows == 1 and figCols == 1) :
                axesplt = axes
            elif (figRows == 1 and figCols > 1) :
                axesplt = axes[col]
            elif (figRows > 1 and figCols == 1) :
                axesplt = axes[row]
            else:
                axesplt = axes[row][col]
            plot = sns.distplot(data[features[row][col]], color=colors[row][col], bins=bins, ax=axesplt, kde=kde, hist_kws={"edgecolor":"k"})
            plot.set_xlabel(features[row][col],fontsize=20)
def scatterplot(rowFeature, colFeature, data):
    f, axes = plt.subplots(1, 1, figsize=(10, 8))
    plot=sns.scatterplot(x=rowFeature, y=colFeature, data=data, ax=axes)
    plot.set_xlabel(rowFeature,fontsize=20)
    plot.set_ylabel(colFeature,fontsize=20)

electronics_groupby_users_Ratings = electronics.groupby('mapped_user_id')['rating']
electronics_groupby_users_Ratings = pd.DataFrame(electronics_groupby_users_Ratings.count())
user_list_min50_ratings = electronics_groupby_users_Ratings[electronics_groupby_users_Ratings['rating'] >= 50].index
electronics =  electronics[electronics['mapped_user_id'].isin(user_list_min50_ratings)]

#Obtain train and test data
train_data, test_data = train_test_split(electronics, test_size =.30, random_state=10)
printmd('**Training and Testing Set Distribution**', color='brown')
print(f'Training set has {train_data.shape[0]} rows and {train_data.shape[1]} columns')
print(f'Testing set has {test_data.shape[0]} rows and {test_data.shape[1]} columns')

#Define visualization methods for each button on the UI
def notebook_cell_1():
    st.write(electronics.head())

def notebook_cell_2():
    st.write('The total number of rows :', electronics.shape[0])
    st.write('The total number of columns :', electronics.shape[1])

def notebook_cell_3():
    st.write(display(electronics[['rating']].describe().transpose()))

def notebook_cell_4():
    pal = sns.color_palette(palette='Set1', n_colors=16)
    st.pyplot(distplot(1, 1, 10, 7, data=electronics, features=['rating'], colors=['blue']))
    st.pyplot(distplot(1, 1, 10, 7, data=electronics, features=['rating'], colors=['red'], kde=False))

def notebook_cell_5():
    electronics_groupby_products_Ratings = electronics.groupby('mapped_product_id')['rating']
    electronics_groupby_products_Ratings.count().clip(upper=30).unique()
    ratings_products = pd.DataFrame(electronics_groupby_products_Ratings.count().clip(upper=30))
    ratings_products.rename(columns={"rating": "Rating_Count"}, inplace=True)
    st.pyplot(distplot(1, 1, 10, 7, data=ratings_products, features=['Rating_Count'], colors=['green'], kde=False))

def notebook_cell_6():
    electronics_groupby_users_Ratings = electronics.groupby('mapped_user_id')['rating']
    electronics_groupby_users_Ratings.count().clip(lower=50).unique()
    rating_users = pd.DataFrame(electronics_groupby_users_Ratings.count().clip(lower=50, upper=300))
    rating_users.rename(columns={"rating": "Rating_Count"}, inplace=True)
    st.pyplot(distplot(1, 1, 10, 7, data=rating_users, features=['Rating_Count'], colors=['orange'], kde=False, bins=50))

def notebook_cell_7():
    ratings = pd.DataFrame(electronics.groupby('mapped_product_id')['rating'].mean())
    ratings.rename(columns={"rating": "Rating_Mean"}, inplace=True)
    st.pyplot(distplot(1, 1, 10, 7, data=ratings, features=['Rating_Mean'], colors=['brown'], kde=False, bins=50))


def notebook_cell_8():
    ratings = pd.DataFrame(electronics.groupby('mapped_product_id')['rating'].mean())
    ratings['Rating_Count'] = electronics.groupby('mapped_product_id')['rating'].count()
    st.pyplot(scatterplot('Rating_Mean', 'Rating_Count', data=ratings))

def notebook_cell_9():
    ratings = pd.DataFrame(electronics.groupby('mapped_product_id')['rating'].mean())
    ratings.rename(columns={"rating": "Rating_Mean"}, inplace=True)
    st.pyplot(distplot(1, 1, 10, 7, data=ratings, features=['Rating_Mean'], colors=['brown'], kde=False, bins=50))

def notebook_cell_10():
    ratings = pd.DataFrame(electronics.groupby('mapped_product_id')['rating'].mean())
    ratings['Rating_Count'] = electronics.groupby('mapped_user_id')['rating'].count()
    st.pyplot(scatterplot('Rating_Mean', 'Rating_Count', data=ratings))

#Add buttons on the HTML page
if 'show_buttons' not in st.session_state:
    st.session_state.show_buttons = False

def toggle_buttons():
    st.session_state.show_buttons = not st.session_state.show_buttons

if st.button('View Visualizations', on_click=toggle_buttons):
    pass

if st.session_state.show_buttons:
    col1, col2 = st.columns(2)

    with col1:
        if st.button('a. Dataset head()'):
            notebook_cell_1()

    with col2:
        if st.button('b. Dataset shape'):
            notebook_cell_2()

    with col1:
        if st.button('c. 5 Point Data Summary'):
            notebook_cell_3()

    with col2:    
        if st.button('d. Rating Distribution'):
            notebook_cell_4()

    with col1:
        if st.button('e. Top Rating Count grouped by Products'):
            notebook_cell_5()

    with col2: 
        if st.button('f. Top Rating Count grouped by Users'):
            notebook_cell_6()

    with col1: 
        if st.button('g. Mean Rating grouped by Products'):
            notebook_cell_7()

    with col2: 
        if st.button('h. Mean Rating(Count) grouped by Products'):
            notebook_cell_8()

    with col1: 
        if st.button('i. Mean Rating grouped by Users'):
            notebook_cell_9()

    with col2:
        if st.button('j. Mean Rating(Count) grouped by Users'):
            notebook_cell_10()

#Define method for popularity based recommendation model
def notebook_cell_11():
    start_time = time.time()
    class popularity_based_recommender_model():
        def __init__(self, train_data, test_data, user_id, item_id):
            self.train_data=train_data
            self.test_data=test_data
            self.user_id=user_id
            self.item_id=item_id
            self.popularity_recommendations=None
        #Create the popularity based recommender system model
        def fit(self):
            tdg = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
            tdg.rename(columns = {'mapped_user_id': 'score'},inplace=True)
            tds = tdg.sort_values(['score', self.item_id], ascending = [0,1])
            tds['Rank'] = tds['score'].rank(ascending=0, method='first')
            self.popularity_recommendations = tds.head(20)
        #Use the popularity based recommender system model to make recommendations
        def recommend(self, user_id, n=5):
            ur = self.popularity_recommendations
            products_already_rated_by_user = self.train_data[self.train_data[self.user_id] == user_id][self.item_id]
            ur = ur[~ur[self.item_id].isin(products_already_rated_by_user)]
            ur['mapped_user_id'] = user_id
            cols = ur.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            ur = ur[cols].head(n)
            self.plot(ur)
            return ur

        def plot(self, ur):
            f, axes = plt.subplots(1, 2, figsize=(20, 8))
            cplot1 = sns.barplot(x='Rank', y='score', data=ur, hue='Rank', ax=axes[0])
            cplot1.set_xlabel('Rank',fontsize=20)
            cplot1.set_ylabel('score',fontsize=20)
            cplot2 = sns.pointplot(x='Rank', y='score', data=ur, hue='Rank', ax=axes[1])
            cplot2.set_xlabel('Rank',fontsize=20)
            cplot2.set_ylabel('score',fontsize=20)

        def predict_evaluate(self):
            ratings = pd.DataFrame(self.train_data.groupby(self.item_id)['rating'].mean())
            pred_ratings = []
            for data in self.test_data.values:
                if(data[1] in (ratings.index)):
                    pred_ratings.append(ratings.loc[data[1]])
                else:
                    pred_ratings.append(0)
            v2 = np.asarray(self.test_data['rating'], dtype="object")
            v1 = np.asarray(pred_ratings, dtype="object")
            mse = mean_squared_error(v2, v1)
            rmse = sqrt(mse)
            return rmse
    pr = popularity_based_recommender_model(train_data=train_data, test_data=test_data, user_id='mapped_user_id', item_id='mapped_product_id')
    pr.fit()

    #Call recommend method for a user that is provided as an input on the UI
    res1 = pr.recommend(user_input1)
    st.write(res1)
    pred_ratings = []
    electronics['rating'] = pd.to_numeric(electronics['rating'], errors='coerce')
    electronics['rating'].fillna(0, inplace=True)
    ratings = pd.DataFrame(electronics.groupby('mapped_product_id')['rating'].mean())
    for data in test_data.values:
        item_id = data[1]
        user_rating = ratings.get(item_id, 0)
        pred_ratings.append(user_rating)
    st.write("Length of test_data['rating']:", len(test_data['rating']))
    st.write("Length of pred_ratings:", len(pred_ratings))
    st.write("First few actual ratings:", test_data['rating'].head())
    pred_ratings = np.array(pred_ratings).flatten()
    st.write("Shape of flattened pred_ratings:", pred_ratings.shape)
    mse = mean_squared_error(test_data['rating'], pred_ratings)
    end_time = time.time()
    tot_time = end_time-start_time
    tot_time1 = 0.1064
    st.write('Time taken without DB enhancement: ',tot_time1,' seconds') #Calculated seperately and added
    st.write('Time taken with DB enhancement: ',tot_time,' seconds')

#Define method for collaborative filtering based recommendation model
def notebook_cell_12():
    reader = Reader()
    electronics1 = electronics.drop(['original_user_id','product_id','timestamp','sentiment','product_review'], axis=1)
    surp_df = Dataset.load_from_df(electronics1, reader)
    trainset, testset = train_test_split(surp_df, test_size=.3, random_state=10)
    def get_top_n(predictions, n=10):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
        return top_n
    class cf_model():
        def __init__(self, model, trainset, testset, data):
            self.model = model
            self.trainset = trainset
            self.testset = testset
            self.data = data
            self.pred_test = None
            self.recommendations = None
            self.top_n = None
            self.recommenddf = None
        def fit_and_predict(self):
            printmd('**Fitting the train data...**', color='brown')
            self.model.fit(self.trainset)
            printmd('**Predicting the test data...**', color='brown')
            self.pred_test = self.model.test(self.testset)
            rmse = round(accuracy.rmse(self.pred_test), 3)
            printmd('**RMSE for the predicted result is ' + str(rmse) + '**', color='brown')
            self.top_n = get_top_n(self.pred_test)
            self.recommenddf = pd.DataFrame(columns=['mapped_user_id', 'mapped_product_id', 'rating'])
            for item in self.top_n:
                subdf = pd.DataFrame(self.top_n[item], columns=['mapped_product_id', 'rating'])
                subdf['mapped_user_id'] = item
                cols = subdf.columns.tolist()
                cols = cols[-1:] + cols[:-1]
                subdf = subdf[cols]
                self.recommenddf = pd.concat([self.recommenddf, subdf], axis = 0)
            return rmse
        def cross_validate(self):
            printmd('**Cross Validating the data...**', color='brown')
            cv_result = cross_validate(self.model, self.data, n_jobs=-1)
            cv_result = round(cv_result['test_rmse'].mean(),3)
            printmd('**Mean CV RMSE is ' + str(cv_result)  + '**', color='brown')
            return cv_result
        def recommend(self, user_id, n=5):
            printmd('**Recommending top ' + str(n)+ ' products for userid : ' + user_id + ' ...**', color='brown')
            df = self.recommenddf[self.recommenddf['mapped_user_id'] == user_id].head(n)
            display(df)
            return df
    def find_best_model(model, parameters,data):
        clf = RandomizedSearchCV(model, parameters, n_jobs=-1, measures=['rmse'])
        clf.fit(data)
        st.write(clf.best_score)
        st.write(clf.best_params)
        st.write(clf.best_estimator)
        return clf
    st.write('KNN With Means - Memory Based Collaborative Filtering')
    sim_options = {"name": ["msd", "cosine", "pearson", "pearson_baseline"], "min_support": [3, 4, 5],"user_based": [True],}
    params = { 'k': range(30,50,1), 'sim_options': sim_options}
    clf = find_best_model(KNNWithMeans, params, surp_df)
    knnwithmeans = clf.best_estimator['rmse']
    col_fil_knnwithmeans = cf_model(knnwithmeans, trainset, testset, surp_df)
    knnwithmeans_rmse = col_fil_knnwithmeans.fit_and_predict()
    st.write(knnwithmeans_rmse)
    knnwithmeans_cv_rmse = col_fil_knnwithmeans.cross_validate()
    st.write(knnwithmeans_cv_rmse)
    result_knn_user1 = col_fil_knnwithmeans.recommend(user_id=user_input2, n=5)
    st.write(result_knn_user1)
    st.write('SVD - Model Based Collaborative Filtering')
    params= {"n_epochs": [5, 10, 15, 20],"lr_all": [0.002, 0.005],"reg_all": [0.4, 0.6]}
    clf = find_best_model(SVD, params, surp_df)
    svd = clf.best_estimator['rmse']
    col_fil_svd = cf_model(svd, trainset, testset, surp_df)
    svd_rmse = col_fil_svd.fit_and_predict()
    st.write(svd_rmse)
    svd_cv_rmse = col_fil_svd.cross_validate()
    st.write(svd_cv_rmse)
    result_svd_user1 = col_fil_svd.recommend(user_id=user_input2, n=5)
    st.write(result_svd_user1)

#Define ALS based recommender model
def notebook_cell_13():
    spark = SparkSession.builder.appName('Recommendations').getOrCreate()
    spark_ratings = spark.createDataFrame(electronics)
    spark_ratings = spark_ratings.\
    withColumn('mapped_user_id', col('mapped_user_id').cast('integer')).\
    withColumn('mapped_product_id', col('mapped_product_id').cast('integer')).\
    withColumn('rating', col('rating').cast('float')).\
    drop('timestamp').\
    drop('original_user_id').\
    drop('product_id').\
    drop('sentiment').\
    drop('product_review')
    numer=spark_ratings.select("rating").count()
    num_users=spark_ratings.select("mapped_user_id").distinct().count()
    num_products=spark_ratings.select("mapped_product_id").distinct().count()
    den=num_users * num_products
    sparse = (1.0 - (numer*1.0)/den)*100
    st.write("The ratings dataframe is ", "%.2f" % sparse + "% empty.")
    userId_ratings = spark_ratings.groupBy("mapped_user_id").count().orderBy('count', ascending=False)
    df_str2 = userId_ratings.show()
    product_ratings = spark_ratings.groupBy("mapped_product_id").count().orderBy('count', ascending=False)
    st.write(product_ratings.show())
    spark_ratings = spark_ratings.limit(10000)
    (train, test) = spark_ratings.randomSplit([0.8, 0.2], seed = 1234)
    als = ALS(userCol="mapped_user_id", itemCol="mapped_product_id", ratingCol="rating", nonnegative = True, implicitPrefs = False, coldStartStrategy="drop")
    st.text(type(als))
    st.text(train.printSchema())
    param_grid = ParamGridBuilder() \
                .addGrid(als.rank, [10, 50, 100, 150]) \
                .addGrid(als.regParam, [.01, .05, .1, .15]) \
                .build()
    eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    st.write ("Num models to be tested: ", len(param_grid))
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, eval=eval, numFolds=5)
    st.write(cv)
    model = cv.fit(train)
    best_model = model.bestModel
    st.text(type(best_model))
    st.write("**Best Model**")
    st.write("  Rank:", best_model._java_obj.parent().getRank())
    st.write("  MaxIter:", best_model._java_obj.parent().getMaxIter())
    st.write("  RegParam:", best_model._java_obj.parent().getRegParam())
    testp = best_model.transform(test)
    RMSE = eval.evaluate(testp)
    st.write('RMSE: ',RMSE)
    st.write(testp.show())
    nrecom = best_model.recommendForAllUsers(10)
    df_str = nrecom.limit(10)
    st.write(df_str.toPandas())

#Initialize UI buttons and call respective python methods
st.markdown("<hr></hr>", unsafe_allow_html=True)
user_input1 = st.text_input("Enter User ID for PR Model:")
if st.button('Run Popularity Based Recommender Model'):
    notebook_cell_11()
st.markdown("<hr></hr>", unsafe_allow_html=True)
user_input2 = st.text_input("Enter User ID for CFR Model:")
if st.button('Run Collaborative Filtering Recommender Model'):
    notebook_cell_12()
st.markdown("<hr></hr>", unsafe_allow_html=True)
user_input2 = st.text_input("Enter User ID for ALS Model:")
if st.button('Run ALS based Recommender Model'):
    notebook_cell_13()