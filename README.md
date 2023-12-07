# Advanced E-Commerce Recommendation Engine

## Introduction
This project, developed for the Database System Principles (DSP) course, focuses on enhancing e-commerce recommendations using database optimization and collaborative filtering. It aims to provide personalized product suggestions based on user ratings and behavior.

## Course Details
- **Course**: Database System Principles (DSP), CS257
- **Institution**: San Jose State University (SJSU), San Jose, CA
- **Instructor**: Ramin Moazeni
- **Term**: Fall 2023

## Implementation
This recommendation system is implemented using Python, with key libraries including Streamlit for the web interface, Matplotlib and Seaborn for data visualization, and Pandas for data processing. Scikit-learn is used for machine learning models, and the Surprise library for collaborative filtering techniques.

## Installation
To install the required libraries, run:
```bash
pip install streamlit matplotlib seaborn pandas scikit-learn scipy surprise
```


## Usage
To run the web application:

```bash
streamlit run [filename].py
```


## Features
- Data visualization of user ratings and product popularity.
- Implementation of popularity-based and collaborative filtering models.
- User interface for interacting with the recommendation engine.

## Dataset
The project uses an e-commerce dataset of user ratings, structured with columns: userId, productId, Rating, and timestamp.

## Methodology
- **Popularity-Based Recommender Model**: Recommends products based on their overall popularity.
- **Collaborative Filtering**: Utilizes both KNN with means and SVD algorithms to generate personalized recommendations.

## Results
The recommendation engine successfully generates user-specific product suggestions. The system's efficiency and accuracy were measured using RMSE (Root Mean Square Error).

## Limitations and Future Work
Future improvements include integrating real-time user data and exploring more advanced machine learning techniques like neural networks.

## References
1. S. Shekhar, et al., "Efficient join-index-based spatial-join processing: a clustering approach," in IEEE Transactions on Knowledge and Data Engineering, 2002.
2. Zarzour, Hafed, et al., "A new collaborative filtering recommendation algorithm," in 2018 9th International Conference on ICICS, IEEE, 2018.
3. Xiaojun, Liu, "An improved clustering-based collaborative filtering recommendation algorithm," Cluster computing, 2017.
4. Liu, Hongjiao, "Implementation and Effectiveness Evaluation of Four Common Algorithms of Recommendation Systems," in 2022 International Conference on CBASE, IEEE, 2022.
5. Zhao, Xuesong, "A study on e-commerce recommender system based on big data," in 2019 IEEE 4th ICCCCBDA, IEEE, 2019.
6. Li, Xiangpo, "Research on the application of collaborative filtering algorithm in mobile e-commerce recommendation system," in 2021 IEEE Asia-Pacific Conference on IPEC, IEEE, 2021.
7. Zhang, Weiwei, et al., "Personalized and Stable Recommendation Algorithm of E-commerce Commodity Information Based on Collaborative Filtering," in 2022 Global Reliability and PHM-Yantai, IEEE, 2022.
8. Gosh, Subasish, et al., "Recommendation system for e-commerce using alternating least squares (ALS) on apache spark," in International Conference on Intelligent Computing & Optimization, Springer International Publishing, 2020.
9. Hug, Nicolas, "Surprise: A Python library for recommender systems," Journal of Open Source Software, 2020.
10. Zhao, Zhi-Dan, and Ming-Sheng Shang, "User-based collaborative-filtering recommendation algorithms on hadoop," in 2010 Third International Conference on Knowledge Discovery and Data Mining, IEEE, 2010.

## Contributors
- Aniruddha Prabhash Chakravarty, aniruddhaprabhash.chakravarty@sjsu.edu
- Sai Mounika Peteti, saimounika.peteti@sjsu.edu
- Sumukh Naveen Aradhya, sumukhnaveen.aradhya@sjsu.edu

## Contact
For more information, contact any of the contributors via their respective SJSU email addresses.


