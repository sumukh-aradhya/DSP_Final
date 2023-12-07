Advanced E-Commerce Recommendation Engine
Introduction
This project, developed for the Database System Principles (DSP) course, focuses on enhancing e-commerce recommendations using database optimization and collaborative filtering. It aims to provide personalized product suggestions based on user ratings and behavior.

Course Details
Course: Database System Principles (DSP), CS257
Institution: San Jose State University (SJSU), San Jose, CA
Instructor: Ramin Moazeni
Term: Fall 2023
Implementation
This recommendation system is implemented using Python, with key libraries including Streamlit for the web interface, Matplotlib and Seaborn for data visualization, and Pandas for data processing. Scikit-learn is used for machine learning models, and Surprise library for collaborative filtering techniques.

Installation
To install the required libraries, run:

bash
Copy code
pip install streamlit matplotlib seaborn pandas scikit-learn scipy surprise
Usage
To run the web application:

bash
Copy code
streamlit run [filename].py
Replace [filename] with the name of the Python script.

Features
Data visualization of user ratings and product popularity.
Implementation of popularity-based and collaborative filtering models.
User interface for interacting with the recommendation engine.
Dataset
The project uses an e-commerce dataset of user ratings, structured with columns: userId, productId, Rating, and timestamp.

Methodology
Popularity-Based Recommender Model: Recommends products based on their overall popularity.
Collaborative Filtering: Utilizes both KNN with means and SVD algorithms to generate personalized recommendations.
Results
The recommendation engine successfully generates user-specific product suggestions. The system's efficiency and accuracy were measured using RMSE (Root Mean Square Error).

Limitations and Future Work
Future improvements include integrating real-time user data and exploring more advanced machine learning techniques like neural networks.

References
List any academic references or papers relevant to the project.

Contributors
Aniruddha Prabhash Chakravarty, aniruddhaprabhash.chakravarty@sjsu.edu
Sai Mounika Peteti, saimounika.peteti@sjsu.edu
Sumukh Naveen Aradhya, sumukhnaveen.aradhya@sjsu.edu
Contact
For more information, contact any of the contributors via their respective SJSU email addresses.

