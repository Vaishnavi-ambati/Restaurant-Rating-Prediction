# Restaurant-Rating-Prediction



#### -- Project Status: Completed

## Project Intro
Restaurant Rating has become the most commonly used parameter for judging a restaurant for any individual. A lot of research has been done on different restaurants and the quality of food it serves. Rating of a restaurant depends on factors like reviews, area situated, average cost for two people, votes, cuisines and the type of restaurant. The objective of this project is to build a webapp that predicts the rating of a restaurant. 


### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling
* Hyperparameter Tuning
* Flask integration

##### Technologies Used:
Type | Used
--- | --- 
Language | Python 3.7
IDE |	PyCharm
Database	|MySQL
Frontend	| HTML5, CSS3, Bootstrap
Integration	| Flask
Deployment |	Google Cloud Platform


## Project Description

This dataset has been obtained by scraping the TA website for information about restaurants.

Feature	| Description
--- | ---
Name |	name of the restaurant
City  |	city location of the restaurant
Cuisine Style |	cuisine style(s) of the restaurant, in a Python list object (94 046non-null)
Rating |	rate of the restaurant on a scale from 1 to 5, as a float object (115 658 non-null)    (Target Column)
Ranking	| rank of the restaurant among the total number of restaurants in the city as a float object (115 645 non-null)
Price Range	| price range of the restaurant among 3 categories , as a categorical type (77 555 non-null)
Number of Reviews |	number of reviews that customers have let to the restaurant, as a float object (108 020 non-null)
Reviews |	2 reviews that are displayed on the restaurants scrolling page of the city, as a list of list object where the first list contains the 2 reviews, and the second le dates when these reviews were written (115 673 non-null)
URL_TA	| part of the URL of the detailed restaurant page that comes after 'www.tripadvisor.com' as a string object (124 995 non-null)
ID_TA	| identification of the restaurant in the TA database constructed a one letter and a number (124 995 non-null


* Initially after exploring the dataset a bit, I was a bit confused about the approach for modelling. I was not sure whether to build a regression model or classification model. After checking the unique values in the target feature i.e., 'ratings' feature(shown in the image below), I have decided to build a classification model because a regression model would give predictions in the range [1,5] (it might be 1.2,3.2,4.7, etc..) which would lead to a poor accuracy since the dataset has only 9 unique values for rating. 

![download](https://user-images.githubusercontent.com/50202237/89147835-8a83ba00-d575-11ea-9a0a-326e4e3308e1.png)

* Each row in review column had a list of two reviews. I have created two new features from review column which contains the sentiments of the two reviews. This increased the accuracy of the model by a bit.


## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept []  within this repo.
3. Data processing/transformation scripts are being kept [here](https://github.com/Vaishnavi-ambati/Restaurant-Rating-Prediction/tree/master/Modules).
4. After cloning the repo, open an editor of your choice and create a new environment.(for help see this [tutorial](https://realpython.com/lessons/creating-virtual-environment/))
5. Install the required modules in the environment using the requirements.txt file. (for help see this [tutorial](https://note.nkmk.me/en/python-pip-install-requirements/))
6. After installing the required modules. You are all set! Run the 'main.py' file and the app will be hosted in your local server.

Note: If you have any issues with project or with the setup. You can always contact me or raise an issue. :)
