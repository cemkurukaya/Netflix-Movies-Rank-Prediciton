# Netflix Movies Rank Preediction

**Project Topic:** Prediction of Movie Ratings with Linear Regression
and XGBoost

**Introduction**

The program reads in a dataset from a file called "Netflix_movies.csv"
and tries to predict the ratings using director, country and genre. And
it does this with two different approaches. These are liner regression
and xgboost.

**Data Analysis and Cleaning**

Almost the first 60 lines consist of data cleaning and analysis. First,
I gave names to our unnamed columns. Then I dropped the "enter year in
netflix" and "actors" columns from my dataframe using the drop()
parameter. Then I found out how many columns and rows I have in a
dataframe and learned the types of the data in these columns.

I checked the missing value with the isnull() command and there was no
missing value. If there were, I would take the average of the other data
and place them in the blanks.

I used the command duplicated().any() to find out if there was any
duplicate yield behind it. There were no duplicated values either.

Then I collected my statistics in my dataset with the df.describe()
command. (such as minimum, maximum, average values)

Some columns contained more than one data and were often redundant data.
For example, the genre column could contain anime, comedy, and drama at
the same time. To prevent this, I added a for loop that collects the
first data in those columns and places it in those columns.

Finally, I added a for loop to sort different genres into different
columns. Thus, I ensured that there was only one element of each type in
the list I prepared. The difficulty I faced in this step was: As I
mentioned before, some movies had more than one genre. These genres
looked like this: \["comedy"," drama"\]. Since there was a space at the
beginning of the elements after the first element of the list, it
perceived them as two different genres. I got around this problem by
using the strip() command with a for loop. Moving these genres to
separate columns will provide a lower mean absolute error than putting
them together in a single column.

**Label Encoding**

Label Encoding refers to converting the labels into a numeric form so as
to convert them into the machine-readable form. Machine learning
algorithms can then decide in a better way how those labels must be
operated. It is an important pre-processing step for the structured
dataset in supervised learning.

![image](https://user-images.githubusercontent.com/57721481/216833990-d68308f6-cae5-4ecc-b834-013c1d2f2875.png) ![image](https://user-images.githubusercontent.com/57721481/216834004-25207ded-8ea3-4aa7-91bd-ebf2efcf84cf.png)

The same values represent the same genre. For example, all 4s are all
the comedy genre.

I moved the genres I mentioned above to separate columns using the
get_dummies() method. In contrast to the scenario where I didn't do
this, separating genres resulted in a 6.12% reduction in mean absolute
error.

**MinMaxScaler**

MinMaxscaler is a type of scaler that scales the minimum and maximum
values to be 0 and 1 respectively. ML algorithm works better when
features are relatively on a similar scale and close to Normal
Distribution. So our MinMaxScaling application before the train test
split will give us much more accurate results.

**Train Test Split**

The train_test_split() method is used to split our data into train and
test sets. The dataset is divided into training and test sets. While 70%
of the data was used for training, the remaining 30% was used for
testing.

**Linear Regression**

Linear regression is a data analysis technique that estimates the value
of unknown data using another relevant and known data value.
Mathematically models the unknown or dependent variable and the known or
independent variable as a linear equation. In machine learning, computer
programs called algorithms analyze large sets of data and work backwards
from that data to calculate the linear regression equation.

**Xgboost Regression**

Gradient boosting refers to a class of ensemble machine learning
algorithms that can be used for classification or regression predictive
modeling problems.

Models are fit using any arbitrary differentiable loss function and
gradient descent optimization algorithm. This gives the technique its
name, “gradient boosting,” as the loss gradient is minimized as the
model is fit, much like a neural network

**Comparison**

After applying the linear regression and xgboost, we can now compare
them. I made this comparison by looking at mean absolute error, r2
score, and execution times.

![image](https://user-images.githubusercontent.com/57721481/216834039-1116f77a-596d-458b-9329-73e14989cf7e.png)

As seen in the figure, the mean absolute error (mae) of Linear Regression
is lower. It also runs faster, and has a higher r2 score. Considering
all of this, we can say that Liner Regression is much better for this
project.
