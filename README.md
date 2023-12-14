# PredictingRecipeRatings
This project builds a random forest regressor that predicts the rating for recipes without a rating.

Our exploratory data analysis can be found <a href="https://aaron-m-r.github.io/PredictingRecipeRatings/">here</a>

## Framing the Problem
During our exploratory data analysis, we discovered many recipes were missing a rating. Although it appears in the original dataset that some reviews have a score of zero, we reasoned that this most likely instead indicates a lack of a rating. Instead of imputing these missing values using the mean rating or a probabilistic model, we would like to use machine learning to predict these ratings based on other available features.

In our data analysis of recipes and their ratings, we found few to no meaningful relationships between ratings and the quantitative variables related to the recipes (number of steps, number of ingredients, nutrition facts etc.). Although we found some subtle relationships, which we will continue to take advantage of, we know that we must continue our investigation to see if part of our data decides ratings. Is recipe quality, truly the only deciding factor? If so, are there other variables, quantitative or quanitfiable, that are correlated with quality, which may allow us to predict quality? **In this project, we would like to predict the ratings for reviews that are missing a rating.** To do this, we'll split our final model's data into training, validation and split, but train the final model on all available data with ratings. Finally, we'll predict the ratings of reviews with no rating (0, which we turn into null value).

Since we are going to predict a quantitative continuous variable such as rating, we will perform regression on our data. In assessing the performance of our model, we'll use the root mean squared error as our performance statistic. This is because we are regressing, not classifying, we value precision and recall equally, and we want equal values for each of these two metrics. Also, a metric such as accuracy is unlikely to be representative of our performance since it's extremely difficult to predict an exact number with multiple decimal points. Root mean squared error is the best metric for this project because it will assess the combined errors and will weight errors based on their size.

Before we make an attempt at regressing recipes to predict their ratings, we must pick which model to use. Again, since were predicting a continuous variable, we'll stick to models that are designed for regression. We'll  use random forests because they can make for quick and excellent models for categorical and numerical data. Depending on the performance, we may produce a decision tree in order to better interpret the strategy behind the tree regression. 

## Baseline Model
### Regression Modeling

We'll begin our modeling process by first only taking into account non-nutrition-related numerical data: number of minutes, steps and ingredients in a recipe. We associate these three variables with the effort required to perform a recipe. The longer it takes, the more steps involved, and the number of ingredients required are all things that usually determine the effort and difficulty of a recipe. We'll use these three variables to predict the rating of a recipe.

### Pipeline and Transformations
For our baseline model, we'll use a pipeline that transforms the design matrix using the natural logarithm, since these features are all skewed right. 

<iframe src="plots/quant_dists.html" width=800 height=600 frameBorder=0></iframe>

Here, we can see from the difference in shapes of the plots that using a logarithmic transformer reduces the skewness of the data, therefore making it easier to predict.

<iframe src="plots/log_quant_dists.html" width=800 height=600 frameBorder=0></iframe>

We began modeling with linear regression. We are restricting our baseline model to variables that indicate recipe difficulty/required effort. We used number of minutes, steps and ingredients, which are all quantitative variables. Using this model, where each feature is the natural log of the origianl data using a column transformer, the predictions have an r squared value of about -350. This is extremely low, making this a very poor model. This means that our model is worse than the ideal constant model, which we imagine also isn't very accurate.

However, we'll also be paying attention to the root mean squared error of the model, which is about 0.5, in order to compare our baseline to our final model. An r squared value of 0.5 is rather high for predicting values between 1 and 5, which also indicates that this is a poor prediction model. 

## Final Model
In order to construct our final model, we will add two new features: average review sentiment and nutrition. We learned from our exploratory data analysis that nutrition facts such as calories, fat and sugar all have some kind of relationship with recipe ratings. Additionally, we know for a fact that the sentiment of the review will be related to the rating. This combination of features should allow us to accurately predict ratings for recipes with ratings.

We will then tune our final model using GridSearchCV to select the optimal maximum depth and number of trees in the forest. Typically, the ideal tree depth for the testing model is 5, however we might find that our model only relies on a single feature, or perhaps relies on all 7 features, thus requiring a depth of up to 10. Also, we want to see if we need fewer or more than 100 trees per forest to obtain predictions with both low bias and variance. If we found out that we only needed 50 trees to train an identical model, we would certainly choose that hyperparameter to save us time.

Now, we can train our final model based on the ideal parameters calculated with our cross validation. Fortunately, our final model has a lower mean squared error (approximately .1 lower) than our baseline model. Although this model is still not very accurate, it's surely an improvement from the previous.

## Fairness Analysis

Our model has done better now that we've taken review sentiment into account, however we can't fully trust these results in future applications. One major thing to take into consideration is that we are trying to predict ratings for recipes without ratings, which is typically only happens to recipes with a single review. When multiple people review a single recipe, the chances of at least one of them leaving a rating is much higher than if just one person reviews a recipe. In fact, we can see that the average number of reviews for recipes without ratings is much lower (1.06) than for the rest of the recipes (2.85). As such, we'd like to perform a permutation test to see if our model predicts ratings equally well between recipes with one rating and recipes with more than one ratings. To test this, we'll use the difference in accuracy between our two groups. We are measuring performance of our model in terms of precision because we are regressing with equal interest in precision and recall.

**Null Hypothesis:** Our model is fair and will predict ratings for recipes with 1 review equally as accurately as for recipes with multiple reviews.

**Alternative Hypothesis:** Our model is unfair, and predicts ratings for recipes with 1 review with less accuracy than for recipes with multipel reviews.

We observed that our model predicts ratings better (difference of about 1.7 in R squared value) for recipes with multiple ratings than for recipes with just one rating. Our permutation test aims to see if this is by chance, or if it is due to a difference in the two populations.

<iframe src="plots/permutation.html" width=800 height=600 frameBorder=0></iframe>

Unfortunately, we must reject our null hypothesis in favor of the alternative. This means we believe our model has unfair predictions in that it is more accurate for recipes with multiple ratings. Although we could blame the model itself, there is most likely a reason behind this that is outside of our control. We suspect our most informative feature is review sentiment. In statistics, predictions improve with a higher sample size. This means that it makes sense why the model performs better on recipes with more reviews: it's simply performing better when there is a higher sample size. Although we would like to improve our model, we can't deny the fact that our aim is to predict data which has little information.
