from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

def load_housing_data():
  tarball_path = Path("datasets/housing.tgz")
  if not tarball_path.is_file():
    Path("datasets").mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
      housing_tarball.extractall(path="datasets")
  return pd.read_csv(Path("datasets/housing/housing.csv"))

def print_info(ds):
  print("head: ")
  print(ds.head(), "\n")
  print("info: ")
  print(ds.info(), "\n")
  print("ocean_proximity categories: ")
  print(ds["ocean_proximity"].value_counts(), "\n")
  print("describe: ")
  print(ds.describe(), "\n") # Los nulls son ignorados, std es la desviacion estandar

  ds.hist(bins=50, figsize=(12, 8))
  plt.savefig("plot.png")
  # plt.show()

'''
  This func picks some instances randomly to build a test set and obviously remove them from the training dataset.
  but it has some problems. Every time you run it, you will have different values for the test set
'''
def shuffle_and_split_data(data, test_ratio):
  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]

'''
  test set with hashes to avoid changing the dataset randomly once it is established
'''
def is_id_in_test_set(identifier, test_ratio):
  return crc32(np.int64(identifier)) < test_ratio * 2 ** 32

def split_data_with_id_hash(data, test_ratio, id_column):
  ids = data[id_column]
  in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
  return data.loc[~in_test_set], data.loc[in_test_set]


if __name__ == '__main__':
  housing = load_housing_data()
  print_info(housing)
  # train_set, test_set = shuffle_and_split_data(housing, 0.2)
  
  # housing_with_id = housing.reset_index() # adds an `index` column to use as id_column because the housing dataset does not have an identifier col
  # train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
  
  '''
  If you use the row index as a unique identifier, you need to make sure that new data gets appended to the end of the dataset and that no row 
  ever gets deleted. If this is not possible, then you can try to use the most stable features to build a unique identifier. 
  For example, a districts latitude and longitude are guaranteed to be stable for a few million years, so you could combine 
  them into an ID like so:
  '''
  # housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
  # train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
  # print_info(train_set)
  # print_info(test_set)

  '''
    Scikit-Learn provides a few functions to split datasets into multiple subsets in various ways. The simplest function is train_test_split(),
    which does pretty much the same thing as the shuffle_and_split_data() function we defined earlier, with a couple of additional features. First, there is a 
    random_state parameter that allows you to set the random generator seed. Second, you can pass it multiple datasets with an identical number of rows, 
    and it will split them on the same indices (this is very useful, for example, if you have a separate DataFrame for labels):
  '''
  train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

  '''
    Suppose you’ve chatted with some experts who told you that the median income is a very important attribute to predict median housing prices. 
    You may want to ensure that the test set is representative of the various categories of incomes in the whole dataset. 
    Since the median income is a continuous numerical attribute, you first need to create an income category attribute.
    The following code uses the pd.cut() function to create an income category attribute with five categories (labeled from 1 to 5); 
    category 1 ranges from 0 to 1.5 (i.e., less than $15,000), category 2 from 1.5 to 3, and so on:
  '''
  housing["income_cat"] = pd.cut(housing["median_income"],
                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                 labels=[1, 2, 3, 4, 5])
  plt.clf() 
  housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
  plt.xlabel("Income category")
  plt.ylabel("Number of districts")
  plt.savefig("income_cat.png")

  '''
    sklearn provides us with a method called split that returns an iterator over differetn training/test splits of the same data
  '''
  splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
  strat_splits = []
  for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

  # For now we can just use the first split
  strat_train_set, strat_test_set = strat_splits[0]

  # Or, since stratified sampling is fairly common, there’s a shorter way to get a single split using the train_test_split() function with the stratify argument
  strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
  print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

  # we can drop income_cat column now. we wont use it again
  for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

  # copy of the original training set to execute some transformations without modifying the original
  housing = strat_train_set.copy()

  # Scatterplot of geographical data
  plt.clf() 
  housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
  plt.savefig("geo_scatter.png")

  # setting alpha to visualize better the most populated places
  plt.clf()
  housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
  plt.savefig("geo_scatter_alpha.png")

  # plotting population as size and median_house_value as color
  plt.clf()
  housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
               s=housing["population"] / 100, label="population", 
               c="median_house_value", cmap="jet", colorbar=True,
               legend=True, sharex=False, figsize=(10, 7))
  plt.savefig("population_house_value.png")

  # Since the dataset is not too large, i can compute the standard correlation coefficient between every pair of attributes to see if i can identify some correlations
  # This coefficient ranges from -1 to 1, indicating a strong positive correlation for values near 1 and strong negative correlation for numbers near -1
  corr_matrix = housing.corr(numeric_only=True)
  print(corr_matrix["median_house_value"].sort_values(ascending=False))

  # Another way to check for correlation is to use pandas
  plt.clf()
  attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
  scatter_matrix(housing[attributes], figsize=(12, 8))
  plt.savefig("pandas_scatter_matrix.png")

  # We see a strong correlation between median_house_value and median_income
  plt.clf()
  housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
  plt.savefig("mhv_vs_mi.png")

  # Creating some new combinations to seek for correlation
  housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
  housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
  housing["people_per_house"] = housing["population"] / housing["households"]

  corr_matrix = housing.corr(numeric_only=True)
  print(corr_matrix["median_house_value"].sort_values(ascending=False))
  
  # revert to a clean training set. With drop i separate predictors and labels(median house value) and create a copy of strat_train_set without affecting it
  housing = strat_train_set.drop("median_house_value", axis=1)
  housing_labels = strat_train_set["median_house_value"].copy()

  # We noticed earlier that total_bedrooms has some missing values, so we can get rid of the corresponding districts, get rid of the whole attribute or set the missing values to some value (zero, the mean, the median)
  housing.dropna(subset=["total_bedrooms"], inplace=True) # Option 1
  housing.drop("total_bedrooms", axis=1) # Option 2
  median = housing["total_bedrooms"].median()
  housing["total_bedrooms"].fillna(median, inplace=True)

  # we decided the third option because it is the least destructive but using scikit-learn class called SimpleImputer
  imputer = SimpleImputer(strategy="median")

  # we cant compute median on non numerical attributes, so we need to create a copy of the data but only with numerical attributes
  housing_num = housing.select_dtypes(include=[np.number])

  imputer.fit(housing_num)
  print(imputer.statistics_)
  print(housing_num.median().values)

  # Now i can use this "trained" imputer to transform the training set by replacing missing values with the learned medias
  X = imputer.transform(housing_num)

  # sklearn.impute has more powerful imputers like KNNImputer or IterativeImputer
  # sklearn transformers returns Numpy arrays, so we need to wrap it in a dataframe and recover column names
  housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

  housing_cat = housing[["ocean_proximity"]] # This is a categorical attribute, it has a limited number of possible values
  print(housing_cat.head(8))

  ordinal_encoder = OrdinalEncoder()
  housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
  print(housing_cat_encoded[:8])
  print(ordinal_encoder.categories_) #One problem with this representation is that ML algorithms will assume that two nearby values are more similar than two distant values, but categories o and 4 are more similar than 0 and 1 here
  
  # One solution here is to create one binary attribute per category. This is called one-hot encoding. The new attributes are sometimes called dummy attributes.
  cat_encoder = OneHotEncoder()
  housing_cat_1hot = cat_encoder.fit_transform(housing_cat) #is a SciPy sparse matrix, instead of a NumPy array
  # A sparse matrix is a very efficient representation for matrices that contain mostly zeros. Indeed, internally it only stores the nonzero values and their positions. 
  print(housing_cat_1hot.toarray())
  print(cat_encoder.categories_)
  print(cat_encoder.feature_names_in_) # SciKit-Learn stores the column names here
  print(cat_encoder.get_feature_names_out()) # Build a dataframe around transformer's output

  # We need to normalize all values between zero because our numeric attributes has very different scales.
  # We can use MinMaxScaler. It substracts the minimum and divide by the difference, giving us a range from 0 to 1. We can set the resulting range with feature_range
  min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
  housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

  # Also we can use standardization. Standardization substracts the mean value (values now have a zero mean), then divides the result
  # by the standard deviation (standardized values have a standard deviation equal to 1). This solution does not restrict values to
  # a specific range but its much less affected by outliers
  std_scaler = StandardScaler()
  housing_num_std_scaled = std_scaler.fit_transform(housing_num)

  # When a feature’s distribution has a heavy tail (i.e., when values far from the mean are not exponentially rare), both min-max scaling and standardization will squash most values into a small range. Machine learning models generally don’t like this at all, as you will see in Chapter 4. So before you scale the feature, you should first transform it to shrink the heavy tail, and if possible to make the distribution roughly symmetrical. For example, a common way to do this for positive features with a heavy tail to the right is to replace the feature with its square root (or raise the feature to a power between 0 and 1). If the feature has a really long and heavy tail, such as a power law distribution, then replacing the feature with its logarithm may help. For example, the population feature roughly follows a power law: districts with 10,000 inhabitants are only 10 times less frequent than districts with 1,000 inhabitants, not exponentially less frequent. Figure 2-17 shows how much better this feature looks when you compute its log: it’s very close to a Gaussian distribution (i.e., bell-shaped).
  # Another approach to handle heavy-tailed features consists in bucketizing the feature. This means chopping its distribution into roughly equal-sized buckets, and replacing each feature value with the index of the bucket it belongs to, much like we did to create the income_cat feature (although we only used it for stratified sampling). For example, you could replace each value with its percentile. Bucketizing with equal-sized buckets results in a feature with an almost uniform distribution, so there’s no need for further scaling, or you can just divide by the number of buckets to force the values to the 0–1 range.
  # When a feature has a multimodal distribution (i.e., with two or more clear peaks, called modes), such as the housing_median_age feature, it can also be helpful to bucketize it, but this time treating the bucket IDs as categories, rather than as numerical values. This means that the bucket indices must be encoded, for example using a OneHotEncoder (so you usually don’t want to use too many buckets). This approach will allow the regression model to more easily learn different rules for different ranges of this feature value. For example, perhaps houses built around 35 years ago have a peculiar style that fell out of fashion, and therefore they’re cheaper than their age alone would suggest.
  # Another approach to transforming multimodal distributions is to add a feature for each of the modes (at least the main ones), representing the similarity between the housing median age and that particular mode. The similarity measure is typically computed using a radial basis function (RBF)—any function that depends only on the distance between the input value and a fixed point. The most commonly used RBF is the Gaussian RBF
  # Using Scikit-Learn’s rbf_kernel() function, you can create a new Gaussian RBF feature measuring the similarity between the housing median age and 35:
  age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

  # So far we’ve only looked at the input features, but the target values may also need to be transformed. For example, if the target distribution has a heavy tail, you may choose to replace the target with its logarithm. But if you do, the regression model will now predict the log of the median house value, not the median house value itself.
  # Luckily, most of Scikit-Learn’s transformers have an inverse_transform() method, making it easy to compute the inverse of their transformations. 
  #  For example, the following code example shows how to scale the labels using a StandardScaler (just like we did for inputs), then train a simple linear regression model on the resulting scaled labels and use it to make predictions on some new data, which we transform back to the original scale using the trained scaler’s inverse_transform() method.
  target_scaler = StandardScaler()
  scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

  model = LinearRegression()
  model.fit(housing[["median_income"]], scaled_labels)
  some_new_data = housing[["median_income"]].iloc[:5] # pretend this is new data

  scaled_predictions = model.predict(some_new_data)
  predictions = target_scaler.inverse_transform(scaled_predictions)

  # This works fine but a simpler option is to use a TransformedTargetRegressor.
  model = TransformedTargetRegressor(regressor=LinearRegression(),
                                     transformer=StandardScaler())
  model.fit(housing[["median_income"]], housing_labels)
  predictions = model.predict(some_new_data)