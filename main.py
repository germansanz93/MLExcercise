from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

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
  