from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split

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