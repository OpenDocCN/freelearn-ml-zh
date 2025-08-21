# 第十二章：创建一个简单的估计器

在本章中，我们将涵盖以下几种方法：

+   创建一个简单的估计器

# 介绍

我们将使用 scikit-learn 创建一个自定义估计器。我们将传统的统计数学和编程转化为机器学习。你可以通过使用 scikit-learn 强大的交叉验证功能，将任何统计学方法转变为机器学习。

# 创建一个简单的估计器

我们将进行一些工作，构建我们自己的 scikit-learn 估计器。自定义的 scikit-learn 估计器至少包括三个方法：

+   一个 `__init__` 初始化方法：该方法接受估计器的参数作为输入

+   一个 `fit` 方法：该方法用于训练估计器

+   一个 `predict` 方法：该方法对未见过的数据进行预测

从图示来看，类大致如下：

```py
#Inherit from the classes BaseEstimator, ClassifierMixin
class RidgeClassifier(BaseEstimator, ClassifierMixin):

 def __init__(self,param1,param2):
 self.param1 = param1
 self.param2 = param2

 def fit(self, X, y = None):
 #do as much work as possible in this method
 return self

 def predict(self, X_test):
 #do some work here and return the predictions, y_pred
 return y_pred 
```

# 准备工作

从 scikit-learn 中加载乳腺癌数据集：

```py
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer() 

new_feature_names = ['_'.join(ele.split()) for ele in bc.feature_names]

X = pd.DataFrame(bc.data,columns = new_feature_names)
y = bc.target
```

将数据划分为训练集和测试集：

```py
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7, stratify = y)
```

# 如何实现...

一个 scikit 估计器应该有一个 `fit` 方法，该方法返回类本身，并且有一个 `predict` 方法，该方法返回预测结果：

1.  以下是我们称之为 `RidgeClassifier` 的分类器。导入 `BaseEstimator` 和 `ClassifierMixin`，并将它们作为参数传递给你的新分类器：

```py
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Ridge

class RidgeClassifier(BaseEstimator, ClassifierMixin):

 """A Classifier made from Ridge Regression"""

 def __init__(self,alpha=0):
 self.alpha = alpha

 def fit(self, X, y = None):
 #pass along the alpha parameter to the internal ridge estimator and perform a fit using it
 self.ridge_regressor = Ridge(alpha = self.alpha) 
 self.ridge_regressor.fit(X, y)

 #save the seen class labels
 self.class_labels = np.unique(y)

 return self

 def predict(self, X_test):
 #store the results of the internal ridge regressor estimator
 results = self.ridge_regressor.predict(X_test)

 #find the nearest class label
 return np.array([self.class_labels[np.abs(self.class_labels - x).argmin()] for x in results])
```

让我们重点关注 `__init__` 方法。在这里，我们输入一个单一参数；它对应于底层岭回归器中的正则化参数。

在 `fit` 方法中，我们执行所有的工作。工作内容包括使用内部的岭回归器，并将类标签存储在数据中。如果类别超过两个，我们可能希望抛出一个错误，因为多个类别通常不能很好地映射到一组实数。在这个示例中，有两个可能的目标：恶性癌症或良性癌症。它们映射到实数，表示恶性程度，可以视为与良性相对立的度量。在鸢尾花数据集中，有 Setosa、Versicolor 和 Virginica 三种花。Setosaness 属性没有一个明确的对立面，除非以一对多的方式查看分类器。

在 `predict` 方法中，你会找到与岭回归器预测最接近的类标签。

1.  现在编写几行代码应用你的新岭回归分类器：

```py
r_classifier = RidgeClassifier(1.5) 
r_classifier.fit(X_train, y_train)
r_classifier.score(X_test, y_test)

0.95744680851063835
```

1.  它在测试集上的表现相当不错。你也可以在其上执行网格搜索：

```py
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0,0.5,1.0,1.5,2.0]}
gs_rc = GridSearchCV(RidgeClassifier(), param_grid, cv = 3).fit(X_train, y_train)

gs_rc.grid_scores_

[mean: 0.94751, std: 0.00399, params: {'alpha': 0},
 mean: 0.95801, std: 0.01010, params: {'alpha': 0.5},
 mean: 0.96063, std: 0.01140, params: {'alpha': 1.0},
 mean: 0.96063, std: 0.01140, params: {'alpha': 1.5},
 mean: 0.96063, std: 0.01140, params: {'alpha': 2.0}]
```

# 它是如何工作的...

创建你自己的估计器的目的是让估计器继承 scikit-learn 基础估计器和分类器类的属性。在以下代码中：

```py
r_classifier.score(X_test, y_test)
```

你的分类器查看了所有 scikit-learn 分类器的默认准确度评分。方便的是，你不需要去查找或实现它。此外，使用你的分类器时，过程与使用任何 scikit 分类器非常相似。

在以下示例中，我们使用逻辑回归分类器：

```py
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)

0.9521276595744681
```

你的新分类器比逻辑回归稍微好了一些。

# 还有更多...

有时，像 Python 的`statsmodels`或`rpy`（Python 中的 R 接口）这样的统计包包含非常有趣的统计方法，你可能希望将它们通过 scikit 的交叉验证来验证。或者，你可以自己编写方法并希望对其进行交叉验证。

以下是一个使用`statsmodels`的**广义估计方程**（**GEE**）构建的自定义估计器，该方法可以在[`www.statsmodels.org/dev/gee.html`](http://www.statsmodels.org/dev/gee.html)找到。

GEE 使用的是一般线性模型（借鉴了 R），我们可以选择一个类似分组的变量，其中观察值在一个聚类内部可能相关，但跨聚类之间无关——根据文档的说法。因此，我们可以根据某个变量进行分组或聚类，并查看组内相关性。

在这里，我们根据 R 风格公式创建一个基于乳腺癌数据的模型：

```py
'y ~ mean_radius + mean_texture + mean_perimeter + mean_area + mean_smoothness + mean_compactness + mean_concavity + mean_concave_points + mean_symmetry + mean_fractal_dimension + radius_error + texture_error + perimeter_error + area_error + smoothness_error + compactness_error + concavity_error + concave_points_error + symmetry_error + fractal_dimension_error + worst_radius + worst_texture + worst_perimeter + worst_area + worst_smoothness + worst_compactness + worst_concavity + worst_concave_points + worst_symmetry + worst_fractal_dimension'
```

我们根据特征`mean_concavity`进行聚类（`mean_concavity`变量未包含在 R 风格公式中）。首先导入`statsmodels`模块的库。示例如下：

```py
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.linear_model import Ridge

class GEEClassifier(BaseEstimator, ClassifierMixin):

 """A Classifier made from statsmodels' Generalized Estimating Equations documentation available at: http://www.statsmodels.org/dev/gee.html
    """

 def __init__(self,group_by_feature):
 self.group_by_feature = group_by_feature

 def fit(self, X, y = None):
 #Same settings as the documentation's example: 
 self.fam = sm.families.Poisson()
 self.ind = sm.cov_struct.Exchangeable()

 #Auxiliary function: only used in this method within the class
 def expand_X(X, y, desired_group): 
 X_plus = X.copy()
 X_plus['y'] = y

 #roughly make ten groups
 X_plus[desired_group + '_group'] = (X_plus[desired_group] * 10)//10

 return X_plus

 #save the seen class labels
 self.class_labels = np.unique(y)

 dataframe_feature_names = X.columns
 not_group_by_features = [x for x in dataframe_feature_names if x != self.group_by_feature]

 formula_in = 'y ~ ' + ' + '.join(not_group_by_features)

 data = expand_X(X,y,self.group_by_feature)
 self.mod = smf.gee(formula_in, 
 self.group_by_feature + "_group", 
 data, 
 cov_struct=self.ind, 
 family=self.fam)

 self.res = self.mod.fit()

 return self

 def predict(self, X_test):
 #store the results of the internal GEE regressor estimator
 results = self.res.predict(X_test)

 #find the nearest class label
 return np.array([self.class_labels[np.abs(self.class_labels - x).argmin()] for x in results])

 def print_fit_summary(self):
 print res.summary()
 return self
```

`fit`方法中的代码与 GEE 文档中的代码类似。你可以根据自己的具体情况或统计方法来调整代码。`predict`方法中的代码与创建的岭回归分类器类似。

如果你像运行岭回归估计器那样运行代码：

```py
gee_classifier = GEEClassifier('mean_concavity') 
gee_classifier.fit(X_train, y_train)
gee_classifier.score(X_test, y_test)

0.94680851063829785
```

关键在于，你将一种传统的统计方法转变为机器学习方法，利用了 scikit-learn 的交叉验证。

# 尝试在皮马糖尿病数据集上使用新的 GEE 分类器

尝试在皮马糖尿病数据集上使用 GEE 分类器。加载数据集：

```py
import pandas as pd

data_web_address = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

column_names = ['pregnancy_x', 
 'plasma_con', 
 'blood_pressure', 
 'skin_mm', 
 'insulin', 
 'bmi', 
 'pedigree_func', 
 'age', 
 'target']

feature_names = column_names[:-1]
all_data = pd.read_csv(data_web_address , names=column_names)

import numpy as np
import pandas as pd

X = all_data[feature_names]
y = all_data['target']
```

将数据集分成训练集和测试集：

```py
from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7,stratify=y)
```

使用 GEE 分类器进行预测。我们将使用`blood_pressure`列作为分组依据：

```py
gee_classifier = GEEClassifier('blood_pressure') 
gee_classifier.fit(X_train, y_train)
gee_classifier.score(X_test, y_test)

0.80519480519480524
```

你也可以尝试岭回归分类器：

```py
r_classifier = RidgeClassifier() 
r_classifier.fit(X_train, y_train)
r_classifier.score(X_test, y_test)

0.76623376623376627
```

你可以将这些方法——岭回归分类器和 GEE 分类器——与第五章中的逻辑回归进行比较，*线性模型 – 逻辑回归*。

# 保存你训练好的估计器

保存你的自定义估计器与保存任何 scikit-learn 估计器相同。按照以下方式将训练好的岭回归分类器保存到文件`rc_inst.save`中：

```py
import pickle

f = open('rc_inst.save','wb')
pickle.dump(r_classifier, f, protocol = pickle.HIGHEST_PROTOCOL)
f.close()
```

要检索训练好的分类器并使用它，请执行以下操作：

```py
import pickle

f = open('rc_inst.save','rb')
r_classifier = pickle.load(f)
f.close()
```

在 scikit-learn 中保存一个训练好的自定义估计器非常简单。
