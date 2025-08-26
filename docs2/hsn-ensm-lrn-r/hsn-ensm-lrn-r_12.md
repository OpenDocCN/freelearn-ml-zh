# 第十二章。接下来是什么？

在整本书中，我们学习了集成学习并探讨了其在许多场景中的应用。在介绍章节中，我们考察了不同的例子、数据集和模型，并发现没有哪个模型或技术比其他模型表现得更好。这意味着在处理这个问题时，我们的警惕性应该始终保持，因此分析师必须极端谨慎地进行。从众多模型中选择最佳模型的方法意味着我们拒绝了所有表现略逊于其他模型的模型，因此，在追求“最佳”模型的过程中浪费了大量资源。

在第七章中，我们看到了“通用集成技术”，如果拥有多个分类器，并且每个分类器的性能都优于随机猜测，那么分类器的多数投票会带来性能的提升。我们还看到，在有相当数量的分类器的情况下，多数投票的整体准确率高于最准确的分类器。尽管多数投票基于过于简化的假设，即分类器是相互独立的，但集成学习的基础和重要性得到了体现，前景看起来更加光明，因为我们确保了具有最高准确率的集成分类器。集成中的分类器或模型被称为“基础分类器/学习器/模型”。如果所有基础模型都属于同一模型家族，或者如果家族是逻辑模型、神经网络、决策树或 SVM，那么我们将这些归类为同质集成。如果基础模型属于两个或更多模型家族，那么集成被称为异质集成。

集成学习的一个基本方面在于*重采样技术*。在第二章“Bootstrapping”中介绍了 Jackknife 和 Bootstrap 方法，并针对不同类别的模型进行了说明。Jackknife 方法使用伪值，我们见证了其在一些复杂场景中的应用。在机器学习领域，伪值的使用不足，它们可能在关系或感兴趣参数相当复杂且可能存在更灵活方法的情况下有用。伪值的概念也可以用于集成诊断，可以创建一个*伪准确度*度量，这将给集成中的分类器带来额外的准确度。Efron 和 Tibshirani（1990）讨论了获取 Bootstrap 样本的不同方法。在 Bagging 和随机森林步骤中获得的重采样中，我们遵循简单的 Bootstrap 样本抽取观察值的方法。从现有的 Bootstrap 文献中获取不同抽样方法是一个潜在的工作领域。

Bagging、随机森林和提升是具有决策树作为其基学习器的同质集成。本书中讨论的决策树可以被称为频率主义方法或经典方法。决策树的贝叶斯实现可以在一种称为**贝叶斯加性回归树**的技术中找到。该方法的 R 实现可在[BART 包](https://cran.r-project.org/web/packages/BART/index.html)中找到。BART 的理论基础可在[`arxiv.org/pdf/0806.3286.pdf`](https://arxiv.org/pdf/0806.3286.pdf)找到。基于 BART 的同质集成扩展需要全面进行。特别是，Bagging 和随机森林需要以 BART 作为基学习器进行扩展。

对于异构的基本模型，使用堆叠集成方法来设置集成模型。对于分类问题，我们预计加权投票比简单多数投票更有用。同样，模型的加权预测预计会比简单平均表现更好。提出的用于测试基本模型相似性的统计测试都是*经典*测试。预期的贝叶斯一致性度量将为我们提供进一步的指导，本书作者现在意识到这种评估在集成多样性背景下正在进行。你可以阅读 Broemeling (2009)了解更多信息，[`www.crcpress.com/Bayesian-Methods-for-Measures-of-Agreement/Broemeling/p/book/9781420083415`](https://www.crcpress.com/Bayesian-Methods-for-Measures-of-Agreement/Broemeling/p/book/9781420083415)。此外，当涉及到大型数据集时，生成独立模型集的选项需要系统地开发。

时间序列数据具有不同的结构，观测值的依赖性意味着我们无法在不进行适当调整的情况下直接应用集成方法。第十一章，*集成时间序列模型*，更详细地探讨了这一主题。最近已经为时间序列数据开发了随机森林。可以在[`petolau.github.io/Ensemble-of-trees-for-forecasting-time-series/`](https://petolau.github.io/Ensemble-of-trees-for-forecasting-time-series/)看到随机森林的实现，如果你感兴趣，可以参考此链接获取更多信息。

高维数据分析是另一个最近的话题，本书没有涉及。Buhlmann 和 van de Geer (2011) 的第十二章（见[`www.springer.com/in/book/9783642201912`](https://www.springer.com/in/book/9783642201912)）在这一领域提供了有用的指导。对于提升数据，Schapire 和 Freund (2012) 的著作（见[`mitpress.mit.edu/books/boosting`](https://mitpress.mit.edu/books/boosting)）是一笔真正的财富。Zhou (2012)（见[`www.crcpress.com/Ensemble-Methods-Foundations-and-Algorithms/Zhou/p/book/9781439830031`](https://www.crcpress.com/Ensemble-Methods-Foundations-and-Algorithms/Zhou/p/book/9781439830031)）是关于集成方法的重要书籍，本书受益于其见解。Kuncheva (2004-14)（见[`onlinelibrary.wiley.com/doi/book/10.1002/9781118914564`](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118914564)）可能是第一本详细阐述集成方法的书籍，并包含了许多关于集成诊断的其他细节。Dixit (2017)（见[`www.packtpub.com/big-data-and-business-intelligence/ensemble-machine-learning`](https://www.packtpub.com/big-data-and-business-intelligence/ensemble-machine-learning)）是关于集成方法的另一本近期著作，书中使用 Python 软件展示了这些方法。

最后，读者应始终关注最新的发展。对于 R 语言的实现，资源最好的地方是[`cran.r-project.org/web/views/MachineLearning.html`](https://cran.r-project.org/web/views/MachineLearning.html)。

接下来，我们将查看一系列重要的期刊。在这些期刊上，关于机器学习以及集成学习的话题有很多发展和讨论。以下是一些期刊：

+   机器学习研究杂志 ([`www.jmlr.org/`](http://www.jmlr.org/))

+   IEEE 信号处理与机器智能杂志 ([`ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34`](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34))

+   模式识别信件 ([`www.journals.elsevier.com/pattern-recognition-letters`](https://www.journals.elsevier.com/pattern-recognition-letters))

+   机器学习([`www.springer.com/computer/ai/journal/10994`](https://www.springer.com/computer/ai/journal/10994))

+   神经计算 ([`www.journals.elsevier.com/neurocomputing`](https://www.journals.elsevier.com/neurocomputing))

+   最后但同样重要的是（尽管这不是一本期刊），[`www.packtpub.com/tech/Machine-Learning`](https://www.packtpub.com/tech/Machine-Learning)

如果你觉得这本书有用，我们下次再版时应该再次见面！

# 附录 A. 参考文献列表

# 参考文献

Abraham, B. 和 Ledolter, J. ([`onlinelibrary.wiley.com/doi/book/10.1002/9780470316610`](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316610)), 1983. *预测的统计方法*. J. Wiley

Andersen, P.K., Klein, J.P. 和 Rosthøj, S., ([`doi.org/10.1093/biomet/90.1.15`](https://doi.org/10.1093/biomet/90.1.15)) 2003\. Generalised linear models for correlated pseudo-observations, with applications to multi-state models. *生物计量学*, *90*(1), pp.15-27.

Berk, R.A., ([`www.springer.com/in/book/9783319440477`](https://www.springer.com/in/book/9783319440477)) 2016\. *从回归视角看统计学习，第二版*. 纽约：Springer.

Bou-Hamad, I., Larocque, D. 和 Ben-Ameur, H., ([`projecteuclid.org/euclid.ssu/1315833185`](https://projecteuclid.org/euclid.ssu/1315833185)) 2011\. A review of survival trees. *统计调查*, *5*, pp.44-71.

Box, G.E., Jenkins, G.M., Reinsel, G.C. 和 Ljung, G.M., ([`onlinelibrary.wiley.com/doi/book/10.1002/9781118619193`](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118619193)) 2015\. *时间序列分析：预测与控制*. 约翰·威利与 Sons.

Breiman, L., ([`link.springer.com/article/10.1007/BF00058655`](https://link.springer.com/article/10.1007/BF00058655)) 1996\. Bagging predictors. *机器学习*, *24*(2), pp.123-140.

Breiman, L., Friedman, J.H., Olshen, R.A. 和 Stone, C.J., ([`www.taylorfrancis.com/books/9781351460491`](https://www.taylorfrancis.com/books/9781351460491)) 1984\. *分类与回归树*. 路透社.

Broemeling, L.D., ([`www.crcpress.com/Bayesian-Methods-for-Measures-of-Agreement/Broemeling/p/book/9781420083415`](https://www.crcpress.com/Bayesian-Methods-for-Measures-of-Agreement/Broemeling/p/book/9781420083415)) 2009\. *度量一致性的贝叶斯方法*. Chapman and Hall/CRC.

Bühlmann, P. 和 Van De Geer, S., ([`www.springer.com/in/book/9783642201912`](https://www.springer.com/in/book/9783642201912)) 2011\. *高维数据统计：方法、理论与应用*. Springer 科学与商业媒体.

Chatterjee, S. 和 Hadi, A.S., ([`www.wiley.com/en-us/Regression+Analysis+by+Example%2C+5th+Edition-p-9780470905845`](https://www.wiley.com/en-us/Regression+Analysis+by+Example%2C+5th+Edition-p-9780470905845)) 2012\. Regression Analysis by Example, 第五版. 约翰·威利与 Sons.

Ciaburro, G., ([`www.packtpub.com/big-data-and-business-intelligence/regression-analysis-r`](https://www.packtpub.com/big-data-and-business-intelligence/regression-analysis-r)) 2018\. Regression Analysis with R, Packt Publishing Ltd.

Cleveland, R.B., Cleveland, W.S., McRae, J.E. 和 Terpenning, I., ([`www.nniiem.ru/file/news/2016/stl-statistical-model.pdf`](http://www.nniiem.ru/file/news/2016/stl-statistical-model.pdf)) 1990\. STL: A Seasonal-Trend Decomposition. *官方统计杂志*, *6(*1), pp.3-73.

Cox, D.R.，([`eclass.uoa.gr/modules/document/file.php/MATH394/Papers/%5BCox(1972)%5D%20Regression%20Models%20and%20Life%20Tables.pdf`](https://eclass.uoa.gr/modules/document/file.php/MATH394/Papers/%5BCox(1972)%5D%20Regression%20Models%20and%20Life%20Tables.pdf)) 1972\. 回归模型和生命表。*《皇家统计学会会刊》*，系列 B（方法论），**34**，第 187-220 页。

Cox, D.R.，([`academic.oup.com/biomet/article-abstract/62/2/269/337051`](https://academic.oup.com/biomet/article-abstract/62/2/269/337051)) 1975\. 部分似然。*《生物计量学》*，*62*(2)，第 269-276 页。

Dixit, A.，([`www.packtpub.com/big-data-and-business-intelligence/ensemble-machine-learning`](https://www.packtpub.com/big-data-and-business-intelligence/ensemble-machine-learning))2017\. *集成机器学习：一本结合强大机器学习算法以构建优化模型的入门指南*。Packt Publishing Ltd.

Draper, N.R. 和 Smith, H.，([`onlinelibrary.wiley.com/doi/book/10.1002/9781118625590`](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118625590))1999/2014\. *应用回归分析*（第 326 卷）。John Wiley & Sons。

Efron, B. ([`projecteuclid.org/download/pdf_1/euclid.aos/1176344552`](https://projecteuclid.org/download/pdf_1/euclid.aos/1176344552)) 1979\. 自举方法 ([`link.springer.com/chapter/10.1007/978-1-4612-4380-9_41`](https://link.springer.com/chapter/10.1007/978-1-4612-4380-9_41)): 对 Jackknife 的另一种看法，*《统计年鉴》*，7，1-26。

Efron, B. 和 Hastie, T.，2016\. ([`web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf`](https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf)) *计算机时代的统计推断*（第 5 卷）。剑桥大学出版社。

Efron, B. 和 Tibshirani, R.J.，([`www.crcpress.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317`](https://www.crcpress.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317)) 1994\. *自举方法导论*。CRC 出版社。

Friedman, J.H.，Hastie, T. 和 Tibshirani, R. ([`projecteuclid.org/download/pdf_1/euclid.aos/1016218223`](https://projecteuclid.org/download/pdf_1/euclid.aos/1016218223))2001\. 渐进函数逼近：梯度提升机。*《统计年鉴》*，29(5)：1189–1232。

Gordon, L. 和 Olshen, R.A.，([`europepmc.org/abstract/med/4042086`](https://europepmc.org/abstract/med/4042086))1985\. 树结构生存分析。癌症治疗报告，69(10)，第 1065-1069 页。

Hastie, T.，Tibshirani, R. 和 Friedman, J. ([`www.springer.com/in/book/9780387848570`](https://www.springer.com/in/book/9780387848570))，2009 年，*统计学习元素，第二版*，Springer。

Haykin, S.S, 2009\. ([`www.pearson.com/us/higher-education/program/Haykin-Neural-Networks-and-Learning-Machines-3rd-Edition/PGM320370.html`](https://www.pearson.com/us/higher-education/program/Haykin-Neural-Networks-and-Learning-Machines-3rd-Edition/PGM320370.html)) *神经网络与学习机器* (第 3 卷). 上萨德尔河，新泽西州，美国:: Pearson.

Kalbfleisch, J.D. and Prentice, R.L. ([`onlinelibrary.wiley.com/doi/abs/10.2307/3315078`](https://onlinelibrary.wiley.com/doi/abs/10.2307/3315078)), 2002\. *失效时间数据的统计分析*. John Wiley & Sons.

Kuncheva, L.I., ([`www.wiley.com/en-us/Combining+Pattern+Classifiers%3A+Methods+and+Algorithms%2C+2nd+Edition-p-9781118315231`](https://www.wiley.com/en-us/Combining+Pattern+Classifiers%3A+Methods+and+Algorithms%2C+2nd+Edition-p-9781118315231)) 2014\. *模式分类器的组合：方法和算法*. 第二版. John Wiley & Sons.

LeBlanc, M. and Crowley, J., ([`www.jstor.org/stable/2532300`](https://www.jstor.org/stable/2532300))1992\. 缩短生存数据的相对风险树. *生物统计学*, 第 411-425 页.

Lee, S.S. and Elder, J.F., ([`citeseerx.ist.psu.edu/viewdoc/download;jsessionid=6B151AAB29C69A4D4C35C8C4BBFC67F5?doi=10.1.1.34.1753&rep=rep1&type=pdf`](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=6B151AAB29C69A4D4C35C8C4BBFC67F5?doi=10.1.1.34.1753&rep=rep1&type=pdf)) 1997\. 使用顾问感知器捆绑异构分类器. *白皮书*.

Mardia, K. , Kent, J., and Bibby, J.M.., ([`www.elsevier.com/books/multivariate-analysis/mardia/978-0-08-057047-1`](https://www.elsevier.com/books/multivariate-analysis/mardia/978-0-08-057047-1)) 1979\. *多元分析*. Academic Press.

Montgomery, D.C., Peck, E.A. and Vining, G.G., ([`www.wiley.com/en-us/Introduction+to+Linear+Regression+Analysis%2C+5th+Edition-p-9781118627365`](https://www.wiley.com/en-us/Introduction+to+Linear+Regression+Analysis%2C+5th+Edition-p-9781118627365)) 2012\. *线性回归分析导论* (第 821 卷). John Wiley & Sons.

Perrone, M.P., and Cooper, L.N., ([`www.worldscientific.com/doi/abs/10.1142/9789812795885_0025`](https://www.worldscientific.com/doi/abs/10.1142/9789812795885_0025))1993\. 当网络意见不一致时：混合神经网络的集成方法. 在 Mammone, R.J. (编者), *语音和图像处理的神经网络*. Chapman Hall.

Ripley, B.D., ([`admin.cambridge.org/fk/academic/subjects/statistics-probability/computational-statistics-machine-learning-and-information-sc/pattern-recognition-and-neural-networks`](http://admin.cambridge.org/fk/academic/subjects/statistics-probability/computational-statistics-machine-learning-and-information-sc/pattern-recognition-and-neural-networks))2007\. *模式识别与神经网络*. 剑桥大学出版社.

Quenouille, M.H., ([`www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/approximate-tests-of-correlation-in-timeseries-3/F6D24B2A8574F1716E44BE788696F9C7`](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/approximate-tests-of-correlation-in-timeseries-3/F6D24B2A8574F1716E44BE788696F9C7)) 1949 年 7 月。时间序列中相关性的近似检验 3。载于 *Mathematical Proceedings of the Cambridge Philosophical Society* (第 45 卷，第 3 期，第 483-484 页)。剑桥大学出版社。

Quinlan, J. R. (1993)，([`www.elsevier.com/books/c45/quinlan/978-0-08-050058-4`](https://www.elsevier.com/books/c45/quinlan/978-0-08-050058-4)) *C4.5：机器学习程序*，Morgan Kaufmann。

Ridgeway, G., Madigan, D. and Richardson, T., ([`dimacs.rutgers.edu/archive/Research/MMS/PAPERS/BNBR.pdf`](http://dimacs.rutgers.edu/archive/Research/MMS/PAPERS/BNBR.pdf)) 1999 年 1 月. 回归问题的提升方法。载于 *AISTATS*。

Schapire, R.E. 和 Freund, Y., ([`dimacs.rutgers.edu/archive/Research/MMS/PAPERS/BNBR.pdf`](http://dimacs.rutgers.edu/archive/Research/MMS/PAPERS/BNBR.pdf)) 2012 年。*提升：基础与算法*。麻省理工学院出版社。

Seni, G. 和 Elder, J.F.，([`www.morganclaypool.com/doi/abs/10.2200/S00240ED1V01Y200912DMK002`](https://www.morganclaypool.com/doi/abs/10.2200/S00240ED1V01Y200912DMK002)) 2010 年。数据挖掘中的集成方法：通过组合预测提高准确性。*数据挖掘与知识发现综合讲座*，第 2 卷(第 1 期)，第 1-126 页。

Tattar, P.N.，Ramaiah, S. 和 Manjunath, B.G.，([`onlinelibrary.wiley.com/doi/book/10.1002/9781119152743`](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119152743)) 2016 年。*使用 R 的统计学课程*。John Wiley & Sons。

Tattar, P.N., 2017 年。([`www.packtpub.com/big-data-and-business-intelligence/statistical-application-development-r-and-python-second-edition`](https://www.packtpub.com/big-data-and-business-intelligence/statistical-application-development-r-and-python-second-edition)) *使用 R 和 Python 进行统计应用开发*。Packt Publishing Ltd.

Tattar, P., Ojeda, T., Murphy, S.P., Bengfort, B. 和 Dasgupta, A., ([`www.packtpub.com/big-data-and-business-intelligence/practical-data-science-cookbook-second-edition`](https://www.packtpub.com/big-data-and-business-intelligence/practical-data-science-cookbook-second-edition)) 2017 年。*实用数据科学食谱*。Packt Publishing Ltd.

Zhang, H. 和 Singer, B.H.，([`www.springer.com/in/book/9781441968234`](https://www.springer.com/in/book/9781441968234)) 2010 年。*递归分割与应用*。Springer Science & Business Media。

Zemel, R.S. 和 Pitassi, T.，([`papers.nips.cc/paper/1797-a-gradient-based-boosting-algorithm-for-regression-problems.pdf`](http://papers.nips.cc/paper/1797-a-gradient-based-boosting-algorithm-for-regression-problems.pdf) )2001。用于回归问题的梯度提升算法。在 *Advances in neural information processing systems* (pp. 696-702)。

(2017)。`ClustOfVar`: 变量聚类。R 包版本 1.1。

# R 包引用

R 包版本 1.6-8\. [`CRAN.R-project.org/package=e1071`](https://CRAN.R-project.org/package=e1071) Alboukadel Kassambara 和 Fabian Mundt (2017)。`factoextra`: 提取

《使用 R 的统计学课程》。R 包版本 1.0。

[`CRAN.R-project.org/package=ACSWR`](https://CRAN.R-project.org/package=ACSWR)

Alfaro, E., Gamez, M. Garcia, N. (2013). `adabag`: 用于 Caret 模型集成的 R 包。R 包版本 2.0.0。

第二版。 Thousand Oaks CA: Sage。URL：

软件，54(2)，1-35\. URL [`www.jstatsoft.org/v54/i02/`](http://www.jstatsoft.org/v54/i02/).

Chris Keefer, Allan Engelhardt, Tony Cooper, Zachary Mayer, Brenton

函数。R 包版本 1.3-19。

[`CRAN.R-project.org/package=caretEnsemble`](https://CRAN.R-project.org/package=caretEnsemble) Venables, W. N. & Ripley, B. D. (2002) Modern Applied Statistics

第二版。 Thousand Oaks CA: Sage。URL：

Angelo Canty 和 Brian Ripley (2017)。`boot`: Bootstrap R (S-Plus)

周志华，([`www.crcpress.com/Ensemble-Methods-Foundations-and-Algorithms/Zhou/p/book/9781439830031`](https://www.crcpress.com/Ensemble-Methods-Foundations-and-Algorithms/Zhou/p/book/9781439830031))2012。*集成方法：基础与算法*。Chapman and Hall/CRC。

Kenkel, R Core 团队，Michael Benesty，Reynald Lescarbeau，Andrew

函数。R 包版本 1.3-19。

Ziem, Luca Scrucca, Yuan Tang, Can Candan 和 Tyler Hunt. (2017).

`caret`: 分类和回归训练。R 包版本

6.0-77\. [`CRAN.R-project.org/package=caret`](https://CRAN.R-project.org/package=caret)

统计学，概率论组（以前称为 E1071），维也纳科技大学。使用提升和装袋进行分类。统计软件，54(2)，1-35\. URL [`www.jstatsoft.org/v54/i02/`](http://www.jstatsoft.org/v54/i02/).

Max Kuhn。贡献来自 Jed Wing, Steve Weston, Andre Williams, Prabhanjan Tattar (2015)。`ACSWR`: 《A Companion Package for the Book "A Modern Course in Statistics with R"` 的配套包。

Zachary A. Deane-Mayer 和 Jared E. Knowles (2016)。`caretEnsemble`:

与 S 的第四版。Springer，纽约。ISBN 0-387-95457-0 `class`

Marie Chavent, Vanessa Kuentz, Benoit Liquet 和 Jerome Saracco

John Fox 和 Sanford Weisberg (2011)。《An {R} Companion to Applied Regression, Second Edition》。 Thousand Oaks CA: Sage。URL：

[`CRAN.R-project.org/package=ClustOfVar`](https://CRAN.R-project.org/package=ClustOfVar) David Meyer, Evgenia Dimitriadou, Kurt Hornik, Andreas Weingessel

和 Friedrich Leisch (2017)。`e1071`: 部门的其他函数。R 包版本

《使用 R 的统计学课程》。R 包版本 1.0。

[`socserv.socsci.mcmaster.ca/jfox/Books/Companion`](http://socserv.socsci.mcmaster.ca/jfox/Books/Companion) `car`

和可视化多元数据分析结果。R 包

版本 1.0.5\. [`CRAN.R-project.org/package=factoextra`](https://CRAN.R-project.org/package=factoextra)

Sebastien Le, Julie Josse, Francois Husson (2008). `FactoMineR`: An R

多变量分析包。统计软件杂志，

25(1), 1-18\. 10.18637/jss.v025.i01

Alina Beygelzimer, Sham Kakadet, John Langford, Sunil Arya, David

Mount and Shengqiao Li (2013). `FNN`: 快速最近邻搜索

算法和应用。R 包版本 1.1。

[`CRAN.R-project.org/package=FNN`](https://CRAN.R-project.org/package=FNN)

Hyndman RJ (2017). `_forecast`: 时间序列预测函数

和线性模型。R 包版本 8.2，<URL:

[`pkg.robjhyndman.com/forecast`](http://pkg.robjhyndman.com/forecast)>.

David Shaub and Peter Ellis (2018). forecastHybrid: 方便

集成时间序列预测函数。R 包版本

2.0.10\. [`CRAN.R-project.org/package=forecastHybrid`](https://CRAN.R-project.org/package=forecastHybrid)

Greg Ridgeway with contributions from others (2017). `gbm`:

广义提升回归模型。R 包版本 2.1.3。

[`CRAN.R-project.org/package=gbm`](https://CRAN.R-project.org/package=gbm)

Vincent J Carey。由 Thomas Lumley 和 Brian Ripley 转移到 R。注意

维护者无法提供有关使用包的建议

他们没有编写。 (2015). `gee`: 广义估计方程

解决器。R 包版本 4.13-19。

[`CRAN.R-project.org/package=gee`](https://CRAN.R-project.org/package=gee)

H2O.ai 团队 (2017). `h2o`: H2O 的 R 接口。R 包版本

3.16.0.2\. [`CRAN.R-project.org/package=h2o`](https://CRAN.R-project.org/package=h2o)

Andrea Peters and Torsten Hothorn (2017). `ipred`: 改进的

预测因子。R 包版本 0.9-6。

[`CRAN.R-project.org/package=ipred`](https://CRAN.R-project.org/package=ipred)

Alexandros Karatzoglou, Alex Smola, Kurt Hornik, Achim Zeileis

(2004). `kernlab` - R 中的核方法 S4 包。统计软件杂志，

统计软件 11(9), 1-20\. URL

[`www.jstatsoft.org/v11/i09/`](http://www.jstatsoft.org/v11/i09/)

Friedrich Leisch & Evgenia Dimitriadou (2010). `mlbench`: 机器

学习基准问题。R 包版本 2.1-1。

Daniel J. Stekhoven (2013). `missForest`: 非参数缺失值

使用随机森林进行插补。R 包版本 1.4。

Alan Genz, Frank Bretz, Tetsuhisa Miwa, Xuefei Mi, Friedrich Leisch,

Fabian Scheipl, Torsten Hothorn (2017). `mvtnorm`: 多变量正态

和 t 分布。R 包版本 1.0-6\. URL

[`CRAN.R-project.org/package=mvtnorm`](http://CRAN.R-project.org/package=mvtnorm)

Beck M (2016). `_NeuralNetTools`: 用于可视化和分析的

神经网络。R 包版本 1.5.0，<URL:

[`CRAN.R-project.org/package=NeuralNetTools`](https://CRAN.R-project.org/package=NeuralNetTools)>.

Venables, W. N. & Ripley, B. D. (2002) 现代应用统计学

使用 S. 第四版。Springer，纽约。ISBN 0-387-95457-0 `nnet`

Michael P. Fay, Pamela A. Shaw (2010). 精确和渐近加权

间隔数据 Logrank 测试：间隔 R 包。

统计软件杂志，36(2)，1-34\. URL

[`www.jstatsoft.org/v36/i02/`](http://www.jstatsoft.org/v36/i02/). `perm`

Hadley Wickham (2011). 数据分割-应用-组合策略

分析。统计软件杂志，40(1)，1-29\. URL

[`www.jstatsoft.org/v40/i01/`](http://www.jstatsoft.org/v40/i01/). `plyr`

Xavier Robin, Natacha Turck, Alexandre Hainard, Natalia Tiberti,

Frédérique Lisacek, Jean-Charles Sanchez 和 Markus Müller (2011).

`pROC`: 用于 R 和 S+ 的开源包，用于分析和比较 ROC 曲线

曲线。生物信息学杂志，12，p. 77\. DOI: 10.1186/1471-2105-12-77

[`www.biomedcentral.com/1471-2105/12/77/`](http://www.biomedcentral.com/1471-2105/12/77/)

Maja Pohar Perme 和 Mette Gerster (2017). `pseudo`: 计算

模型中的伪观察。R 包版本 1.4.3。

[`CRAN.R-project.org/package=pseudo`](https://CRAN.R-project.org/package=pseudo)

A. Liaw 和 M. Wiener (2002). 分类和回归

`randomForest`. R News 2(3), 18--22.

Aleksandra Paluszynska 和 Przemyslaw Biecek (2017).

`randomForestExplainer`: 解释和可视化随机森林

变量重要性术语。R 包版本 0.9。

[`CRAN.R-project.org/package=randomForestExplainer`](https://CRAN.R-project.org/package=randomForestExplainer)

Terry Therneau, Beth Atkinson 和 Brian Ripley (2017). `rpart`: 

递归分割和回归树。R 包版本

4.1-11\. [`CRAN.R-project.org/package=rpart`](https://CRAN.R-project.org/package=rpart)

Prabhanjan Tattar (2013). `RSADBE`: 数据与书籍 "R"

通过示例进行统计应用开发。R 包版本

1.0\. [`CRAN.R-project.org/package=RSADBE`](https://CRAN.R-project.org/package=RSADBE)

Therneau T (2015). _S 生存分析包 _ 版本

2.38, <URL: [`CRAN.R-project.org/package=survival`](https://CRAN.R-project.org/package=survival)>. `survival`

Terry M. Therneau 和 Patricia M. Grambsch (2000). _ 生存分析建模 _

数据：扩展 Cox 模型。Springer，纽约。ISBN

0-387-98784-3.

Tianqi Chen, Tong He, Michael Benesty, Vadim Khotilovich 和 Yuan

Tang (2018). `xgboost`: 极端梯度提升。R 包版本

0.6.4.1\. [`CRAN.R-project.org/package=xgboost`](https://CRAN.R-project.org/package=xgboost)
