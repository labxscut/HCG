
# Identifying gastric cancer molecular subtypes by integrating DNA-based hierarchical classification strategy and clinical stratification
*Binyu Yang, Siying Liu, Jiemin Xie, Xi Tang, Pan Guan, Yifan Zhu, Li C. Xia**

## Summary
We develop an accurate, robust, and easily adoptable subtyping classifier (HCG) for gastric cancer (GC) molecular subtypes. It leveraged all DNA-level alterations(including gene mutations,copy number aberrations and methylations) as predictors to delineate specific molecular subtypes of GC. The classifier optimize with hierarchical classification strategy and Lasso-Logistic regression, and benchmark for subtype prediction performance and clinical stratification capacity. The prediction performance is assessed by AUC values and the clinical stratification capacity is assessed by multivariate survival analysis.

The repository contains all the data (https://github.com/labxscut/HCG/tree/main/Data) and code (https://github.com/labxscut/HCG/blob/main/code.md) used for training and evaluating the classifiers, as well as some important results (please refer to the figures(https://github.com/labxscut/HCG/tree/main/Figures)).

![Framework](https://github.com/labxscut/HCG/blob/main/Figures/Framework.jpg?raw=true)

# Data 

## Original data 

The original data are available in  cBioPortal database (https://www.cbioportal.org/)

* ① Stomach Adenocarcinoma (TCGA, PanCancer Atlas)(https://www.cbioportal.org/study/summary?id=stad_tcga_pan_can_atlas_2018)

* ② Stomach Adenocarcinoma (TCGA, Nature 2014)(https://www.cbioportal.org/study/summary?id=stad_tcga_pub)

* ③ Esophageal Adenocarcinoma (TCGA, PanCancer Atlas)(https://www.cbioportal.org/study/summary?id=esca_tcga_pan_can_atlas_2018)

The dataset ① and ② are two large-scale multi-omics datasets of gastric cancer patients, containing the genetic, epigenetic, expression and clinical data, etc. The dataset  ③ is  a dataset of esophageal Adenocarcinoma patients,  which shares similar data types and structures with datasets ① and ②.The dataset ① actually is an additional dataset from the TCGA PanCanAtlas published in 2018, as the supplementary data of  dataset ②.  

The original data are restored in https://github.com/labxscut/HCG/releases . 

* **[stad_tcga_pan_can_atlas_2018.tar.gz](https://github.com/labxscut/HCG/releases/download/HCG/stad_tcga_pan_can_atlas_2018.tar.gz)** is the  dataset ①

* **[stad_tcga_pub.tar.gz](https://github.com/labxscut/HCG/releases/download/HCG/stad_tcga_pub.tar.gz)** is the  dataset ②

* **[esca_tcga_pan_can_atlas_2018.tar.gz](https://github.com/labxscut/HCG/releases/download/HCG/esca_tcga_pan_can_atlas_2018.tar.gz)** is the  dataset ③ 

## Pre-processing the data and their use

We used the alterations of multi-omics data, including gene mutations, copy number aberrations, methylation alterations and the clinical data of samples, including subtype, age, sex, patient ID, etc. in our analysis. The pre-processing data are restored in https://github.com/labxscut/HCG/releases . The readers can download them directly. We used data  augmentation technique called Synthetic Minority Oversampling Technique (**SMOTE**) to balance the training set.

* **[SMOTE_train_data.csv](https://github.com/labxscut/HCG/releases/download/HCG/SMOTE_train_data.csv)** is the training set after data  augmentation

* **[test_data.csv](https://github.com/labxscut/HCG/releases/download/HCG/test_data.csv)**  is the test set

* **[survival_data.csv](https://github.com/labxscut/HCG/releases/download/HCG/survival_data.csv)**  is data used in survival analysis.

After pre-processing, we have the data for total  453 patients.

# Code

The dependencies required are R and Python. All of the code can be found  [here](https://github.com/labxscut/HCG/tree/main/Code).

The code base structure is explained below:

* **[Parameters_adjustment.py](https://github.com/labxscut/HCG/blob/main/Code/1.Parameters_adjustment.py)**: script for constructing classifiers and finding the optimal parameters using cross-validation.
* **[Evaluation_AUROC.py](https://github.com/labxscut/HCG/blob/main/Code/2.Evaluation_AUROC.py)**: script for plotting the ROC curve and calculating the AUC value.
* **[Survival_analysis.R](https://github.com/labxscut/HCG/blob/main/Code/3.Survival_analysis.R)**: script for multivariate analysis based on TCGA, I-MC(A), II-HC(A) and III-HC(A) defined subtypes.
* **[Alluvial_map.R](https://github.com/labxscut/HCG/blob/main/Code/4.Alluvial_map.R)**: script for plotting alluvial maps between TCGA subtypes and I-MC(A), HCG, III-HC(A) subtypes.
* **[Heatmap.R](https://github.com/labxscut/HCG/blob/main/Code/5.Heatmap.R)**: script for plotting Heatmap based on the genes identified by HCG classifier
* **[Difference_test.R](https://github.com/labxscut/HCG/blob/main/Code/6.Difference_test.R)**:  script for differential test for identifying significant DNA alterations 




# Contact & Support:

* Li C. Xia: email: [lcxia@scut.edu.cn](mailto:lcxia@scut.edu.cn)
* Jiemin Xie: email: [202120130808@mail.scut.edu.cn](mailto:202120130808@mail.scut.edu.cn)
* Binyu Yang: email: [201930051012@mail.scut.edu.cn](mailto:201930051012@mail.scut.edu.cn)
* Siying Liu: email: [202030321135@mail.scut.edu.cn](mailto:202030321135@mail.scut.edu.cn)



