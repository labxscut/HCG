
# Identifying gastric cancer molecular subtypes by integrating DNA-based hierarchical classification strategy and clinical stratification
*Binyu Yang, Siying Liu, Jiemin Xie, Xi Tang, Pan Guan, Yifan Zhu, Li C. Xia**

## Summary
We develop an accurate, robust, and easily adoptable subtyping classifier for gastric cancer (GC) molecular subtypes.It leveraged all DNA-level alterations(including gene mutations,copy number aberrations and methylations) as predictors to delineate specific molecular subtypes of GC. The classifier optimize with hierarchical classification strategy and Lasso-Logistic regression, and benchmark for subtype prediction performance and clinical stratification capacity. The prediction performance is assessed by AUC values and the clinical stratification capacity is assessed by multivariate survival analysis.

The repository contains all the datasets and code used for training and evaluating the classifiers, as well as important findings (please refer to the figures).

# Data:
# The data for HCG

The original data are available in  cBioPortal database (https://www.cbioportal.org/)

· ① Stomach Adenocarcinoma (TCGA, PanCancer Atlas)(https://www.cbioportal.org/study/summary?id=stad_tcga_pan_can_atlas_2018)

· ② Stomach Adenocarcinoma (TCGA, Nature 2014)(https://www.cbioportal.org/study/summary?id=stad_tcga_pub)

· ③ Esophageal Adenocarcinoma (TCGA, PanCancer Atlas)(https://www.cbioportal.org/study/summary?id=esca_tcga_pan_can_atlas_2018)

The datasets downloaded from cbioportal and the pre-processing data are restored in https://github.com/labxscut/HCG/releases or in the latest release named HCG at https://github.com/labxscut/UGES. The readers can download them directly.

## Brief description for the data and their potential use

The dataset ① and ② are two large-scale multi-omics datasets of gastric cancer patients, containing the genetic, epigenetic, expression and clinical data, etc. The dataset  ③ is  a dataset of esophageal Adenocarcinoma patients,  which shares similar data types and structures with datasets ① and ②.The dataset ① actually is an additional dataset from another TCGA study published in 2018, as the supplementary data of  dataset ②.

In HCG, we used these large-scale multi-omics data as the original data. <same_esca_data.csv> and <same_esca_data.csv> are the pro-precessing data. <survival.csv> is data used in survival analysis. They provided us the sufficient condition to do the analysis.

## Data types

In HCG, the data are:

· The alterations of multi-omics data, including gene mutations, copy number abberations, methylation alterations, etc.

· The clinical data of samples, including subtype, age, sex, patient ID, etc.

## Estimate of dataset size

After pre-processing, we have the data for 453 patients.
The required dependency is R and Python.

The code is contained in https://github.com/labxscut/HCG/blob/main/code.md

The data details are shown in https://github.com/labxscut/HCG/tree/main/Data


# Contact & Support:

* Li C. Xia: email: [lcxia@scut.edu.cn](mailto:lcxia@scut.edu.cn)
* Jiemin Xie: email: [202120130808@mail.scut.edu.cn](mailto:202120130808@mail.scut.edu.cn)
* Binyu Yang: email: [201930051012@mail.scut.edu.cn](mailto:201930051012@mail.scut.edu.cn)
* Siying Liu: email: [202030321135@mail.scut.edu.cn](mailto:202030321135@mail.scut.edu.cn)



