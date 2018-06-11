# Tree

Find below the detailed organization of the project. 

```
.
├── Carte_Identite_rectoverso.pdf
├── data
│   ├── holidays.csv
│   ├── metadata.csv
│   ├── submission_format.csv
│   ├── submission_frequency.csv
│   ├── train.csv
│   └── weather.csv
├── DrivenData - Competition Winner Documentation.docx
├── electricity-prediction-machine-learning.Rproj
├── output
├── README.md
└── src
    ├── install_packages.R
    ├── main.R
    ├── model_presentation.html
    ├── model_presentation.Rmd
    └── utilities.R
```

# Local workstation

Information about packages versions and OS. 

It ran on my Mac with only 16 Go of RAM but heavily used swap. 

```
─ Session info ─────────────────────────────────────────────────────────────────────────────────────────────────────
 setting  value                       
 version  R version 3.4.2 (2017-09-28)
 os       macOS High Sierra 10.13.1   
 system   x86_64, darwin15.6.0        
 ui       RStudio                     
 language (EN)                        
 collate  en_US.UTF-8                 
 tz       Europe/Paris                
 date     2018-04-04                  

─ Packages ─────────────────────────────────────────────────────────────────────────────────────────────────────────
 package     * version  date       source                              
 assertthat    0.2.0    2017-04-11 CRAN (R 3.4.0)                      
 bindr         0.1      2016-11-13 CRAN (R 3.4.0)                      
 bindrcpp      0.2      2017-06-17 CRAN (R 3.4.0)                      
 broom         0.4.2    2017-02-13 CRAN (R 3.4.0)                      
 cellranger    1.1.0    2016-07-27 CRAN (R 3.4.0)                      
 clisymbols    1.2.0    2017-05-21 CRAN (R 3.4.0)                      
 colorspace    1.3-2    2016-12-14 CRAN (R 3.4.0)                      
 data.table  * 1.10.4-3 2017-10-27 CRAN (R 3.4.2)                      
 dplyr       * 0.7.4    2017-09-28 CRAN (R 3.4.2)                      
 dtplyr      * 0.0.2    2017-04-21 CRAN (R 3.4.0)                      
 forcats       0.2.0    2017-01-23 CRAN (R 3.4.0)                      
 foreign       0.8-69   2017-06-22 CRAN (R 3.4.2)                      
 ggplot2     * 2.2.1    2016-12-30 CRAN (R 3.4.0)                      
 glue          1.2.0    2017-10-29 CRAN (R 3.4.2)                      
 gtable        0.2.0    2016-02-26 CRAN (R 3.4.0)                      
 haven         1.1.0    2017-07-09 CRAN (R 3.4.1)                      
 hms           0.3      2016-11-22 CRAN (R 3.4.0)                      
 httr          1.3.1    2017-08-20 CRAN (R 3.4.1)                      
 jsonlite      1.5      2017-06-01 CRAN (R 3.4.0)                      
 lattice       0.20-35  2017-03-25 CRAN (R 3.4.2)                      
 lazyeval      0.2.1    2017-10-29 CRAN (R 3.4.2)                      
 lightgbm    * 2.0.12   2017-12-27 local                               
 lubridate   * 1.7.0    2017-10-29 CRAN (R 3.4.2)                      
 magrittr      1.5      2014-11-22 CRAN (R 3.4.0)                      
 mnormt        1.5-5    2016-10-15 CRAN (R 3.4.0)                      
 modelr        0.1.1    2017-07-24 CRAN (R 3.4.1)                      
 munsell       0.4.3    2016-02-13 CRAN (R 3.4.0)                      
 nlme          3.1-131  2017-02-06 CRAN (R 3.4.2)                      
 pkgconfig     2.0.1    2017-03-21 CRAN (R 3.4.0)                      
 plyr          1.8.4    2016-06-08 CRAN (R 3.4.0)                      
 psych         1.7.8    2017-09-09 CRAN (R 3.4.2)                      
 purrr       * 0.2.4    2017-10-18 CRAN (R 3.4.2)                      
 R6          * 2.2.2    2017-06-17 CRAN (R 3.4.0)                      
 Rcpp          0.12.15  2018-01-20 cran (@0.12.15)                     
 RcppRoll    * 0.2.3    2018-01-07 Github (kevinushey/RcppRoll@90c0ed5)
 readr       * 1.1.1    2017-05-16 CRAN (R 3.4.0)                      
 readxl        1.0.0    2017-04-18 CRAN (R 3.4.0)                      
 reshape2      1.4.2    2016-10-22 CRAN (R 3.4.0)                      
 rlang         0.1.4    2017-11-05 cran (@0.1.4)                       
 rvest         0.3.2    2016-06-17 CRAN (R 3.4.0)                      
 scales        0.5.0    2017-08-24 CRAN (R 3.4.1)                      
 sessioninfo   1.0.0    2017-06-21 CRAN (R 3.4.1)                      
 stringi       1.1.5    2017-04-07 CRAN (R 3.4.0)                      
 stringr     * 1.2.0    2017-02-18 CRAN (R 3.4.0)                      
 tibble      * 1.3.4    2017-08-22 CRAN (R 3.4.1)                      
 tidyr       * 0.7.2    2017-10-16 CRAN (R 3.4.2)                      
 tidyverse   * 1.1.1    2017-01-27 CRAN (R 3.4.0)                      
 withr         2.0.0    2017-07-28 CRAN (R 3.4.1)                      
 xml2          1.1.1    2017-01-24 CRAN (R 3.4.0)                      
 yaml          2.1.16   2017-12-12 cran (@2.1.16) 
```

# GCP Ubuntu 16.04

I was able to replicated my results on Google Cloud Platform on an instance with 60 Go of RAM, which is probably too much. 

Below are instruction how to make everything work on GCP with Ubuntu 16.04. 

Install R

```
sudo echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | sudo tee -a /etc/apt/sources.list
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | sudo apt-key add -
sudo apt-get update
sudo apt-get install r-base r-base-dev
```

Install Rstudio server

```
sudo apt-get install gdebi-core
wget https://download2.rstudio.org/rstudio-server-1.1.447-amd64.deb
sudo gdebi rstudio-server-1.1.447-amd64.deb


```

Install packages needed by R packages.

```
sudo apt-get install libxml2 -y
sudo apt-get install libxml2-dev -y
sudo apt-get install libcurl4-openssl-dev -y
sudo apt-get install libssl-dev -y
sudo apt-get install git
sudo apt-get install cmake

```

Change password to ubuntu user you're curently using (the one when connecting to GCP with ssh)
```
sudo passwd USER
```

Transfer this repository zipped to GCP

```
gcloud compute scp electricity-prediction-machine-learning.tar.gz instance-name:. --recurse
```

Unzip
```
tar -xzf electricity-prediction-machine-learning.tar.gz
```

Connect to Rstudio now. 

# Script execution

Scripts are expected to be run from the root of the project which is naturally defined with file `electricity-prediction-machine-learning.Rproj` . 

Simply run `src/main.R` script. It will generate a file `/output/submission_YYYY-MM-DD-HH-MM-SS.csv.zip` which contains the final submission. 




