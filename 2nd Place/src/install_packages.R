# install openssl and stuff here
cran_packages = c("httr","rvest","ggplot2", "data.table", "Metrics", "tidyverse", "dtplyr", "lubridate", "devtools")
install.packages(cran_packages)

devtools::install_github("kevinushey/RcppRoll")
# install cmake first and git
devtools::install_github("Laurae2/lgbdl")
lgbdl::lgb.dl()
