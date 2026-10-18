#Ponti et al. Island birds evolve towards flightless and terrestrial forms

## Load packages
if(!require(devtools)) install.packages("devtools")
library(devtools)
library(mulTree)
library(ggplot2)
library(tibble)
library(stringr) 
library(MCMCglmm)
library(coda)
library(hdrcde)
library(ape)
library(corpcor)
library(curl)
library(dplyr)
library(avotrex)

#set working directory
setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/Manuscript/Data availability")


#1. import database
db <- read.csv("morpho_db.csv", sep = ";", row.names = NULL, header = T)

#2.import trees. #do not do this, read direcly the trees_ex.nex
tree2 <- read.nexus("tree_all.nex") #import 100 random trees from Jetz et al.

data(BirdTree_tax) #read the extinct birds taxonomy
data(AvotrexPhylo) #read the Avotrex phylogeny
trees <- AvoPhylo(ctrees = tree2,   # this function adds the extinct species from AVOTREX to my set of trees (tree2)
                  avotrex = AvotrexPhylo,    # The extinct species phylogeny database
                  PER = 0.2,                 # Perecentage/fraction for branch truncation 
                  PER_FIXED = 0.75,          # Point along the branch to graft the species 
                  tax = BirdTree_tax,        # Taxonomy
                  Ntree = 100,                 # Number of trees
                  n.cores = 2)


write.nexus(trees, file = "C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/Manuscript/Global Ecology and Biogeography/Review 1/trees_ex.nex")

#read directly without doing the previous steps
tree <- read.nexus("trees_ex.nex") #this trees have the extinct taxa.
sp <- unique(db$phy_sp) #select unique values for the column phy_sp
sp <-na.omit(sp) #omit na (if there are)
tree <- keep.tip(tree, sp) # drop tip from phylo: remove the species not present in the dataset

#3.convert the variables
db$averaged.nr_species_same_family<- as.numeric(db$averaged.nr_species_same_family)
db$averaged.predator_richness<- as.numeric(db$averaged.predator_richness)
db$Hand.wing.Index<- as.numeric(db$Hand.wing.Index)
db$Wing.Length <- as.numeric(db$Wing.Length)
db$Tarsus.Length <- as.numeric(db$Tarsus.Length)
db$BM <- as.numeric(db$BM)
db$latitud1 <- scale(abs(db$latitud))
db$IMb <- as.factor(db$IMb)
db$BMscale <- scale(db$BM)
db$HWIlg <- log(db$Hand.wing.Index+0.1)
db$winglg <- log(db$Wing.Length+0.1)
db$tarsuslg <- log(db$Tarsus.Length+0.1)
db$competitors <-scale(db$averaged.nr_species_same_family)
db$predators <-scale(db$averaged.predator_richness)



##################################### 
###                               ###
###     Models with all birds     ###
###                               ###
#####################################

#########################################################
##Simple models with Latitude, Sex and Insularity + BM ##
#########################################################

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution")

## HWI model1
#------------

db <- db %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.))) #filter the lines with NAs
test_formula <- HWIlg ~ IMb + latitud1 + BMscale #model of the formula


mulTree.data <- as.mulTree(db, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID) #multree object with database, tree, taxa= the column in which there are the species names matching with the tree, and random effects
mcmc_priors <- list(R = list(V = 1, nu = 0.002), #define the priors, R for fixed terms, and G for random terms (in our case we have 3 random terms)
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))
mcmc_parameters <- c(640000, 500, 40000) #c(run, thinning, burn-in)

hwm.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors, #run the multree model
                     output = "./MCMCmodels/GEB/HWI/HWI1_all", ESS = 1000)

#results
setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI")

all_models <- read.mulTree("HWI1_all") #read the multree objects
summarised_results <- summary(all_models, use.hdr = FALSE, cent.tend = mean, #summarize all the models for HWI1 (200 models)
                              prob = c(95))
plot(summarised_results, horizontal = TRUE, ylab = "", cex.coeff = 0.8, #plot the estimates of the model with the CI intervals
     main = "Posterior distributions", ylim = c(-0.2,1), cex.terms = 0.5,
     terms = c("Intercept", "Insularity", "latitud", "BM", "phylogenetic.variance", "residual.variance"),
     col = "grey", cex.main = 0.8)
abline(v = 0, lty = 3)

####check models (repeat for each one and all following models)
one_model <- read.mulTree("HWI1_all-tree1_chain1", model = TRUE) #read model 1 for HWI, repeat it with all the models
one_model2 <- read.mulTree("HWI1_all-tree1_chain2", model = TRUE)
plot(one_model$Sol)
plot(one_model$VCV)
autocorr.diag(one_model$Sol) #checking the autocorrelation of the fixed factors
autocorr.diag(one_model$VCV) #checking the autocorrelation of the random factors

effectiveSize(one_model$Sol) #Checking the effect size of the fixed factors. the effect size should be between 1000 & 2000
effectiveSize(one_model$VCV) #Checking the effect size of the random factors.


gelman.diag(mcmc.list(one_model$Sol, one_model2$Sol)) # Checking convergence for our fixed factors # good values near 1
gelman.diag(mcmc.list(one_model$VCV, one_model2$VCV)) # Checking convergence for our random terms

h <- ( var(one_model$VCV[,"animal"]))/     #heritability
  (var(one_model$VCV[,"animal"]) + var(one_model$VCV[,"units"]))



## Wing length model1 
#---------------------

db <- db %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))
test_formula <- winglg ~ IMb + latitud1 + BMscale

mulTree.data <- as.mulTree(db, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))
mcmc_parameters <- c(640000, 500, 40000)

wing.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                      output = "./MCMCmodels/GEB/WING2/WING1_all", ESS = 1000)

all_models <- read.mulTree("WING1_all")
summarised_results <- summary(all_models, use.hdr = FALSE, cent.tend = mean,
                              prob = c(95))


## Tarsus length model1
#----------------------

db <- db %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))
test_formula <- tarsuslg ~ IMb + latitud1 + BMscale

mulTree.data <- as.mulTree(db, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))
mcmc_parameters <- c(640000, 500, 40000)

tarsus.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                        output = "./MCMCmodels/GEB/TARSUS/TARSUS1_all", ESS = 1000)

all_models <- read.mulTree("TARSUS1")
summarised_results <- summary(all_models, use.hdr = FALSE, cent.tend = mean,
                              prob = c(95))



################################################################################################
##Predators and competitors models with Latitude, and Insularity + BM ##
################################################################################################

## HWI model2
#---------------

db <- db %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))

mulTree.data <- as.mulTree(db, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))

#mcmc_parameters <- c(101000, 10, 1000)
mcmc_parameters <- c(640000, 500, 40000)


test_formula <- HWIlg ~ IMb + latitud1 +  BMscale + IMb*predators


hwm.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                     output = "./MCMCmodels/GEB/HWI/HWI2_all", ESS = 1000)

all_models <- read.mulTree("HWI2")
summarised_results <- summary(all_models, use.hdr = FALSE, cent.tend = mean,
                              prob = c(95))

## Tarsus model2
#---------------

db <- db %>% filter_at(vars(BMscale, latitud1, competitors, predators),all_vars(!is.na(.)))

mulTree.data <- as.mulTree(db, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))

#mcmc_parameters <- c(101000, 10, 1000)
#mcmc_parameters <- c(240000, 100, 4000)
mcmc_parameters <- c(640000, 500, 40000)


test_formula <- tarsuslg ~ IMb + latitud1 + BMscale + IMb*competitors + IMb*predators


tarsus.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                        output = "./MCMCmodels/GEB/TARSUS/TARSUS2_all", ESS = 1000)

all_models <- read.mulTree("TARSUS2_all")
summarised_results <- summary(all_models, use.hdr = FALSE, cent.tend = mean,
                              prob = c(95))




##################################### 
###                               ###
## Models with only passeriformes ###
###                               ###
#####################################

#load database and tree

db <- read.csv("morpho_db.csv", sep = ";", row.names = NULL, header = T)
db_passer <- db %>% filter(Order == "Passeriformes") #filter the database and select only the individuals within Order Passeriformes

db_passer$latitud1 <- scale(abs(db_passer$latitud))
db_passer$IMb <- as.factor(db_passer$IMb)
db_passer$BMscale <- scale(db_passer$BM)
db_passer$HWIlg <- log(db_passer$Hand.wing.Index+0.1)
db_passer$winglg <- log(db_passer$Wing.Length+0.1)
db_passer$tarsuslg <- log(db_passer$Tarsus.Length+0.1)
db_passer$competitors <-scale(db_passer$averaged.nr_species_same_family)
db_passer$predators <-scale(db_passer$averaged.predator_richness)


#read directly without doing the previous steps
tree <- read.nexus("trees_ex.nex") #this trees have the extinct taxa.
sp <- unique(db$phy_sp)
sp <-na.omit(sp)
tree <- keep.tip(tree, sp) # drop tip from phylo: remove the species not present in the dataset


#########################################################
##Simple models with Latitude, Sex and Insularity + BM ##
#########################################################

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution")

## HWI model1
#------------

db_passer <- db_passer %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))
test_formula <- HWIlg ~ IMb + latitud1 + BMscale 


mulTree.data <- as.mulTree(db_passer, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))
mcmc_parameters <- c(640000, 500, 40000)

hwm.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                     output = "./MCMCmodels/GEB/HWI_passer/HWI1_passer", ESS = 1000)

all_models_hwi_p <- read.mulTree("HWM1_passer")
summarised_results_hwi_p <- summary(all_models_hwi_p, use.hdr = FALSE, cent.tend = mean,
                                    prob = c(95))


## WING model1
#-------------

db_passer <- db_passer %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))
test_formula <- winglg ~ IMb + latitud1 + BMscale 


mulTree.data <- as.mulTree(db_passer, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))
mcmc_parameters <- c(640000, 500, 40000)


wgp1.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                      output = "./MCMCmodels/GEB/WING_passer/Wing1_passer", ESS = 1000)

all_models_hwi_p <- read.mulTree("Wing1_passer")
summarised_results_wing_p <- summary(all_models_hwi_p, use.hdr = FALSE, cent.tend = mean,
                                     prob = c(95))


## Tarsus model1
#---------------

db_passer <- db_passer %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))
test_formula <- tarsuslg ~ IMb + latitud1 + BMscale 


mulTree.data <- as.mulTree(db_passer, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))
mcmc_parameters <- c(640000, 500, 40000)

hwm.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                     output = "./MCMCmodels/GEB/TARSUS_passer/Tarsus1_passer", ESS = 50)

all_models_tar_p <- read.mulTree("Tarsus1_passer")
summarised_results_t_p <- summary(all_models_hwi_p, use.hdr = FALSE, cent.tend = mean,
                                  prob = c(95))


################################################################################################
##Predators and competitors as CONTINUOUS variables models with Latitude, and Insularity + BM ##
################################################################################################

## HWI model2
#---------------

db_passer <- db_passer %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))

mulTree.data <- as.mulTree(db_passer, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))

#mcmc_parameters <- c(101000, 10, 1000)

mcmc_parameters <- c(640000, 500, 40000)


test_formula <- HWIlg ~ IMb + latitud1 +  BMscale + IMb*predators


hwm.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                     output = "./MCMCmodels/GEB/HWI_passer/HWI2_passer", ESS = 1000)

all_models <- read.mulTree("HWI2_passer")
summarised_results <- summary(all_models, use.hdr = FALSE, cent.tend = mean,
                              prob = c(95))

## Tarsus model2
#---------------

db_passer <- db_passer %>% filter_at(vars(BMscale, latitud1, competitors, predators),all_vars(!is.na(.)))

mulTree.data <- as.mulTree(db_passer, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))

#mcmc_parameters <- c(101000, 10, 1000)
#mcmc_parameters <- c(240000, 100, 4000)
mcmc_parameters <- c(640000, 500, 40000)


test_formula <- tarsuslg ~ IMb + latitud1 + BMscale + IMb*competitors + IMb*predators

tarsus.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                        output = "./MCMCmodels/GEB/TARSUS_passer/TARSUS2_passer", ESS = 1000)

all_models <- read.mulTree("TARSUS2_passer")
summarised_results <- summary(all_models, use.hdr = FALSE, cent.tend = mean,
                              prob = c(95))




################################################
###                                          ###
## Models with non-repeated mainland species ###
###                                          ###
###############################################


#load database and tree

db_m <- read.csv("morpho_db_filtered.csv", sep = ";", row.names = NULL, header = T) #new database with no mainland species repeated


#3.convert the variables
db_m$averaged.nr_species_same_family<- as.numeric(db_m$averaged.nr_species_same_family)
db_m$averaged.predator_richness<- as.numeric(db_m$averaged.predator_richness)
db_m$Hand.wing.Index<- as.numeric(db_m$Hand.wing.Index)
db_m$Wing.Length <- as.numeric(db_m$Wing.Length)
db_m$Tarsus.Length <- as.numeric(db_m$Tarsus.Length)
db_m$BM <- as.numeric(db_m$BM)
db_m$latitud1 <- scale(abs(db_m$latitud))
db_m$IM <- as.factor(db_m$IM)
db_m$IMb <- as.factor(db_m$IMb)
db_m$BMscale <- scale(db_m$BM)
db_m$HWIlg <- log(db_m$Hand.wing.Index+0.1)
db_m$winglg <- log(db_m$Wing.Length+0.1)
db_m$tarsuslg <- log(db_m$Tarsus.Length+0.1)
db_m$competitors <-scale(db_m$averaged.nr_species_same_family)
db_m$predators <-scale(db_m$averaged.predator_richness)


#read directly without doing the previous steps
tree <- read.nexus("trees_ex.nex") #this trees have the extinct taxa.
sp <- unique(db_m$phy_sp)
sp <-na.omit(sp)
tree <- keep.tip(tree, sp) # drop tip from phylo: remove the species not present in the dataset


#########################################################
##Simple models with Latitude, Sex and Insularity + BM ##
#########################################################

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution")

## HWI model1
#------------

db_m <- db_m %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))
test_formula <- HWIlg ~ IMb + latitud1 + BMscale 


mulTree.data <- as.mulTree(db_m, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))
mcmc_parameters <- c(640000, 500, 40000)

hwm.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                     output = "./MCMCmodels/GEB/HWI_no_rep_main/HWI1_nrp", ESS = 1000)

all_models_hwi_p <- read.mulTree("HWM1_nrp")
summarised_results_hwi_p <- summary(all_models_hwi_p, use.hdr = FALSE, cent.tend = mean,
                                    prob = c(95))


## WING model1
#-------------

db_m <- db_m %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))
test_formula <- winglg ~ IMb + latitud1 + BMscale 


mulTree.data <- as.mulTree(db_m, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))
mcmc_parameters <- c(640000, 500, 40000)


wgp1.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                      output = "./MCMCmodels/GEB/WING_no_rep_main/Wing1_nrp", ESS = 1000)

all_models_hwi_p <- read.mulTree("Wing1_nrp")
summarised_results_wing_p <- summary(all_models_hwi_p, use.hdr = FALSE, cent.tend = mean,
                                     prob = c(95))


## Tarsus model1
#---------------

db_m <- db_m %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))
test_formula <- tarsuslg ~ IMb + latitud1 + BMscale 


mulTree.data <- as.mulTree(db_m, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))
mcmc_parameters <- c(640000, 500, 40000)

hwm.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                     output = "./MCMCmodels/GEB/TARSUS_no_rep_main/Tarsus1_nrp", ESS = 50)

all_models_tar_p <- read.mulTree("Tarsus1_nrp")
summarised_results_t_p <- summary(all_models_hwi_p, use.hdr = FALSE, cent.tend = mean,
                                  prob = c(95))


################################################################################################
##Predators and competitors as CONTINUOUS variables models with Latitude, and Insularity + BM ##
################################################################################################

## HWI model2
#---------------

db_m <- db_m %>% filter_at(vars(BMscale, latitud1),all_vars(!is.na(.)))

mulTree.data <- as.mulTree(db_m, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))

#mcmc_parameters <- c(101000, 10, 1000)

mcmc_parameters <- c(640000, 500, 40000)


test_formula <- HWIlg ~ IMb + latitud1 +  BMscale + IMb*predators


hwm.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                     output = "./MCMCmodels/GEB/HWI_no_rep_main/HWI2_nrp", ESS = 1000)

all_models <- read.mulTree("HWI2_nrp")
summarised_results <- summary(all_models, use.hdr = FALSE, cent.tend = mean,
                              prob = c(95))

## Tarsus model2
#---------------

db_m <- db_m %>% filter_at(vars(BMscale, latitud1, competitors, predators),all_vars(!is.na(.)))

mulTree.data <- as.mulTree(db_m, tree, taxa = "phy_sp", rand.terms = ~phy_sp+pairID+avibase.ID)
mcmc_priors <- list(R = list(V = 1, nu = 0.002),
                    G = list(G1 = list(V = 1, nu = 0.002), G2 = list(V = 1, nu = 0.002), G3 = list(V = 1, nu = 0.002)))

#mcmc_parameters <- c(101000, 10, 1000)
#mcmc_parameters <- c(240000, 100, 4000)
mcmc_parameters <- c(640000, 500, 40000)


test_formula <- tarsuslg ~ IMb + latitud1 + BMscale + IMb*competitors + IMb*predators

tarsus.model <- mulTree(mulTree.data, formula = test_formula, parameters = mcmc_parameters, priors = mcmc_priors,
                        output = "./MCMCmodels/GEB/TARSUS_no_rep_main/TARSUS2_nrp", ESS = 1000)

all_models <- read.mulTree("TARSUS2_nrp")
summarised_results <- summary(all_models, use.hdr = FALSE, cent.tend = mean,
                              prob = c(95))







######################################################################################################################

########################
##  Extract results   ##
########################


####list of models.Extract results

clean.MCMC <- function(x) { #this function allow to summarize MCMCglmm models' results in a table
  sols <- summary(x)$solutions  ## pull out relevant info from model summary
  Gcovs <- summary(x)$Gcovariances
  Rcovs <- summary(x)$Rcovariances
  fixed <- data.frame(row.names(sols), sols, row.names = NULL)  ## convert to dataframes with the row.names as the first col
  random <- data.frame(row.names(Gcovs), Gcovs, row.names = NULL)
  residual <- data.frame(row.names(Rcovs), Rcovs, row.names = NULL)
  names(fixed)[names(fixed) == "row.names.sols."] <- "variable"  ## change the columns names to variable, so they all match
  names(random)[names(random) == "row.names.Gcovs."] <- "variable"
  names(residual)[names(residual) == "row.names.Rcovs."] <- "variable"
  fixed$effect <- "fixed"  ## add ID column for type of effect (fixed, random, residual)
  random$effect <- "random"
  residual$effect <- "residual"
  modelTerms <- as.data.frame(bind_rows(fixed, random, residual))  # merge it all together
}


######MODELS WITH ALL BIRDS########

## HWI model1
#------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI", pattern= "HWI1_all*") #create a list of files within the directory which share the pattern HWI1_all
INFILES <- INFILES[grepl("chain",INFILES)] #select only the files with "chain" 

for (i in 1:length(INFILES)) { #for each item in INFILES
  m <- read.mulTree(INFILES[i], model = TRUE) #read the model multree
  m_clean <- clean.MCMC(m) #use the function clean.MCMC
  m_clean$model <- paste0("model",i) #paste the result in the table
  if (i == 1){ #if the result is the first one
    df_concat <- m_clean #include the result in the database df_concat
  } else { #if the result is not the first one
    df_concat <- rbind(df_concat, m_clean) #paste the result after the previous results 
  }
}

write.csv(df_concat, "hwi1_all_geb.csv")

res_hwi1_a <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## WING model1
#------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/WING")


INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/WING", pattern= "WING1_all*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}

write.csv(df_concat, "wing1_all_geb.csv")

res_wing1_a <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## TARSUS model1
#---------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS", pattern= "TARSUS1_all*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}

write.csv(df_concat, "tarsus1_all_geb.csv")

res_tarsus1_a <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## HWI model2
#------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI", pattern= "HWI2_all*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}

write.csv(df_concat, "hwi2_all_geb.csv")

res_hwi2_a <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


##TARSUS model2
#---------------
setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS", pattern= "TARSUS2_all*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}
write.csv(df_concat, "tarsus2_all_geb.csv")

res_tarsus2_a <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))



######MODELS WITH PASSERINES########


## HWI model1 passer
#-------------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI_passer")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI_passer", pattern= "HWI1_passer*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}
write.csv(df_concat, "hwi1_passer_geb.csv")

res_hwi1_p <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## WING model1 passer
#--------------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/WING_passer")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/WING_passer", pattern= "Wing1_passer*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}
write.csv(df_concat, "wing1_passer_geb.csv")

res_wing1_p <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## Tarsus model1 passer
#----------------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS_passer")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS_passer", pattern= "Tarsus1_passer*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}
write.csv(df_concat, "tarsus1_passer_geb.csv")

res_tarsus1_p <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## HWI model2 passer
#-------------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI_passer")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI_passer", pattern= "HWI2_passer*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}
write.csv(df_concat, "hwi2_passer_geb.csv")

res_hwi2_p <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## Tarsus model2 passer
#----------------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS_passer")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS_passer", pattern= "TARSUS2_passer*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}

write.csv(df_concat, "tarsus2_passer_geb.csv")

res_tarsus2_p <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


######MODELS WITH NON-REPEATED MAINLAND SPECIES########


## HWI model1 non-repeated mainland
#----------------------------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI_no_rep_main")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI_no_rep_main", pattern= "HWI1_nrp*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}
write.csv(df_concat, "hwi1_nrp_geb.csv")

res_hwi1_n <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## WING model1 non-repeated mainland
#-----------------------------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/WING_no_rep_main")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/WING_no_rep_main", pattern= "Wing1_nrp*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}
write.csv(df_concat, "wing1_nrp_geb.csv")

res_wing1_n <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## Tarsus model1 non-repeated mainland
#-------------------------------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS_no_rep_main")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS_no_rep_main", pattern= "Tarsus1_nrp*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}
write.csv(df_concat, "tarsus1_nrp_geb.csv")

res_tarsus1_n <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## HWI model2 non-repeated mainland
#----------------------------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI_no_rep_main")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/HWI_no_rep_main", pattern= "HWI2_nrp*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}
write.csv(df_concat, "hwi2_nrp_geb.csv")

res_hwi2_n <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))


## Tarsus model2 non-repeated mainland
#-------------------------------------

setwd("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS_no_rep_main")

INFILES <- list.files("C:/Science/POSTDOC ISLANDS/Island rule/2. Island morphology/2.1. Flightlessness evolution/MCMCmodels/GEB/TARSUS_no_rep_main", pattern= "TARSUS2_nrp*")
INFILES <- INFILES[grepl("chain",INFILES)]

for (i in 1:length(INFILES)) {
  m <- read.mulTree(INFILES[i], model = TRUE)
  m_clean <- clean.MCMC(m)
  m_clean$model <- paste0("model",i)
  if (i == 1){
    df_concat <- m_clean
  } else {
    df_concat <- rbind(df_concat, m_clean)
  }
}

write.csv(df_concat, "tarsus2_nrp_geb.csv")

res_tarsus2_n <- summarise(select(group_by(df_concat, variable), post.mean, l.95..CI, u.95..CI, eff.samp, pMCMC), Esimate = mean(post.mean, na.rm = TRUE), lower = mean(l.95..CI, na.rm = TRUE), upper = mean(u.95..CI, na.rm = TRUE), effsample = mean(eff.samp, na.rm = TRUE), pMCMC = mean(pMCMC, na.rm = TRUE))



#####Calculate R2

# pseudoR2 function for MCMCglmm models

pseudoR2.MCMCglmm <- function(mod, data){
  # create model matrix using the MCMCglmm model and data objects
  dt <- model.matrix(summary(mod)$fixed.formula, data=data)
  # set logical condition to identify "Intercept" column in model matrix
  cond <- grep("Intercept", colnames(dt))
  # if "Intercept" exists, then subset the remaining columns without dropping dimensions for a one-column matrix
  # this will create a one-column matrix instead of a vector
  if(sum(cond)>0){dt <- dt[,-cond, drop=F]}
  
  # Identify variable names making sure they match the output
  cols <- colnames(mod$Sol)
  # set condition where colnames in mod$Sol match those in model matrix
  cond <- is.element(cols, colnames(dt))
  # subset mod$Sol to only columns matching those in model matrix
  sol <- mod$Sol[,cond, drop=F]
  # do the oposite: subset model matrix where colnames match colnames in mod$Sol
  cond <- is.element(colnames(dt), cols)
  dt <- dt[,cond, drop=F]
  
  # First calculate the variance associated with the fixed effects
  betas <- colMeans(sol)
  xs <- t(dt) * betas
  bx <- colSums(xs)
  sigf <- var(bx)
  
  # Then, calculate the variance associated with the random effects
  mod.vcv <- data.frame(mod$VCV[, 1:(ncol(mod$VCV)-1)])
  sigr <- sum(as.numeric(colMeans(mod.vcv)))
  
  # Now, calculate the variance associated with the residual effects
  sige <- mean(mod$VCV[,"units"])
  
  # marginal R2 (variance explained by fixed factors)
  r2m.i <- sigf / {sigf + sige + sigr}
  # conditional R2 (variance explained by the entire model)
  r2c.i <- {sigf + sigr} / {sigf + sige + sigr}
  r2 <- c(r.squared.marg=r2m.i, r.squared.cond=r2c.i)
  return(r2)
}




# Initialize a list to store the R2 results
r2_results <- list()
h_results <- list()
# Loop over the files
for (i in 1:length(INFILES)) {
  # Load the MCMCglmm model (assuming RDS format)
  m <- read.mulTree(INFILES[i], model = TRUE)
  
  # Assuming you have a corresponding dataset 'data' for each model
  # This could be modified if the dataset is different for each model
  r2 <- pseudoR2.MCMCglmm(m, db)
  
  # Store the results
  r2_results[[INFILES[i]]] <- r2
}

# Now, calculate the mean marginal and conditional R2 across all models
marginal_r2_mean <- mean(sapply(r2_results, function(x) x['r.squared.marg']))
conditional_r2_mean <- mean(sapply(r2_results, function(x) x['r.squared.cond']))

# Loop over the files
for (i in 1:length(INFILES)) {
  # Load the MCMCglmm model (assuming RDS format)
  m <- read.mulTree(INFILES[i], model = TRUE)
  h <- ( var(m$VCV[,"animal"]))/     #heritability
  (var(m$VCV[,"animal"]) + var(m$VCV[,"units"]))
  # Store the results
  h_results[[INFILES[i]]] <- h
}

mean_h <- mean(unlist(h_results))


#Calculate DIC statistics


# Initialize an empty vector to store DIC values
dic_values <- numeric(length(INFILES))

# Load models and extract DIC
for (i in 1:length(INFILES)) {
  # Load the MCMCglmm model from the file
  model <- read.mulTree(INFILES[i], model = TRUE)
  
  # Extract DIC value from the model summary
  dic_values[i] <- model$DIC
}

# Calculate the mean DIC
mean_dic <- mean(dic_values)

# Calculate 95% confidence interval (using quantiles for simplicity)
ci_dic <- quantile(dic_values, probs = c(0.25, 0.95))
                