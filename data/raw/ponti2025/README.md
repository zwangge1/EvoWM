# Dryad dataset

Dataset DOI: [10.5061/dryad.1zcrjdg57](10.5061/dryad.1zcrjdg57)

## Description of the data and file structure

**trees_ex.nex** file contains 100 alternative Ericson tree topologies obtained from Jetz et al. 2012 after adding extinct species with AVOTREX function.

**morpho_db.csv** file contains all the species with morphological data used in the analyses.

**morpho_db_filtered.csv** file contains only pairs with non-repeated mainland species

See below for further description:

### Files and variables

**trees_ex.nex** file contains 100 alternative Ericson tree topologies obtained from Jetz et al. 2012 after adding extinct species with AVOTREX function.

**morpho_db.csv** and **morpho_db_filtered.csv** file contains all the morphological data used in the analyses. See below for further description:

NEW2_sequence: unique number for each species. Repeated mainland species have a different number in each pair

pairID: number of the pair

Museum: museum where the morphological measurements were collected. NA (not available) correspond to missing information.

Specimen ID: unique ID number for each specimen (museum information). NA (not available) correspond to missing information.

Sex: sex (F: Females/ M: Males). NA (not available) correspond to missing information.

species name birdlife: species name following Birdlife nomenclature

species name ebird clements: species name following Clements nomenclature

phy_sp: name of the species referred in Jetz et al. (2012) phylogeny: BIRDTREE

avibase.ID: avibase ID

IMb: island (1I) or mainland (0M) distribution

Order: order to which the the species belong to

Family: family to which the the species belong to

Location: species distribution

averaged predator_richness: number of raptors of suitable mass accross the range of the focal species in the island or in the mainland. This number was corrected by calculating the number of overlapping cells divided by the total cells of the focal species

averaged nr_species_same_family: number of species belonging to the same family as the focal species in the island or in the mainland. This number was corrected by calculating the number of overlapping cells divided by the total cells of the focal species

latitud: latitude

Migratory/sedentary: sedentary (0), migratory (1) or partial migratory (2) status

source_BM: source of body mass information. NA (not available) correspond to missing information.

InfoBM: body mass information obtained, meanB (mean of the whole species), meanF (mean of the species for females), meanM (mean of the species for males), specimen (body mass of the specimen). NA (not available) correspond to missing information.

BM: body mass in grams (g). NA (not available) correspond to missing information.

source_morph: source of morphological traits information

Tarsus.Length: tarsus length (mm). NA (not available) correspond to missing information.

Wing.Length: wing length (mm). NA (not available) correspond to missing information.

Secondary1: wing secondary length (mm). NA (not available) correspond to missing information.

Hand-wing.Index: hand-wing index defined as HWI=100*(WL-SL)/WL, where WL is wing length and SL secondary length. NA (not available) correspond to missing information.

## Code/software

analyses_morpho.R : R code

**References:**

Jetz, W., Thomas, G. H., Joy, J. B., Hartmann, K., & Mooers, A. O. (2012). The global diversity of birds in space and time. *Nature*, *491*(7424), 444-448.
