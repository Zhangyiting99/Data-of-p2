
# C-A-S-H Material Simulation and Machine Learning Analysis

This repository contains the data, scripts, and models used in the study of C-A-S-H (Calcium-Aluminum-Silicate Hydrate) materials. It includes molecular simulations (MD) for nanoindentation testing, machine learning models for prediction, and statistical analysis of bond lengths and bond angles in silicon-oxygen and aluminum-oxygen tetrahedra.

## File Structure

### 1. Models Constructed in MS

1.1. The C-A-S-H coordinate files, generated using Materials Studio (MS), are provided in two formats: CIF and XYZ. Each format contains 117 models  
1.2. CIF files are stored in /C-A-S-H Coordinate File/CIF files and contain detailed crystallographic information  
1.3. XYZ files are stored in /C-A-S-H Coordinate File/XYZ files and include atomic coordinates used for molecular dynamics simulations and structural analysis

### 2. Nanoindentation Simulations in MD

#### 2.1 Input Data

2.1.1. Data files  
- The initial C-A-S-H models with varying Qn distributions are available in the /MD/Input/Data files folder  
- These models were created using Materials Studio and converted into data file format using the msi2lmp tool

2.1.2. Scripts for forcefield and nanoindentation  
- All relevant scripts are stored in the /MD/Input/scripts folder, including:  
  1. ffield.reax.choCaAlSi: Defines potential functions for reactive force field simulations  
  2. lmp_control: Contains the LAMMPS simulation control settings  
  3. param.qeq: Parameters for charge equilibration  
  4. in.equ: LAMMPS script for model equilibration  
  5. in.ind: Settings for the nanoindentation simulation

#### 2.2 Output Data

2.2.1. The output consists of 117 sets of trajectory and output files, each corresponding to a simulation group (1–117)  
2.2.2. All output files are stored in the /MD/Output folder, organized in subfolders by group  
2.2.3. Each group contains:  
  1. dump.*.relax.lammpstrj: Trajectory files from the equilibration process  
  2. dump.*.ind.lammpstrj: Trajectory files from the nanoindentation process  
  3. out.dat: Energy output file  
  4. slurm-*.out: Simulation log file

### 3. Training Models with Machine Learning

3.1. Machine learning datasets and scripts are located in the /ML folder  
3.2. Contents include:  
  1. p2data250312.csv: Dataset for ML training derived from MD nanoindentation results  
  2. MLR.py: Script for training the Multiple Linear Regression model  
  3. SVR.py: Script for training the Support Vector Regression model  
  4. RF.py: Script for training the Random Forest model  
  5. BPNN.py: Script for training the Backpropagation Neural Network model  
  6. final_bpnn_model.h5: The best-performing BPNN model in HDF5 format  
  7. SHAP.py: Script for SHAP (SHapley Additive exPlanations) analysis

### 4. Analysis of Bond Lengths and Bond Angles

4.1. Analysis files are stored in /Bond length and angle analysis  
4.2. Contents include:  
  1. SiO4_analysis.py: Identifies Q2 and Q3 silicon-oxygen tetrahedra and calculates bond lengths and angles  
  2. AlO4_analysis.py: Analyzes aluminum-oxygen tetrahedra  
  3. Q2_bond_kde.png: KDE plot of bond length distribution at Q2 Si-O site  
  4. Q2_angle_kde.png: KDE plot of bond angle distribution at Q2 Si-O site  
  5. Q3_bond_kde.png: KDE plot of bond length distribution at Q3 Si-O site  
  6. Q3_angle_kde.png: KDE plot of bond angle distribution at Q3 Si-O site  
  7. Q2_bond_kde_Al.png: KDE plot of bond length distribution at Q2 Al-O site  
  8. Q2_angle_kde_Al.png: KDE plot of bond angle distribution at Q2 Al-O site  
  9. Q3_bond_kde_Al.png: KDE plot of bond length distribution at Q3 Al-O site  
  10. Q3_angle_kde_Al.png: KDE plot of bond angle distribution at Q3 Al-O site  
  11. Results.csv: Statistical summary of average bond lengths and angles for Si-O and Al-O, as well as the absolute changes at 30 ps and 60 ps

## Usage Instructions

1. LAMMPS is required for MD simulations  
2. Python 3.x is required for running ML and data analysis scripts  
3. Required Python libraries include numpy, pandas, scikit-learn, and keras  
4. Follow instructions in the respective scripts for simulation, model training, and analysis
