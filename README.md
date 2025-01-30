# TVFEMD-NRBO-CNN-BiLSTM-Attention
# Application Installation, Testing, and Deployment Guide

## 1. Software Requirements

- **MATLAB** (version ≥ R2020a)
- Required Toolboxes:
  - Deep Learning Toolbox
  - Signal Processing Toolbox
  - Optimization Toolbox
- Ensure the working directory includes all project files:
  - `NRBO/` folder
  - `MAIN_VMD.m`
  - `VMD_CNNbiLSTM.m`

------

## 2. Workflow Execution

### **Step 1: Parameter Optimization via NRBO**

**Objective**: Obtain optimized hyperparameters (`best_hd`, `best_lr`, `best_l2`).

1. Navigate to the `NRBO/` directory:

   matlab

   ```
   cd('path/to/NRBO');  
   ```

2. Execute the main optimization script:

   matlab

   ```
   main;  
   ```

3. **Output**:

   - Optimized parameters saved to the MATLAB workspace.

   - Verify outputs:

     ```
     disp(['best_hd = ', num2str(best_hd)]);  
     disp(['best_lr = ', num2str(best_lr)]);  
     disp(['best_l2 = ', num2str(best_l2)]);  
     ```

------

### **Step 2: Time Series Decomposition via VMD**

**Objective**: Generate variational mode decomposition (VMD) components.

1. Return to the root project directory:

   ```
   cd('path/to/project_root');  
   ```

2. Run the VMD decomposition script:

   ```
   MAIN_VMD;  
   ```

3. **Output**:

   - Decomposed components saved as `vmd_data.mat`.

   - Validate file generation:

     ```
     if exist('vmd_data.mat', 'file')  
         disp('VMD decomposition successful.');  
     end  
     ```

------

### **Step 3: Predictive Modeling with CNN-BiLSTM**

**Objective**: Train and validate the hybrid CNN-BiLSTM model.

1. Load the VMD components and optimized parameters:

   ```
   load('vmd_data.mat');  
   best_hd = [value];  % Replace [value] with actual optimized parameters  
   best_lr = [value];  
   best_l2 = [value];  
   ```

2. Execute the prediction script:

   ```
   VMD_CNNbiLSTM;  
   ```

3. **Output**:

   - Performance metrics (e.g., RMSE, MAE, R²) displayed in the command window.
   - Visualization plots (e.g., time series forecasts, error distributions).

------

## 4. Troubleshooting

- **Missing Toolboxes**: Validate toolbox installation using:

  ```
  ver('Deep_Learning_Toolbox');  
  ```

- **Path Errors**: Add project directories to MATLAB's search path:

  ```
  addpath(genpath('path/to/project_root'));  
  ```

------

This documentation ensures reproducibility and aligns with academic standards for computational workflows. Adjust file paths and parameter placeholders (`[value]`) as needed.
