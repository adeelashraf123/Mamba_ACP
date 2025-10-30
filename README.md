# Mamba-ACP: A Hybrid State-Space and Transformer Framework for Interpretable Anticancer Peptide Prediction
# handcrafted biochemical features (AAindex, BLOSUM62), and the Mamba state-space model.


**Welcome to Mamba-ACP, a hybrid deep learning framework for ACP prediction tool developed by a team from the 
Nanjing University of Science and Technology, Nanjing **

We provided our datasets (Set 1 and Set 2) and you can find them here

Mamba-ACP integerates ESM-2 transformer embeddings, handcrafted biochemical features (AAindex, BLOSUM62), and the Mamba state-space model for the ACP prediction.

We have provided the basic to captures both evolutionary and physicochemical properties of peptides to enhance prediction performance.

1. Features Exttractions both evolutionary and physicochemical properties of peptides.
2. pca used on handcrafted features only
3. Combine both evolutionary and physicochemical properties and then train the model.
4. Final prediction is done on Mamba model, either it is ACP or Non-ACP.
5. To support and validate the model, various ablation experiments are done.

If you have any questions or confusion, please feel free to contact me.

**Name: Professor Dong-Jun Yu/Adeel Ashraf
njyudj@njust.edu.cn/adeel.ashraf@njust.edu.cn
