import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# plot_confusion_matrices

def fn_PlotConfusionMatrics(cf_matrix, class_labels):
    
    fig_cf, axes_cd = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    fig_cf.suptitle("Seaborn Confusion Matrix with labels")
    fig_cf.supxlabel("--Predicted-- Bug Report Category")
    fig_cf.supylabel("--Actual-- Bug Report Category")
    
    datasetNames = ["validation", "train"]
    
    mycbar = True
    snax = sns.heatmap(
        cf_matrix / np.sum(cf_matrix), 
        annot=True, 
        fmt=".2%", 
        cmap="Blues", 
        ax=axes_cd, 
        vmin=0, 
        vmax=1, 
        cbar=mycbar, 
        annot_kws={"size": 12}
        )
    
    
    snax.set_title("[Cosine Similarity]\n")
    
    labels = list(map(str, class_labels))
    
    snax.xaxis.set_ticklabels(labels)
    snax.yaxis.set_ticklabels(labels)
    
    fig_cf.tight_layout(w_pad=6.0)

    plt.show()

