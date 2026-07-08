
import sys
sys.path.insert(1, "./")
import ilm
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Callable, List
from scipy.cluster.hierarchy import linkage, leaves_list

def user_interface(
        ilmodel: ilm.IntuinisticLanguageModel, 
        tokenizer: Callable[[str], List[Optional[str]]], 
        detokenizer: Callable[[List[str]], List[Optional[str]]],
        completed_words: int = 100,
        syllable_num: int = 3,
        ) -> None:
    
    while True:
        string = input(">>> ")

        if string == "!exit":
            break
        
        if string == "!plot":
            state_dict = ilmodel.state_dict()
            state_dict_vals = list(state_dict.values())
            state_dict_keys = list(state_dict.keys())

            for i, (k,v) in enumerate(state_dict.items()):
                k: str
                v: torch.Tensor
                print(f"{i}) {k} {v.shape}")
            
            while True:
                string = input("~ >>> ")
                if string == "!exit":
                    break
                
                index = None
                try:
                    if 0 <= int(string) < len(state_dict_vals):
                        index = int(string)
                    else:
                        print("Please choose an index from one of the options.")
                except:
                    pass
                
                if index is not None:
                    tensor: torch.Tensor = state_dict_vals[index]

                    if not(tensor.is_cpu):
                        tensor = tensor.cpu()

                    if tensor.ndimension() == 1:
                        plt.plot(tensor.numpy())

                    elif tensor.ndimension() == 2:
                        image_data = tensor.numpy()

                        # Compute linkage matrix for hierarchical clustering
                        linkage_matrix = linkage(image_data, method="ward")  # You can also try 'average', 'single', etc.

                        # Get the optimal leaf order (best ordering of rows based on clustering)
                        optimal_order = leaves_list(linkage_matrix)

                        # Reorder rows using this optimal order
                        sorted_image = image_data[optimal_order]

                        # Plot side by side
                        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                        axes[0].imshow(image_data, cmap="gray")
                        axes[0].set_title("Original Image")

                        axes[1].imshow(sorted_image, cmap="gray")
                        axes[1].set_title("Hierarchically Ordered Image")

                        # This adds a global title to the entire figure
                        fig.suptitle(state_dict_keys[index], fontsize=16)

                        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the global title
                        plt.show()

                    else:
                        print("Tensor is not 1D or 2D, unable to plot.")
                        continue
                    
                    plt.show()
            continue

        try:
            single_context = ilm.format_context(string, tokenizer=tokenizer).unsqueeze(0) # turn (T, ) to (1, T)
        except:
            print("Language not recongized!")
            continue
                
            
        generated_tokens = ilmodel.generate(single_context, 
                                            max_new_tokens= syllable_num * completed_words,
                                            temperature=0.7,
                                            top_k=10,
                                            ).detach().cpu()[0].tolist() # turn (1, T) to #list=T
        # print("[GEN]", generated_tokens)
        out = ilm.gather_tokens(generated_tokens, syllable_num=syllable_num)
        print("".join([str(x) for x in detokenizer(out)]).replace(string, "", 1))