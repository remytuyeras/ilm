
import sys
sys.path.insert(1, "./")
import ilm
import torch

from typing import Optional, Callable, List, Sequence

ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[36m"
ANSI_MAGENTA = "\033[35m"
ANSI_BOLD = "\033[1m"


def color_text(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def _plot_tensor_bank(tensor: torch.Tensor, title: str, slice_prefix: str = "slice") -> None:
    import matplotlib.pyplot as plt

    tensor = tensor.detach().cpu().float()
    row_sums = tensor.sum(dim=-1)
    print(f"Plotting {title}: {tuple(tensor.shape)}")
    if tensor.ndimension() == 3:
        print(
            "row sums: "
            f"min={row_sums.min().item():.4f}, "
            f"max={row_sums.max().item():.4f}"
        )

    fig, axes = plt.subplots(
        1,
        tensor.shape[0],
        figsize=(5 * tensor.shape[0], 5),
        squeeze=False,
    )
    vmin = float(tensor.min().item())
    vmax = float(tensor.max().item())
    for p in range(tensor.shape[0]):
        ax = axes[0, p]
        image = ax.imshow(tensor[p].numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{slice_prefix}_{p}")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def _plot_tensor(name: str, tensor: torch.Tensor) -> None:
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, leaves_list

    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()

    if tensor.dtype == torch.bool:
        tensor = tensor.float()

    if tensor.ndimension() == 1:
        plt.plot(tensor.float().numpy())
        plt.title(name)
        plt.show()

    elif tensor.ndimension() == 2:
        image_data = tensor.float().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(image_data, cmap="gray")
        axes[0].set_title("Original Image")

        if image_data.shape[0] >= 2:
            linkage_matrix = linkage(image_data, method="ward")
            optimal_order = leaves_list(linkage_matrix)
            sorted_image = image_data[optimal_order]
            axes[1].imshow(sorted_image, cmap="gray")
            axes[1].set_title("Hierarchically Ordered Image")
        else:
            axes[1].imshow(image_data, cmap="gray")
            axes[1].set_title("Single Row")

        fig.suptitle(name, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    elif tensor.ndimension() == 3:
        slice_prefix = "J" if "J_" in name or "_j" in name.lower() else "slice"
        _plot_tensor_bank(tensor=tensor, title=name, slice_prefix=slice_prefix)

    else:
        print("Tensor is not 1D, 2D, or 3D, unable to plot.")


def user_interface(
        ilmodel: ilm.IntuinisticLanguageModel, 
        tokenizer: Callable[[str], List[Optional[str]]], 
        detokenizer: Callable[[List[str]], List[Optional[str]]],
        completed_words: int = 100,
        syllable_num: int = 3,
        temperature: float = 0.8,
        top_k: Optional[int] = 10,
        top_k_by_coordinate: Optional[Sequence[int]] = None,
        temperature_by_coordinate: Optional[Sequence[float]] = None,
        stream: bool = False,
        generation_seed: Optional[int] = None,
        oov_policy: str = "error",
        oov_fallback_code: Optional[str] = None,
        ) -> None:
    
    while True:
        string = input(color_text(">>> ", ANSI_BOLD + ANSI_CYAN))

        if string == "!exit":
            break
        
        if string == "!plot":
            state_dict = ilmodel.state_dict()
            plot_entries = list(state_dict.items())

            for i, (k,v) in enumerate(plot_entries):
                k: str
                v: torch.Tensor
                marker = " [J]" if "J_" in k or "J_p" in k or "_j" in k.lower() else ""
                print(f"{i}){marker} {k} {v.shape}")
            
            while True:
                string = input(color_text("~ >>> ", ANSI_BOLD + ANSI_MAGENTA))
                if string == "!exit":
                    break
                
                index = None
                try:
                    if 0 <= int(string) < len(plot_entries):
                        index = int(string)
                    else:
                        print("Please choose an index from one of the options.")
                except:
                    pass
                
                if index is not None:
                    name, tensor = plot_entries[index]
                    _plot_tensor(name=name, tensor=tensor)
            continue

        try:
            single_context = ilm.format_context(
                string,
                tokenizer=tokenizer,
                oov_policy=oov_policy,
                fallback_code=oov_fallback_code,
            ).unsqueeze(0) # turn (T, ) to (1, T)
        except:
            print("Language not recongized!")
            continue

        if generation_seed is not None:
            ilm.set_seed(generation_seed)
                
            
        if stream:
            streamed_tokens = []

            def print_completed_word(token: int):
                streamed_tokens.append(token)
                if len(streamed_tokens) < syllable_num:
                    return
                code = ":".join(str(part) for part in streamed_tokens)
                word = detokenizer([code])[0]
                print(str(word), end="", flush=True)
                streamed_tokens.clear()

            ilmodel.generate(single_context,
                             max_new_tokens=syllable_num * completed_words,
                             temperature=temperature,
                             top_k=top_k,
                             syllable_num=syllable_num,
                             top_k_by_coordinate=top_k_by_coordinate,
                             temperature_by_coordinate=temperature_by_coordinate,
                             token_callback=print_completed_word,
                             show_progress=False,
                             )
            print()
            continue

        generated_tokens = ilmodel.generate(single_context,
                                            max_new_tokens=syllable_num * completed_words,
                                            temperature=temperature,
                                            top_k=top_k,
                                            syllable_num=syllable_num,
                                            top_k_by_coordinate=top_k_by_coordinate,
                                            temperature_by_coordinate=temperature_by_coordinate,
                                            ).detach().cpu()[0].tolist() # turn (1, T) to #list=T
        # print("[GEN]", generated_tokens)
        out = ilm.gather_tokens(generated_tokens, syllable_num=syllable_num)
        print("".join([str(x) for x in detokenizer(out)]).replace(string, "", 1))
