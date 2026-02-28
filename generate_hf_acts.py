import torch as t
import gc
from datasets import load_dataset
from tqdm import tqdm
import argparse

from generate_acts import load_model, get_acts

def main():
    parser = argparse.ArgumentParser(description="Generate HF dataset activations for a specific model and layer.")
    
    # Model specification arguments
    parser.add_argument("--model_family", type=str, required=True, help="Model family (e.g., Qwen2.5, Llama3, Mistral)")
    parser.add_argument("--model_size", type=str, required=True, help="Model size (e.g., 7B, 8B)")
    parser.add_argument("--model_type", type=str, required=True, help="Model type (e.g., instruct, base)")
    
    # Layer and execution arguments
    parser.add_argument("--layer", type=int, default=-1, help="Specific layer to extract (-1 for last layer)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for mapping")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HF hub directly")
    args = parser.parse_args()

    layer = args.layer
    device = args.device
    model_family = args.model_family
    model_size = args.model_size
    model_type = args.model_type

    # Load the base dataset
    print("Loading base dataset KingTechnician/tiu-splits...")
    base_ds = load_dataset("KingTechnician/tiu-splits")

    # Process the specified model
    print(f"\n{'='*50}")
    print(f"Processing {model_family} {model_size} {model_type} at Layer {layer}...")
    
    # Load Model & Tokenizer
    t.set_grad_enabled(False)
    tokenizer, model = load_model(model_family, model_size, model_type, device)
    
    # Resolve the layer index if -1 is used (to get the actual integer index for naming)
    actual_layer = layer
    if layer == -1:
        actual_layer = len(model.model.layers) - 1
        print(f"Resolved layer -1 to actual layer index: {actual_layer}")

    # Define the mapping function
    def extract_activations(batch):
        statements = batch['text']
        
        # Pass the single layer as a list to get_acts
        acts_dict = get_acts(statements, tokenizer, model, [layer], device)
        
        # Format outputs for the HuggingFace dataset
        output = {}
        # get_acts returns a dict keyed by the layer index requested
        output[f"layer_{actual_layer}_acts"] = acts_dict[layer].cpu().numpy().tolist()
        return output

    # Apply mapping
    print(f"Extracting activations across Train/Val/Test splits...")
    act_ds = base_ds.map(
        extract_activations,
        batched=True,
        batch_size=args.batch_size,
        desc=f"Extracting {model_family} Layer {actual_layer}"
    )

    # Save or Push the resulting dataset with the layer appended
    dataset_name = f"tiu-acts-{model_family}-{model_size}-{model_type}-layer-{actual_layer}".lower()
    
    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: KingTechnician/{dataset_name} ...")
        act_ds.push_to_hub(f"KingTechnician/{dataset_name}")
    else:
        local_path = f"./hf_datasets/{dataset_name}"
        print(f"Saving locally to {local_path} ...")
        act_ds.save_to_disk(local_path)

    # Clean up VRAM
    del model
    del tokenizer
    t.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()