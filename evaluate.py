import os
import torch
import argparse
import numpy as np
from src.models import HyPINO
from src.data.utils import plot_grids
from src.data.datasets import EvalPDEDataset
from src.models.utils import convert_ckpt_to_safetensors

MODEL_REGISTRY = {
    "hypino": HyPINO,
}

BENCHMARKS = ['eval_data/heat.pkl', 'eval_data/helmholtz.pkl',
              'eval_data/helmholtz_G.pkl', 'eval_data/poisson_C.pkl',
              'eval_data/poisson_L.pkl', 'eval_data/poisson_G.pkl',
              'eval_data/wave.pkl']

def smape(y_true, y_pred):
    return 100 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def evaluate_model(weights_path, model_name, output_dir, plot=True):
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, "w") as f_out:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ModelClass = MODEL_REGISTRY[model_name.lower()]

        if weights_path.endswith('.ckpt'):
            weights_path = convert_ckpt_to_safetensors(weights_path)
            
        model = ModelClass.load_from_safetensors(weights_path, strict=False, ignore_size_mismatch=True)
        model.eval().to(device)

        dataset = EvalPDEDataset(BENCHMARKS)

        for index in range(len(BENCHMARKS)):
            data = dataset[index]
            pde_coeffs = data['pde_coeffs'].unsqueeze(0).to(device)
            grids = data['mat_inputs'].unsqueeze(0).to(device)
            ref = data['u_grid'].cpu().numpy().reshape(224, 224)

            with torch.no_grad():
                x = np.linspace(-1, 1, 224)
                y = np.linspace(1, -1, 224)
                xy = torch.tensor(np.stack(np.meshgrid(x, y), axis=-1)).flatten(0, 1).unsqueeze(0).to(device, dtype=torch.float32)
                pred = model(pde_coeffs, grids, xy=xy)
            pred = pred.detach().cpu().numpy().reshape(224, 224) * data['domain_mask'].numpy()

            mse = np.mean((ref - pred)**2)
            mae = np.mean(np.abs(ref - pred))
            max_error = np.max(np.abs(ref - pred))
            smape_val = smape(ref, pred)

            benchmark_name = os.path.splitext(os.path.basename(BENCHMARKS[index]))[0]
            f_out.write(f"{benchmark_name}:\n")
            f_out.write(f"MSE: {mse:.4e}\n")
            f_out.write(f"MAE: {mae:.4e}\n")
            f_out.write(f"Max Error: {max_error:.4e}\n")
            f_out.write(f"SMAPE: {smape_val:.2f}%\n\n")

            if plot:
                fields = [ref, pred, np.abs(ref - pred)]
                titles = ['Reference Solution', 'Predicted Solution', 'L1 Difference']
                plot_path = os.path.join(output_dir, f"plot_{benchmark_name}.png")
                plot_grids(fields, titles=titles, save_path=plot_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a sample.")
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--model', type=str, required=True, choices=MODEL_REGISTRY.keys(), help='Model name')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results and plots')
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting of results')

    args = parser.parse_args()

    evaluate_model(
        weights_path=args.weights,
        model_name=args.model,
        output_dir=args.output_dir,
        plot=not args.no_plot
    )

if __name__ == '__main__':
    main()
