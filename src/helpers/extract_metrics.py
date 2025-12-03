import argparse
import os
import glob
import json

import numpy as np
import optuna
import pandas as pd


def extract_sb3_evaluations(log_dir: str):
    """
    Extrai avaliações de modelos SB3 do arquivo evaluations.npz.
    Retorna DataFrame com timesteps, returns, ep_lengths.
    """
    eval_file = os.path.join(log_dir, "eval_logs", "evaluations.npz")
    if not os.path.exists(eval_file):
        return None
    
    try:
        data = np.load(eval_file, allow_pickle=True)
        
        # SB3 EvalCallback salva: timesteps, results, ep_lengths
        timesteps = data.get("timesteps", np.array([]))
        results = data.get("results", np.array([]))  # Shape: (n_evals, n_episodes)
        ep_lengths = data.get("ep_lengths", np.array([]))  # Shape: (n_evals, n_episodes)
        
        if len(timesteps) == 0:
            return None
        
        # Calcular estatísticas por avaliação
        mean_returns = results.mean(axis=1) if results.ndim > 1 else results
        std_returns = results.std(axis=1) if results.ndim > 1 else np.zeros_like(mean_returns)
        mean_ep_lengths = ep_lengths.mean(axis=1) if ep_lengths.ndim > 1 else ep_lengths
        
        df = pd.DataFrame({
            "timestep": timesteps,
            "mean_return": mean_returns,
            "std_return": std_returns,
            "mean_ep_length": mean_ep_lengths,
        })
        
        return df
    except Exception as e:
        print(f"Erro ao ler {eval_file}: {e}")
        return None


def extract_custom_evaluations(log_dir: str):
    """
    Extrai avaliações de modelos custom do arquivo eval_returns.txt.
    Formato: step,return (um por linha)
    """
    eval_file = os.path.join(log_dir, "eval_logs", "eval_returns.txt")
    if not os.path.exists(eval_file):
        return None
    
    try:
        data = []
        with open(eval_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    step = int(parts[0])
                    return_val = float(parts[1])
                    data.append({"timestep": step, "mean_return": return_val})
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Erro ao ler {eval_file}: {e}")
        return None


def extract_optuna_study(log_dir: str):
    """
    Extrai resultados de estudos Optuna do arquivo optuna.db.
    Retorna dicionário com informações dos trials.
    """
    optuna_db = os.path.join(log_dir, "optuna.db")
    if not os.path.exists(optuna_db):
        return None
    
    try:
        # Tentar carregar o estudo
        storage_url = f"sqlite:///{optuna_db}"
        
        # Listar todos os estudos no storage
        study_summaries = optuna.get_all_study_summaries(storage_url)
        
        if not study_summaries:
            return None
        
        # Pegar o primeiro estudo (ou podemos iterar sobre todos)
        study_name = study_summaries[0].study_name
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "user_attrs": trial.user_attrs,
                }
                trials_data.append(trial_data)
        
        best_trial = study.best_trial if study.best_trial else None
        
        return {
            "study_name": study_name,
            "n_trials": len(study.trials),
            "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
            } if best_trial else None,
            "trials": trials_data,
        }
    except Exception as e:
        print(f"Erro ao ler {optuna_db}: {e}")
        return None


def get_model_info(log_dir: str) -> dict:
    """
    Extrai informações gerais sobre o modelo (configs, checkpoints, etc).
    """
    info = {
        "log_dir": log_dir,
        "exp_name": os.path.basename(log_dir),
        "has_sb3_config": os.path.exists(os.path.join(log_dir, "cfgs_sb3.pkl")),
        "has_custom_config": os.path.exists(os.path.join(log_dir, "cfgs_custom.pkl")),
        "has_ppo_config": os.path.exists(os.path.join(log_dir, "cfgs.pkl")),
        "checkpoints": [],
        "best_model": None,
        "final_model": None,
    }
    
    # Procurar checkpoints
    checkpoint_patterns = [
        "*_final.zip",
        "*_final.pt",
        "*_*_steps.zip",
        "model_*.pt",
    ]
    
    for pattern in checkpoint_patterns:
        matches = glob.glob(os.path.join(log_dir, pattern))
        for match in matches:
            if "best_model" in match:
                info["best_model"] = os.path.basename(match)
            elif "final" in match.lower():
                info["final_model"] = os.path.basename(match)
            else:
                info["checkpoints"].append(os.path.basename(match))
    
    # Verificar se há best_model em subdiretório
    best_model_dir = os.path.join(log_dir, "best_model")
    if os.path.exists(best_model_dir):
        best_files = glob.glob(os.path.join(best_model_dir, "*"))
        if best_files:
            info["best_model"] = os.path.join("best_model", os.path.basename(best_files[0]))
    
    return info


def extract_all_metrics(log_dir: str) -> dict:
    """
    Extrai todas as métricas disponíveis de um diretório de log.
    """
    results = {
        "info": get_model_info(log_dir),
        "sb3_evaluations": None,
        "custom_evaluations": None,
        "optuna_study": None,
    }
    
    # Tentar extrair avaliações SB3
    sb3_df = extract_sb3_evaluations(log_dir)
    if sb3_df is not None:
        results["sb3_evaluations"] = sb3_df
    
    # Tentar extrair avaliações custom
    custom_df = extract_custom_evaluations(log_dir)
    if custom_df is not None:
        results["custom_evaluations"] = custom_df
    
    # Tentar extrair estudo Optuna
    optuna_data = extract_optuna_study(log_dir)
    if optuna_data is not None:
        results["optuna_study"] = optuna_data
    
    return results


def print_summary(results: dict, verbose: bool = False):
    """
    Imprime resumo das métricas.
    """
    info = results["info"]
    print(f"\n{'='*60}")
    print(f"Experimento: {info['exp_name']}")
    print(f"Diretório: {info['log_dir']}")
    print(f"{'='*60}")
    
    # Informações gerais
    print("\n📁 Informações do Modelo:")
    print(f"  Config SB3: {'✓' if info['has_sb3_config'] else '✗'}")
    print(f"  Config Custom: {'✓' if info['has_custom_config'] else '✗'}")
    print(f"  Config PPO: {'✓' if info['has_ppo_config'] else '✗'}")
    
    if info["best_model"]:
        print(f"  Melhor modelo: {info['best_model']}")
    if info["final_model"]:
        print(f"  Modelo final: {info['final_model']}")
    if info["checkpoints"]:
        print(f"  Checkpoints: {len(info['checkpoints'])} encontrados")
    
    # Avaliações SB3
    if results["sb3_evaluations"] is not None:
        df = results["sb3_evaluations"]
        print("\n📊 Avaliações SB3:")
        print(f"  Total de avaliações: {len(df)}")
        if len(df) > 0:
            print(f"  Melhor retorno: {df['mean_return'].max():.4f} (timestep: {df.loc[df['mean_return'].idxmax(), 'timestep']})")
            print(f"  Último retorno: {df['mean_return'].iloc[-1]:.4f}")
            print(f"  Retorno médio: {df['mean_return'].mean():.4f} ± {df['mean_return'].std():.4f}")
            if verbose:
                print("\n  Primeiras 10 avaliações:")
                print(df.head(10).to_string(index=False))
    
    # Avaliações Custom
    if results["custom_evaluations"] is not None:
        df = results["custom_evaluations"]
        print("\n📊 Avaliações Custom:")
        print(f"  Total de avaliações: {len(df)}")
        if len(df) > 0:
            print(f"  Melhor retorno: {df['mean_return'].max():.4f} (timestep: {df.loc[df['mean_return'].idxmax(), 'timestep']})")
            print(f"  Último retorno: {df['mean_return'].iloc[-1]:.4f}")
            print(f"  Retorno médio: {df['mean_return'].mean():.4f} ± {df['mean_return'].std():.4f}")
            if verbose:
                print("\n  Primeiras 10 avaliações:")
                print(df.head(10).to_string(index=False))
    
    # Estudo Optuna
    if results["optuna_study"] is not None:
        study_data = results["optuna_study"]
        print("\n🔬 Estudo Optuna:")
        print(f"  Nome: {study_data['study_name']}")
        print(f"  Total de trials: {study_data['n_trials']}")
        print(f"  Trials completos: {study_data['n_complete']}")
        if study_data["best_trial"]:
            best = study_data["best_trial"]
            print(f"  Melhor trial: #{best['number']}")
            print(f"  Melhor valor: {best['value']:.4f}")
            if verbose:
                print("  Melhores hiperparâmetros:")
                for key, value in best["params"].items():
                    print(f"    {key}: {value}")


def save_metrics(results: dict, output_dir: str, format: str = "json"):
    """
    Salva métricas em arquivo.
    """
    os.makedirs(output_dir, exist_ok=True)
    exp_name = results["info"]["exp_name"]
    
    if format == "json":
        output_file = os.path.join(output_dir, f"{exp_name}_metrics.json")
        output_data = {
            "info": results["info"],
            "optuna_study": results["optuna_study"],
        }
        
        # Converter DataFrames para dict
        if results["sb3_evaluations"] is not None:
            output_data["sb3_evaluations"] = results["sb3_evaluations"].to_dict("records")
        
        if results["custom_evaluations"] is not None:
            output_data["custom_evaluations"] = results["custom_evaluations"].to_dict("records")
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n💾 Métricas salvas em: {output_file}")
    
    elif format == "csv":
        # Salvar avaliações como CSV
        if results["sb3_evaluations"] is not None:
            csv_file = os.path.join(output_dir, f"{exp_name}_sb3_evaluations.csv")
            results["sb3_evaluations"].to_csv(csv_file, index=False)
            print(f"💾 Avaliações SB3 salvas em: {csv_file}")
        
        if results["custom_evaluations"] is not None:
            csv_file = os.path.join(output_dir, f"{exp_name}_custom_evaluations.csv")
            results["custom_evaluations"].to_csv(csv_file, index=False)
            print(f"💾 Avaliações Custom salvas em: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extrai e apresenta métricas de modelos treinados"
    )
    parser.add_argument(
        "-e", "--exp_name",
        type=str,
        default=None,
        help="Nome do experimento (ou 'all' para todos)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Diretório base de logs",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostrar informações detalhadas",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Salvar métricas em arquivo (formato: json ou csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metrics",
        help="Diretório para salvar métricas",
    )
    args = parser.parse_args()
    
    log_base = args.log_dir
    
    if args.exp_name is None or args.exp_name == "all":
        # Processar todos os experimentos
        if not os.path.exists(log_base):
            print(f"Diretório {log_base} não existe!")
            return
        
        exp_dirs = [d for d in os.listdir(log_base) if os.path.isdir(os.path.join(log_base, d))]
        exp_dirs.sort()
        
        print(f"Encontrados {len(exp_dirs)} experimentos\n")
        
        all_results = {}
        for exp_name in exp_dirs:
            exp_dir = os.path.join(log_base, exp_name)
            results = extract_all_metrics(exp_dir)
            all_results[exp_name] = results
            print_summary(results, verbose=args.verbose)
        
        # Salvar todas as métricas se solicitado
        if args.save:
            for exp_name, results in all_results.items():
                save_metrics(results, args.output_dir, format=args.save)
    
    else:
        # Processar experimento específico
        exp_dir = os.path.join(log_base, args.exp_name)
        if not os.path.exists(exp_dir):
            print(f"Experimento {args.exp_name} não encontrado em {log_base}!")
            return
        
        results = extract_all_metrics(exp_dir)
        print_summary(results, verbose=args.verbose)
        
        if args.save:
            save_metrics(results, args.output_dir, format=args.save)


if __name__ == "__main__":
    main()

