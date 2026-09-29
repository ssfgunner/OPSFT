import pathlib

from lcb_runner.lm_styles import LanguageModel, LMStyle
from lcb_runner.utils.scenarios import Scenario


def ensure_dir(path: str, is_file=True):
    if is_file:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def _output_directory(model_repr: str, args) -> str:
    # A trajectory evaluator can supply a distinct directory per checkpoint.
    # Keep the original local-model basename behavior for existing invocations.
    return args.custom_output_save_name or f"./lcb_outputs/{args.local_model_path.split('/')[-1]}"


def get_cache_path(model_repr: str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    path = f"{_output_directory(model_repr, args)}/cache/{scenario}_{n}_{temperature}.json"
    ensure_dir(path)
    return path


def get_output_path(model_repr: str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    path = f"{_output_directory(model_repr, args)}/{scenario}_{n}_{temperature}{cot_suffix}.json"
    ensure_dir(path)
    return path


def get_eval_all_output_path(model_repr: str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    return f"{_output_directory(model_repr, args)}/{scenario}_{n}_{temperature}{cot_suffix}_eval_all.json"

