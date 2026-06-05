import os
import re
import sys
import pickle
import shutil
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# --- Core LlamaIndex Imports (Safe for older PyTorch) ---
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

PHASE1_REQUIRED_KEYS = ("Description", "AdvType", "AdvPos", "AdvBehavior")
HEADER_LINE_RE = re.compile(r"^#\s*(Description|AdvType|AdvPos|AdvBehavior):")

scenario_dict = {
    1: "Straight Obstacle",
    2: "Turning Obstacle",
    3: "Lane Changing",
    4: "Vehicle Passing",
    5: "Red-light Running",
    6: "Unprotected Left-turn",
    7: "Right-turn",
    8: "Crossing Negotiation",
}

# Same default CARLA world model as Safebench `ScenicSimulator` / scenic CLI.
DEFAULT_SCENIC_MODEL = "scenic.simulators.carla.model"

SAFEBENCH_SCENIC_CHECKLIST = """
Safebench alignment checklist (must satisfy when generating/fixing):
1) Read externally injected values from globalParameters: at least town, spawnPt, yaw, waypoints; if the script uses lanePts, also require globalParameters.lanePts.
2) Map: param map = localPath(f'../maps/{Town}.xodr'), and param carla_map = Town (consistent with Safebench examples in this repo).
3) model scenic.simulators.carla.model
4) Tunable hyperparameters: param OPT_xxx = Range(...) and reference globalParameters.OPT_xxx in logic.
5) Adversarial behavior names should include Adv (e.g. AdvBehavior) so Safebench can identify the adversarial actor.
6) ego is typically Car, with regionContainedIn None, with blueprint EGO_MODEL.
7) Do not use Vector(x,y,z) or EgoSpawnPt+Vector; spawning must use Safebench patterns such as OrientedPoint, left/right of, and @ offsets.
8) AdvAgent must have with behavior AdvBehavior().
"""


def _msgen_root() -> str:
    return os.path.dirname(os.path.realpath(__file__))


def _repo_root() -> str:
    # safebench/scenge/msgen -> safebench/scenge -> safebench -> repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(_msgen_root())))


def load(file_path: str) -> Any:
    suffix = file_path.split(".")[-1]
    if suffix == "txt":
        with open(file_path, "r") as file:
            return file.read()
    elif suffix == "pkl":
        with open(file_path, "rb") as file:
            return pickle.load(file)
    else:
        raise Exception("Only txt and pkl file suffixes are supported.")


def clean_qwq_output(text: str) -> str:
    """Helper function: remove QwQ's <redacted_thinking> tags"""
    if "</redacted_thinking>" in text:
        return text.split("</redacted_thinking>")[-1].strip()
    return text.strip()


def configure_llm(
    llm_backend: str,
    llm_name: str,
    *,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_temperature: float = 0.1,
    openai_max_tokens: Optional[int] = None,
) -> None:
    """Configure ``Settings.llm`` for HuggingFace local or OpenAI-compatible APIs."""
    backend = llm_backend.lower()
    if backend == "openai":
        from llama_index.llms.openai import OpenAI

        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI backend requires --openai-api-key or env OPENAI_API_KEY"
            )
        api_base = openai_base_url or os.environ.get("OPENAI_BASE_URL")
        kwargs = {
            "model": llm_name,
            "api_key": api_key,
            "temperature": openai_temperature,
        }
        if api_base:
            kwargs["api_base"] = api_base
        if openai_max_tokens is not None:
            kwargs["max_tokens"] = openai_max_tokens
        Settings.llm = OpenAI(**kwargs)
        base_msg = api_base or "https://api.openai.com/v1"
        print(f">> LLM backend: OpenAI API ({llm_name}, base={base_msg})")
    elif backend in ("huggingface", "hf"):
        from llama_index.llms.huggingface import HuggingFaceLLM

        Settings.llm = HuggingFaceLLM(model_name=llm_name, tokenizer_name=llm_name)
        print(f">> LLM backend: HuggingFace ({llm_name})")
    else:
        raise ValueError(
            f"Unknown llm_backend={llm_backend!r}; use 'openai' or 'huggingface'"
        )


def configure_embed_model(
    embed_backend: str,
    embed_name: str,
    *,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
) -> None:
    """Configure ``Settings.embed_model`` for RAG retrieval."""
    backend = embed_backend.lower()
    if backend == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI embed backend requires --openai-api-key or env OPENAI_API_KEY"
            )
        api_base = openai_base_url or os.environ.get("OPENAI_BASE_URL")
        kwargs = {"model": embed_name, "api_key": api_key}
        if api_base:
            kwargs["api_base"] = api_base
        Settings.embed_model = OpenAIEmbedding(**kwargs)
        base_msg = api_base or "https://api.openai.com/v1"
        print(f">> Embed backend: OpenAI API ({embed_name}, base={base_msg})")
    elif backend in ("huggingface", "hf"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_name)
        print(f">> Embed backend: HuggingFace ({embed_name})")
    else:
        raise ValueError(
            f"Unknown embed_backend={embed_backend!r}; use 'openai' or 'huggingface'"
        )


def resolve_embed_backend(embed_backend: Optional[str], llm_backend: str) -> str:
    if embed_backend:
        return embed_backend
    return "openai" if llm_backend.lower() == "openai" else "huggingface"


def default_embed_name(embed_backend: str) -> str:
    if embed_backend.lower() == "openai":
        return "text-embedding-3-small"
    return "../weights/bge-m3"

def extract_text_block(text: str) -> str:
    """Parse phase-1 structured fields from a ```Text``` block."""
    text = clean_qwq_output(text)
    patterns = [
        r"```Text\s*(.*?)```",
        r"```text\s*(.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    raise ValueError("No ```Text ... ``` code block found in LLM output")


def parse_phase1_dict(text: str) -> Dict[str, str]:
    body = extract_text_block(text)
    phase1_dict: Dict[str, str] = {}
    for sentence in body.splitlines():
        sentence = sentence.strip()
        if not sentence or ": " not in sentence:
            continue
        key, value = sentence.split(": ", 1)
        phase1_dict[key.strip()] = value.strip()
    missing = [k for k in PHASE1_REQUIRED_KEYS if k not in phase1_dict]
    if missing:
        raise ValueError(f"phase-1 missing fields: {missing}")
    return phase1_dict


def strip_scenic_header(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if HEADER_LINE_RE.match(line.strip()):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def pick_template_name(scenario_id: int, adv_type: str) -> str:
    adv = (adv_type or "").lower()
    if scenario_id == 1 and adv in ("car", "vehicle", "truck", "bus"):
        return "scenario_1_vehicle_cutin.scenic"
    if scenario_id == 1:
        return "scenario_1_pedestrian_cross.scenic"
    return f"scenario_{scenario_id}_default.scenic"


def load_scenic_template_body(scenario_id: int, adv_type: str) -> str:
    msgen_root = _msgen_root()
    repo_root = _repo_root()
    name = pick_template_name(scenario_id, adv_type)
    candidates = [
        os.path.join(msgen_root, "scenic_templates", name),
        os.path.join(
            repo_root,
            "safebench",
            "scenario",
            "scenario_data",
            "scenic_data",
            f"scenario_{scenario_id}_bak",
            "behavior_1_opt.scenic",
        ),
        os.path.join(
            repo_root,
            "safebench",
            "scenario",
            "scenario_data",
            "scenic_data",
            f"scenario_{scenario_id}",
            "behavior_1_opt.scenic",
        ),
        os.path.join(msgen_root, "scenic_templates", "scenario_1_pedestrian_cross.scenic"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            return strip_scenic_header(fh.read())
    raise FileNotFoundError(
        f"Scenic template for scenario_{scenario_id} not found; check scenic_templates/ or scenario_{scenario_id}_bak/"
    )


def build_scenic_from_template(
    phase1_dict: Dict[str, str], scenario_id: int, adv_type: str
) -> str:
    body = load_scenic_template_body(scenario_id, adv_type)
    return build_scenic_file_content(phase1_dict, body)


def ensure_scenic_maps_symlink(work_dir: str, repo_root: str) -> None:
    """
    Generated files live under work_dir/scenario_k/behavior_*.scenic.
    Safebench-style scripts use ``localPath('../maps/{Town}.xodr')`` → need work_dir/maps.
    """
    maps_dst = os.path.join(work_dir, "maps")
    src = os.path.join(
        repo_root,
        "safebench",
        "scenario",
        "scenario_data",
        "scenic_data",
        "maps",
    )
    if not os.path.isdir(src):
        return
    if os.path.lexists(maps_dst):
        return
    try:
        os.symlink(os.path.abspath(src), maps_dst, target_is_directory=True)
    except OSError:
        try:
            shutil.copytree(src, maps_dst)
        except OSError:
            pass


def try_load_safebench_route_params(
    repo_root: str, scenario_id: int, route_id: int, pickle_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    if pickle_path:
        candidates.append(pickle_path)
    candidates.append(
        os.path.join(
            repo_root,
            "safebench",
            "scenario",
            "scenario_data",
            "route",
            "scenic_route.pickle",
        )
    )
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        key = f"scenario_id_{scenario_id}_route_id_{route_id}"
        if key not in data:
            continue
        row = data[key]
        spawn = row["spawnPt"]
        return {
            "town": row["town"],
            "weather": row.get("weather", "ClearNoon"),
            "waypoints": row["waypoints"],
            "lanePts": row.get("lanePts", []),
            "spawnPt": (float(spawn["x"]), float(spawn["y"])),
            "z": float(spawn["z"]),
            "yaw": float(spawn["yaw"]),
        }
    return None


def default_stub_global_params(scenario_id: int) -> Dict[str, Any]:
    """
    Minimal globals when scenic_route.pickle is missing.
    Coordinates are placeholders; compile may succeed while scene sampling fails — still useful feedback.
    """
    _ = scenario_id
    return {
        "town": "Town01",
        "weather": "ClearNoon",
        "spawnPt": (335.0, 273.5),
        "z": 0.5,
        "yaw": 180.0,
        "waypoints": [(335.0, 273.5), (360.0, 273.5)],
        "lanePts": [],
    }


def format_scenic_exception(exc: BaseException) -> str:
    lines = [f"exception_type: {type(exc).__name__}", f"message: {exc}"]
    for name in ("filename", "lineno", "offset", "end_lineno", "end_offset", "text", "msg", "loc"):
        if hasattr(exc, name):
            try:
                lines.append(f"{name}: {getattr(exc, name)!r}")
            except Exception:
                pass
    return "\n".join(lines)


def validate_scenic_for_safebench(
    scenic_path: str,
    params: Dict[str, Any],
    model: str = DEFAULT_SCENIC_MODEL,
    try_sample_scene: bool = True,
) -> Tuple[bool, str]:
    """
    Tier 1: same entry as Safebench — scenarioFromFile(..., params=params, model=model).
    Tier 2: one scene sample — catches many issues that compile alone misses.
    """
    try:
        import scenic.syntax.translator as translator
        import scenic.core.errors as errors
    except ImportError as e:
        return False, f"Scenic not importable: {e}\nInstall the same scenic as Safebench uses."

    notes: List[str] = []

    try:
        scenario = errors.callBeginningScenicTrace(
            lambda: translator.scenarioFromFile(
                scenic_path, params=params, model=model, scenario=None
            )
        )
        notes.append("[compile] OK: Scenario object built.")
    except Exception:
        notes.append("[compile] FAILED")
        notes.append(traceback.format_exc())
        notes.append(format_scenic_exception(sys.exc_info()[1]))
        return False, "\n".join(notes)

    if not try_sample_scene:
        return True, "\n".join(notes)

    try:
        scene, iterations = errors.callBeginningScenicTrace(
            lambda: scenario.generate(verbosity=0)
        )
        notes.append(
            f"[sample] OK: generated scene (dynamicScenario={scene.dynamicScenario!r}, iterations={iterations})."
        )
        return True, "\n".join(notes)
    except Exception:
        notes.append(
            "[sample] FAILED (often what Safebench hits even when compile passes)"
        )
        notes.append(traceback.format_exc())
        notes.append(format_scenic_exception(sys.exc_info()[1]))
        return False, "\n".join(notes)


def extract_scenic_from_llm_output(text: str) -> str:
    patterns = [
        r"<redacted_thinking>.*?</redacted_thinking>\s*```Scenic\s*(.*?)```",
        r"```Scenic\s*(.*?)```",
        r"```scenic\s*(.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    raise ValueError("No ```Scenic ... ``` code block found in LLM output")


def build_scenic_file_content(phase1_dict: Dict[str, str], phase2_body: str) -> str:
    body = strip_scenic_header(phase2_body)
    return "\n".join(
        [
            f'# Description: {phase1_dict["Description"]}',
            f'# AdvType: {phase1_dict["AdvType"]}',
            f'# AdvPos: {phase1_dict["AdvPos"]}',
            f'# AdvBehavior: {phase1_dict["AdvBehavior"]}',
            body,
        ]
    )


def repair_scenic_with_llm(
    scenic_code: str,
    validation_feedback: str,
    phase1_dict: Dict[str, str],
) -> str:
    prompt = f"""You are fixing a Scenic 2.x script for the CARLA + Safebench pipeline.

{SAFEBENCH_SCENIC_CHECKLIST}

Below is the output from the static/sampling validator (may include both compile and sample stages):
---
{validation_feedback}
---

Current full file content:
---
{strip_scenic_header(scenic_code)}
---

Scenario semantic fields (do not change event meaning; only fix syntax, API, and Safebench constraints):
- Description: {phase1_dict.get("Description", "")}
- AdvType: {phase1_dict.get("AdvType", "")}
- AdvPos: {phase1_dict.get("AdvPos", "")}
- AdvBehavior: {phase1_dict.get("AdvBehavior", "")}

Spawn example (must follow this structure; do not invent syntax):
IntSpawnPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = Car left of IntSpawnPt by globalParameters.OPT_GEO_X_DISTANCE,
    with heading IntSpawnPt.heading,
    with regionContainedIn None,
    with behavior AdvBehavior()

Output only the Scenic source body (no # Description header comments), strictly in this format:
```Scenic
(full scenic source body here)
```
"""
    response = Settings.llm.complete(prompt)
    text = getattr(response, "text", None) or str(response)
    text = clean_qwq_output(text)
    body = extract_scenic_from_llm_output(text)
    return build_scenic_file_content(phase1_dict, body)


def run(
    llm_name: str,
    embed_name: str,
    scenario_ids: List[int],
    max_repair_rounds: int = 4,
    scenario_route_id: int = 1,
    route_pickle: Optional[str] = None,
    scenic_try_sample: bool = True,
    llm_backend: str = "huggingface",
    embed_backend: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_temperature: float = 0.1,
    openai_max_tokens: Optional[int] = None,
    use_template_fallback: bool = True,
    template_only: bool = False,
):
    repo_root = _repo_root()
    msgen_root = _msgen_root()
    work_dir = os.path.join(
        repo_root, "safebench", "scenario", "scenario_data", "scenic_data"
    )
    print(f"Writing Scenic files to: {work_dir}")
    for i in sorted(set(scenario_ids)):
        scenario_dir = os.path.join(work_dir, f"scenario_{i}")
        if os.path.exists(scenario_dir):
            shutil.rmtree(scenario_dir)

    ensure_scenic_maps_symlink(work_dir, repo_root)

    embed_backend = resolve_embed_backend(embed_backend, llm_backend)
    if embed_name in ("../weights/bge-m3", "BAAI/bge-small-en-v1.5", "BAAI/bge-large-en-v1.5"):
        if embed_backend == "openai":
            embed_name = default_embed_name(embed_backend)

    configure_llm(
        llm_backend,
        llm_name,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_temperature=openai_temperature,
        openai_max_tokens=openai_max_tokens,
    )
    configure_embed_model(
        embed_backend,
        embed_name,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
    )

    print("Loading RAG...")
    phase1_docs = SimpleDirectoryReader(
        input_dir=os.path.join(msgen_root, "knowledge_docs"),
        recursive=True,
    ).load_data()
    phase1_prompt = load(
        os.path.join(msgen_root, "prompts", "description_generate_extract.txt")
    )
    phase1_index = VectorStoreIndex.from_documents(phase1_docs)
    phase1_query_engine = phase1_index.as_query_engine()

    phase2_docs = SimpleDirectoryReader(
        input_dir=os.path.join(msgen_root, "scenic_docs"),
        recursive=True,
    ).load_data()
    phase2_prompt = load(os.path.join(msgen_root, "prompts", "scenic_generate.txt"))
    phase2_index = VectorStoreIndex.from_documents(phase2_docs)
    phase2_query_engine = phase2_index.as_query_engine()

    scenario_counter: Dict[int, int] = {}

    custom_bar_format = "{desc}: {bar} {n_fmt}/{total_fmt} | {postfix}"
    with tqdm(
        enumerate(scenario_ids), total=len(scenario_ids), bar_format=custom_bar_format
    ) as pbar:
        for idx, scenario_id in pbar:
            if scenario_id not in scenario_dict:
                print(f"Invalid scenario id {scenario_id}; please use a number from 1 to 8.")
                continue
            scenario_dir = os.path.join(work_dir, f"scenario_{scenario_id}")
            os.makedirs(scenario_dir, exist_ok=True)
            scenario_counter.setdefault(scenario_id, 0)
            scenario_counter[scenario_id] += 1
            scenario_idx = scenario_counter[scenario_id]

            while True:
                try:
                    pbar.set_description(
                        f"Scenario {idx + 1}: {scenario_dict[scenario_id]}, Generating Phase 1!"
                    )
                    phase1_output = phase1_query_engine.query(
                        phase1_prompt.format(base_scenario=scenario_dict[scenario_id])
                    ).response
                    phase1_dict = parse_phase1_dict(phase1_output)
                    break
                except Exception as e:
                    pbar.set_postfix(
                        retry_reason=f"Error occurred in phase 1: {e}, retrying..."
                    )
            pbar.set_postfix({})

            if template_only:
                scenic_code = build_scenic_from_template(
                    phase1_dict,
                    scenario_id,
                    phase1_dict.get("AdvType", ""),
                )
                pbar.set_description(
                    f"Scenario {idx + 1}: {scenario_dict[scenario_id]}, using template"
                )
            else:
                while True:
                    try:
                        pbar.set_description(
                            f"Scenario {idx + 1}: {scenario_dict[scenario_id]}, Generating Phase 2!"
                        )
                        phase2_output = phase2_query_engine.query(
                            phase2_prompt.format(
                                Description=f'[{phase1_dict["Description"]}]',
                                AdvType=f'[{phase1_dict["AdvType"]}]',
                                AdvPos=f'[{phase1_dict["AdvPos"]}]',
                                AdvBehavior=f'[{phase1_dict["AdvBehavior"]}]',
                            )
                        ).response
                        phase2_output = extract_scenic_from_llm_output(
                            clean_qwq_output(phase2_output)
                        )
                        break
                    except Exception as e:
                        pbar.set_postfix(
                            retry_reason=f"Error occurred in phase 2: {e}, retrying..."
                        )
                scenic_code = build_scenic_file_content(phase1_dict, phase2_output)
            pbar.set_postfix({})
            file_path = os.path.join(
                scenario_dir, f"behavior_{scenario_idx}_opt.scenic"
            )

            params = try_load_safebench_route_params(
                repo_root, scenario_id, scenario_route_id, route_pickle
            )
            if params is None:
                params = default_stub_global_params(scenario_id)
                pbar.set_postfix_str("using stub globalParameters (no route pickle)")

            # repair_attempt = number of LLM repair calls completed; write to disk then validate; on failure, call repair LLM up to max_repair_rounds more times
            repair_attempt = 0
            last_feedback = ""
            used_template_fallback = template_only
            while True:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(scenic_code)

                ok, last_feedback = validate_scenic_for_safebench(
                    file_path,
                    params,
                    model=DEFAULT_SCENIC_MODEL,
                    try_sample_scene=scenic_try_sample and not used_template_fallback,
                )
                if ok:
                    tag = "ok"
                    if repair_attempt:
                        tag = f"ok (llm_repairs={repair_attempt})"
                    if used_template_fallback:
                        tag += ", template"
                    pbar.set_postfix(scenic_ok=tag)
                    break

                if used_template_fallback:
                    ok_compile, compile_feedback = validate_scenic_for_safebench(
                        file_path,
                        params,
                        model=DEFAULT_SCENIC_MODEL,
                        try_sample_scene=False,
                    )
                    if ok_compile:
                        print(
                            f"\n[info] {file_path} template compile passed"
                            + (" (sample failed, still usable)" if scenic_try_sample else "")
                        )
                        pbar.set_postfix(scenic_ok="ok, template, compile_only")
                        break
                    last_feedback = compile_feedback

                if repair_attempt >= max_repair_rounds:
                    if use_template_fallback and not used_template_fallback:
                        print(
                            f"\n[info] {file_path} LLM validation failed; falling back to built-in Scenic template."
                        )
                        scenic_code = build_scenic_from_template(
                            phase1_dict,
                            scenario_id,
                            phase1_dict.get("AdvType", ""),
                        )
                        used_template_fallback = True
                        repair_attempt = max_repair_rounds
                        pbar.set_description(
                            f"Scenario {idx + 1}: validating template"
                        )
                        continue

                    print(
                        f"\n[warn] {file_path} validation failed; saved last version.\n"
                        f"{last_feedback[:2000]}"
                    )
                    break

                if used_template_fallback:
                    break

                repair_attempt += 1
                pbar.set_description(
                    f"Scenario {idx + 1}: Scenic LLM repair {repair_attempt}/{max_repair_rounds}"
                )
                try:
                    scenic_code = repair_scenic_with_llm(
                        scenic_code, last_feedback, phase1_dict
                    )
                except Exception as e:
                    pbar.set_postfix(repair_err=str(e))

            pbar.set_postfix({})


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=("huggingface", "hf", "openai"),
        default="huggingface",
        help="LLM provider: local HuggingFace weights or OpenAI-compatible API.",
    )
    parser.add_argument(
        "-l",
        "--llm_name",
        type=str,
        default="../weights/QwQ-32B",
        help="HuggingFace model path/name, or OpenAI model id (e.g. gpt-4o-mini).",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (fallback: env OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL (fallback: env OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for OpenAI backend.",
    )
    parser.add_argument(
        "--openai-max-tokens",
        type=int,
        default=None,
        help="Optional max tokens for OpenAI backend.",
    )
    parser.add_argument(
        "--embed-backend",
        type=str,
        choices=("huggingface", "hf", "openai"),
        default=None,
        help="RAG embedding provider. Defaults to openai when --llm-backend=openai.",
    )
    parser.add_argument(
        "-e",
        "--embed_name",
        type=str,
        default=None,
        help="Embedding model: HF repo/path or OpenAI model (default: text-embedding-3-small / ../weights/bge-m3).",
    )
    parser.add_argument(
        "-id",
        "--scenario_ids",
        nargs="*",
        type=int,
        default=[1, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=4,
        help="Max LLM repair calls after the first validation failure (re-validate after each repair).",
    )
    parser.add_argument(
        "--scenario-route-id",
        type=int,
        default=1,
        help="route_id when loading from scenic_route.pickle (consistent with Safebench).",
    )
    parser.add_argument(
        "--route-pickle",
        type=str,
        default=None,
        help="Optional: explicitly specify the scenic_route.pickle path.",
    )
    parser.add_argument(
        "--no-scenic-sample",
        action="store_true",
        help="Only run scenarioFromFile, skip scenario.generate (faster but weaker validation).",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Skip LLM Scenic generation; use Phase 1 semantics + built-in template only.",
    )
    parser.add_argument(
        "--no-template-fallback",
        action="store_true",
        help="Do not fall back to built-in/repo Scenic templates when LLM fails.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    embed_backend = resolve_embed_backend(opt.embed_backend, opt.llm_backend)
    embed_name = opt.embed_name or default_embed_name(embed_backend)
    run(
        llm_name=opt.llm_name,
        embed_name=embed_name,
        scenario_ids=opt.scenario_ids,
        max_repair_rounds=opt.max_repair_rounds,
        scenario_route_id=opt.scenario_route_id,
        route_pickle=opt.route_pickle,
        scenic_try_sample=not opt.no_scenic_sample,
        llm_backend=opt.llm_backend,
        embed_backend=opt.embed_backend,
        openai_api_key=opt.openai_api_key,
        openai_base_url=opt.openai_base_url,
        openai_temperature=opt.openai_temperature,
        openai_max_tokens=opt.openai_max_tokens,
        use_template_fallback=not opt.no_template_fallback,
        template_only=opt.template_only,
    )
