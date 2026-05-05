import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import accuracy
import grouping
import llama_parser
import regex_manager


parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="nanohttpd")
parser.add_argument(
    "--model",
    type=str,
    default="../models/Meta-Llama-3-8B-Instruct",
)
parser.add_argument("--sample", type=str, default="3")
parser.add_argument("--similarity", type=str, default="jaccard")
parser.add_argument("--do_self_reflection", type=str, default="True")
parser.add_argument("--smoke_lines", type=int, default=0)
args = parser.parse_args()

model_path = args.model
similarity = args.similarity
regex_sample = int(args.sample)
do_self_reflection = args.do_self_reflection
smoke_lines = args.smoke_lines
LOGBASE_ROOT = Path("../LogBase")
RESULT_ROOT = Path(
    "../result_offline_similar_logbase_smoke"
    if smoke_lines > 0
    else "../result_offline_similar_logbase"
)


def resolve_projects(project_arg):
    if project_arg == "all":
        return sorted(
            directory.name
            for directory in LOGBASE_ROOT.iterdir()
            if directory.is_dir()
            and next(directory.glob("*.GeneralAnnotation.csv"), None) is not None
        )
    return [project.strip() for project in project_arg.split(",") if project.strip()]


def find_project_csv(project):
    project_dir = LOGBASE_ROOT / project
    if not project_dir.is_dir():
        raise FileNotFoundError(f"LogBase project directory not found: {project_dir}")

    matches = sorted(project_dir.glob("*.GeneralAnnotation.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one GeneralAnnotation CSV for {project}, found {len(matches)}"
        )
    return matches[0]


def group_logs_using_parser(grouped_logs):
    df = pd.DataFrame(grouped_logs, columns=["Content", "EventId", "EventTemplate"])
    grouped = df.groupby("EventId")
    groups_dict = {}
    for name, group in grouped:
        groups_dict[name] = group.to_dict("records")
    return groups_dict


def get_logs_from_group(group_list):
    return [entry["Content"] for entry in group_list]


def res_list_to_file(res_list, out_path, sample_size):
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{sample_size}.csv"
    file_exists = file_path.is_file()

    with open(file_path, "a", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        if not file_exists:
            writer.writerow(["Content", "EventId", "RegexTemplate"])
        writer.writerows(res_list)

    return str(file_path)


def reorder_csv_in_place(csv_path, order_list):
    data = []
    with open(csv_path, mode="r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        data = list(reader)

    rows_by_key = {row[0]: row for row in data if row}
    sorted_data = [rows_by_key[key] for key in order_list if key in rows_by_key]

    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(sorted_data)


def prepare_results(output_dir, parser_name, sample_size, list_to_insert, order_list):
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"summary_[parser={parser_name},sample_size={sample_size}].csv"

    if not result_file.exists() or result_file.stat().st_size == 0:
        with open(result_file, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "Dataset",
                    "Total_time",
                    "LLaMA_parsing_time",
                    "Drain_parsing_time",
                    "Regex_parsing_time",
                    "GA",
                    "PA",
                    "Event_count",
                ]
            )

    with open(result_file, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(list_to_insert)
    reorder_csv_in_place(result_file, order_list)
    return str(result_file)


def sort_dict_by_content_length(input_dict):
    def count_words_in_content(entry):
        return len(entry["Content"].split())

    sorted_items = sorted(
        input_dict.items(), key=lambda item: count_words_in_content(item[1][0])
    )
    return {key: value for key, value in sorted_items}


def append_unique_to_csv(data_list, file_path):
    new_data = pd.DataFrame(data_list)
    file = Path(file_path)

    if "Count" in new_data.columns:
        new_data = new_data.drop(columns="Count")
    new_data = new_data.groupby(new_data.columns.tolist(), as_index=False).size()
    new_data = new_data.rename(columns={"size": "Count"})

    if file.is_file():
        existing_data = pd.read_csv(file_path, dtype={1: str})
    else:
        existing_data = pd.DataFrame(columns=new_data.columns)

    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    combined_data.to_csv(file_path, index=False, header=True)
    return file_path


def load_logbase_ground_truth(ground_truth_file, max_rows=None):
    df_gtlog = pd.read_csv(
        ground_truth_file, usecols=["Content", "EventTemplate"], dtype=str
    )
    if max_rows and max_rows > 0:
        df_gtlog = df_gtlog.head(max_rows).copy()
    df_gtlog["Content"] = df_gtlog["Content"].fillna("")
    df_gtlog["EventTemplate"] = df_gtlog["EventTemplate"].fillna("")
    df_gtlog["EventId"] = pd.factorize(df_gtlog["EventTemplate"], sort=False)[0].astype(str)
    return df_gtlog


def evaluate_logbase_result(
    predic_file, df_gtlog, sorted_file, save_sorted=False, sort=True
):
    column_names = ["Content", "RegexTemplate", "EventId"]

    if sort:
        df_parsedlog = pd.read_csv(predic_file, usecols=column_names, dtype=str)
        df_parsedlog = accuracy.sort_csv_by_content_order(
            df_parsedlog, df_gtlog, sorted_file, save_sorted
        )
        print("df_parsedlog sorted! ", flush=True)
    else:
        df_parsedlog = pd.read_csv(sorted_file, usecols=column_names, dtype=str)
        print("df_parsedlog sorted file loaded! ", flush=True)
    return accuracy.evaluate_result_dataframes(df_parsedlog, df_gtlog, sorted_file)


if __name__ == "__main__":
    datasets_full = resolve_projects(args.project)
    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            "../models/chatglm3-6b", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "../models/chatglm3-6b", trust_remote_code=True, device="cuda"
        )
        model = model.eval()
        pipeline = (model, tokenizer)
    else:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    print(f"{model_path} Pipeline is ready.", flush=True)
    if smoke_lines > 0:
        print(f"Smoke test mode enabled: first {smoke_lines} lines per project", flush=True)
    for system in datasets_full:
        print(f"Start Parsing {system}", flush=True)
        ground_truth_file = find_project_csv(system)
        out_path = RESULT_ROOT / system
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        out_path.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()
        tree_parser = grouping.LogParser()
        df_gtlog = load_logbase_ground_truth(ground_truth_file, max_rows=smoke_lines)
        logs = df_gtlog["Content"].tolist()
        grouped_logs = tree_parser.parse(logs)
        groups_dict = sort_dict_by_content_length(group_logs_using_parser(grouped_logs))

        print("==================", flush=True)
        print(
            "initial set grouping finished, start parsing. ",
            len(groups_dict.keys()),
            " groups in total for ",
            len(logs),
            " logs",
            flush=True,
        )
        print("==================", flush=True)

        regex_manager1 = regex_manager.RegexTemplateManager()
        llama_parser1 = llama_parser.LogParser(
            pipeline=pipeline,
            model=model_path,
            regex_manager1=regex_manager1,
            regex_sample=regex_sample,
            similarity=similarity,
            do_self_reflection=do_self_reflection,
        )

        for eventid in tqdm(groups_dict.keys(), desc=f"Processing events {system}"):
            append_unique_to_csv(groups_dict[eventid], out_path / "group.csv")
            logs_from_group = get_logs_from_group(groups_dict[eventid])
            res_list = llama_parser1.parse(groups_dict[eventid], logs_from_group)
            out_file = res_list_to_file(res_list, out_path, sample_size=regex_sample)

        tree_parser.print_time()
        regex_manager1.print_time()
        regex_manager1.print_regex_templates()
        total_time = datetime.now() - start_time
        print(system + " Parsing done. [Time taken: {!s}]".format(total_time), flush=True)

        sorted_file = out_path / f"{regex_sample}_sorted.csv"
        ga, pa, event_count = evaluate_logbase_result(
            out_file, df_gtlog, sorted_file, save_sorted=True
        )
        print("==================", flush=True)
        print(
            system,
            total_time,
            llama_parser1.total_time - regex_manager1.total_time,
            tree_parser.total_time,
            regex_manager1.total_time,
            ga,
            pa,
            event_count,
            flush=True,
        )
        prepare_results(
            output_dir=RESULT_ROOT,
            parser_name="Drain",
            sample_size=regex_sample,
            list_to_insert=[
                system,
                total_time,
                llama_parser1.total_time - regex_manager1.total_time,
                tree_parser.total_time,
                regex_manager1.total_time,
                ga,
                pa,
                event_count,
            ],
            order_list=datasets_full,
        )
        print("==================", flush=True)