import re
import pandas as pd
import numpy as np
import llama_parser
import contextlib
import io


_REGEX_CONVERTER = None


def _get_regex_converter():
    global _REGEX_CONVERTER
    if _REGEX_CONVERTER is None:
        # No model pipeline is needed for template normalization utilities.
        with contextlib.redirect_stdout(io.StringIO()):
            _REGEX_CONVERTER = llama_parser.LogParser(
                pipeline=None,
                regex_manager1=None,
                model="Meta-Llama-3-8B-Instruct",
                regex_sample=1,
                similarity="jaccard",
                do_self_reflection="False",
            )
    return _REGEX_CONVERTER


def canonicalize_regex(regex):
    if not regex:
        return ""
    regex = regex.strip()
    if regex.startswith("^"):
        regex = regex[1:]
    if regex.endswith("$"):
        regex = regex[:-1]
    regex = regex.replace("\\ ", " ")
    regex = re.sub(r"\s+", " ", regex)
    return regex.strip()


def template_to_normalized_regex(template):
    if pd.isna(template):
        return ""
    converter = _get_regex_converter()
    regex = converter.template_to_regex(str(template))
    return canonicalize_regex(converter.clean_regex(log=None, regex=regex))


def normalize_parsed_regex(regex):
    if pd.isna(regex):
        return ""
    converter = _get_regex_converter()
    return canonicalize_regex(converter.clean_regex(log=None, regex=str(regex)))

# file_df -> parsed log, file2_df -> gt log
def sort_csv_by_content_order(file1_df, file2_df, to_file, save_sorted=False):
    file1_df_unique = file1_df.drop_duplicates(subset='Content', keep='first')
    merged_df = pd.merge(file2_df[['Content']], file1_df_unique, on='Content', how='left')
    if save_sorted:
        merged_df.to_csv(to_file, index=False)
    return merged_df

def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (
            parsed_eventId,
            series_groundtruth_logId_valuecounts.index.tolist(),
        )
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if (
                    logIds.size
                    == series_groundtruth[series_groundtruth == groundtruth_eventId].size
            ):
                accurate_events += logIds.size
                error = False
        if error and debug:
            print(
                "(parsed_eventId, groundtruth_eventId) =",
                error_eventIds,
                "failed",
                logIds.size,
                "messages",
            )
    precision = 0
    recall = 0
    f_measure = 0
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy

def evaluate_result_dataframes(df_parsedlog, df_gtlog, sorted_file):
    df_parsedlog = df_parsedlog.copy()
    df_gtlog = df_gtlog.copy()

    df_parsedlog["RegexTemplate_Normalized"] = df_parsedlog["RegexTemplate"].apply(
        normalize_parsed_regex
    )
    print("df_parsedlog RegexTemplate normalized", flush=True)
    df_gtlog["EventTemplate_Normalized"] = df_gtlog["EventTemplate"].apply(
        template_to_normalized_regex
    )
    print("df_gtlog EventTemplate normalized to regex", flush=True)

    df_debug = pd.concat([df_parsedlog, df_gtlog], axis=1)
    df_debug['CorrectlyParsed'] = df_debug["RegexTemplate_Normalized"].eq(df_debug["EventTemplate_Normalized"])
    df_debug.to_csv(sorted_file.parent / "df_parsedlog_gtlog_combined.csv", index=False)

    correctly_parsed_messages = df_parsedlog["RegexTemplate_Normalized"].eq(
        df_gtlog["EventTemplate_Normalized"]
    ).sum()
    PA = float(correctly_parsed_messages) / len(df_parsedlog[["Content"]])
    print(f"PA: {PA}", flush=True)

    (precision, recall, f_measure, GA) = get_accuracy(
        df_gtlog["EventId"], df_parsedlog["RegexTemplate_Normalized"]
    )
    print(f"GA: {GA}", flush=True)
    event_count = str(df_parsedlog["RegexTemplate_Normalized"].nunique())
    return (
        GA,
        PA,
        event_count,
    )


def evaluate_result(predic_file, gt_file, sorted_file, save_sorted=False,sort=True):
    column_names = ["Content", "RegexTemplate", "EventId"]
    if sort:
        df_parsedlog = pd.read_csv(
            predic_file
            , usecols=column_names, dtype=str
        )
        df_gtlog = pd.read_csv(
            gt_file, usecols=["Content", "EventId", "EventTemplate"], dtype=str
        )
        df_parsedlog = sort_csv_by_content_order(df_parsedlog, df_gtlog, sorted_file, save_sorted)
        print("df_parsedlog sorted! ", flush=True)
    else:
        df_parsedlog = pd.read_csv(
            sorted_file
            , usecols=column_names, dtype=str
        )
        df_gtlog = pd.read_csv(
            gt_file, usecols=["Content", "EventId", "EventTemplate"], dtype=str
        )
        print("df_parsedlog sorted file loaded! ", flush=True)
    return evaluate_result_dataframes(df_parsedlog, df_gtlog, sorted_file)
    
def clean_content(content):
    content = content.replace(",", "")
    content = content.replace(".", "")
    segments = content.split(" ")
    cleaned_segments = ["" if "<*>" in segment else segment for segment in segments]
    return "".join(cleaned_segments)

def clean_regex_content(content):
    if pd.isna(content):
        return ""
    content = content.replace("</s>", '')
    pattern = r"\((?:\?P<[^>]+>)?(?:\\.|[^()\\])*?\)"
    segments = smart_split(content)
    cleaned_segments = []
    for segment in segments:
        if not (re.search(pattern, segment) or re.search(r"^([+\.?()\[\]{}]+)$", segment) or re.search(r'a-z',
                                                                                                     segment) or re.search(
            r'a-f', segment) or re.search(
            r'\\w', segment) or re.search(r'\\d', segment)):
            cleaned_segments.append(segment)
    result = "".join(cleaned_segments)
    if result.startswith("^"):
        result = result[1:]
    if result.endswith("$"):
        result = result[:-1]
    result = result.replace("\\", "")
    result = result.replace(".", "")
    return result

def smart_split(input_string):
    initial_segments = input_string.split(" ")
    final_segments = []
    for segment in initial_segments:
        if "\\s+" in segment:
            sub_segments = re.split(r"\\s\+", segment)
            final_segments.extend(sub_segments)
        elif "\\s*" in segment:
            sub_segments = re.split(r"\\s\*", segment)
            final_segments.extend(sub_segments)
        elif "\\s" in segment:
            sub_segments = re.split(r"\\s", segment)
            final_segments.extend(sub_segments)
        else:
            final_segments.append(segment)
    return final_segments