import pandas as pd
import json


phenomena = {
    "anaphor agreement": [
        "anaphor_gender_agreement",
        "anaphor_number_agreement",
    ],
    "argument structure": [
        "animate_subject_passive",
        "animate_subject_trans",
        "causative",
        "drop_argument",
        "inchoative",
        "intransitive",
        "passive_1",
        "passive_2",
        "transitive",
    ],
    "binding": [
        "principle_A_c_command",
        "principle_A_case_1",
        "principle_A_case_2",
        "principle_A_domain_1",
        "principle_A_domain_2",
        "principle_A_domain_3",
        "principle_A_reconstruction",
    ],
    "control/raising": [
        "existential_there_object_raising",
        "existential_there_subject_raising",
        "expletive_it_object_raising",
        "tough_vs_raising_1",
        "tough_vs_raising_2",
    ],
    "determiner-noun agreement": [
        "determiner_noun_agreement_1",
        "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1",
        "determiner_noun_agreement_irregular_2",
        "determiner_noun_agreement_with_adj_1",
        "determiner_noun_agreement_with_adj_2",
        "determiner_noun_agreement_with_adj_irregular_1",
        "determiner_noun_agreement_with_adj_irregular_2",
    ],
    "ellipsis": [
        "ellipsis_n_bar_1",
        "ellipsis_n_bar_2",
    ],
    "filler gap": [
        "wh_questions_object_gap",
        "wh_questions_subject_gap",
        "wh_questions_subject_gap_long_distance",
        "wh_vs_that_no_gap",
        "wh_vs_that_no_gap_long_distance",
        "wh_vs_that_with_gap",
        "wh_vs_that_with_gap_long_distance",
    ],
    "irregular forms": [
        "irregular_past_participle_adjectives",
        "irregular_past_participle_verbs",
    ],
    "island effects": [
        "adjunct_island",
        "complex_NP_island",
        "coordinate_structure_constraint_complex_left_branch",
        "coordinate_structure_constraint_object_extraction",
        "left_branch_island_echo_question",
        "left_branch_island_simple_question",
        "sentential_subject_island",
        "wh_island",
    ],
    "npi licensing": [
        "matrix_question_npi_licensor_present",
        "npi_present_1",
        "npi_present_2",
        "only_npi_licensor_present",
        "only_npi_scope",
        "sentential_negation_npi_licensor_present",
        "sentential_negation_npi_scope",
    ],
    "quantifiers": [
        "existential_there_quantifiers_1",
        "existential_there_quantifiers_2",
        "superlative_quantifiers_1",
        "superlative_quantifiers_2",
    ],
    "subject-verb agreement": [
        "distractor_agreement_relational_noun",
        "distractor_agreement_relative_clause",
        "irregular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_2",
        "regular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_2",
    ],
}

dataname_to_phenomenon = {
    dataname: phenomenon
    for phenomenon, datanames in phenomena.items()
    for dataname in datanames
}


result_files = {
    "Llama-3 8B": "eval_output.json",
    "Minnow_0": "eval_output_minnow_seed_0.json",
    "Minnow_1": "eval_output_minnow_seed_1.json",
    "Minnow_2": "eval_output_minnow_seed_2.json",
}


results = {}
for result_name, result_file in result_files.items():
    with open(result_file, "r") as f:
        res = json.load(f)
    res["determiner_noun_agreement_with_adj_1"] = res["determiner_noun_agreement_with_adjective_1"]
    del res["determiner_noun_agreement_with_adjective_1"]
    results[result_name] = pd.Series(res)

df = pd.DataFrame(results)
df = df * 100
mean_row = df.mean()
mean_row.name = 'Mean'
df = pd.concat([df, mean_row.to_frame().T])

cols = ["Minnow_0", "Minnow_1", "Minnow_2"]
means = df[cols].mean(axis=1)
stds = df[cols].std(axis=1)
df["Minnow"] = [f"{m:.1f}({s:.1f})" for m, s in zip(means, stds)]
df = df.drop(columns=cols)

df.index.name = "UID"
phen = df.index.map(dataname_to_phenomenon)
df["Phenomenon"] = phen
df = df.set_index("Phenomenon", append=True)
df = df.reorder_levels(["Phenomenon", "UID"])
df = df.sort_index(level=[0, 1])

print(df.to_latex(index=True, float_format="{:.1f}".format).replace(r"_", r"\_"))