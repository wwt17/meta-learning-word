import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def frac_repr(a, b, prec=2):
    return f"{a}/{b}={a/b:.{prec}%}"


blimp_subsets = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance']


def compute_batch_nll(model, tokenizer, sentences):
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction='none',
    ).view(shift_labels.size())

    nll = loss.sum(dim=1)

    return nll


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--pretrained_model", default="Meta-Llama-3-8B-hf",
        help="Pretrained model name or path to resume from."
    )
    argparser.add_argument(
        "--tokenizer", default="Meta-Llama-3-8B-hf",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size."
    )
    argparser.add_argument(
        "--out", default="eval_output.json",
    )
    args = argparser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model,
        device_map=device,
    )

    accs = {}
    for subset in blimp_subsets:
        print(f"{subset}:")
        dataset = load_dataset("blimp", subset)["train"]
        n_acc = 0
        total = 0
        for i in tqdm.tqdm(range(0, len(dataset), args.batch_size)):
            good_batch = dataset['sentence_good'][i : i + args.batch_size]
            bad_batch = dataset['sentence_bad'][i : i + args.batch_size]
            ll_good = compute_batch_nll(model, tokenizer, good_batch)
            ll_bad = compute_batch_nll(model, tokenizer, bad_batch)
            correct = (ll_good < ll_bad)
            total += len(correct)
            n_acc += correct.sum().item()
        print(f"{subset}: {frac_repr(n_acc, total)}")
        accs[subset] = n_acc / total

    if args.out is not None:
        with open(args.out, "w") as f:
            json.dump(accs, f)