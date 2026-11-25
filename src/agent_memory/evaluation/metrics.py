from __future__ import annotations

import re
import string

from ..core.llm import LLMClient


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text.split()


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization used by MemoryAgentBench."""
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match_norm(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def substring_exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(gold) in normalize_answer(pred))


def metric_max_over_golds(metric_fn, pred: str, golds) -> float:
    """Max score over acceptable answers (handles str / list / nested list)."""
    if isinstance(golds, str):
        golds = [golds]
    scores = []
    for g in golds:
        if isinstance(g, (list, tuple)):
            g = " ".join(map(str, g))
        scores.append(metric_fn(pred, str(g)))
    return max(scores) if scores else 0.0


def bleu1(pred: str, gold: str) -> float:
    """Sentence-level BLEU-1 with nltk smoothing (as in the A-MEM eval code)."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        return float(
            sentence_bleu(
                [gold.split()],
                pred.split(),
                weights=(1.0,),
                smoothing_function=SmoothingFunction().method1,
            )
        )
    except ImportError:
        # unigram precision with brevity penalty, no smoothing
        pred_toks, gold_toks = pred.split(), gold.split()
        if not pred_toks or not gold_toks:
            return 0.0
        common = set(pred_toks) & set(gold_toks)
        prec = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in common) / len(pred_toks)
        bp = min(1.0, len(pred_toks) / len(gold_toks))
        return bp * prec


def f1_set(pred: str, gold: str) -> float:
    """Set-based token F1 (A-MEM eval code) — tokens deduplicated before P/R."""
    pred_toks, gold_toks = set(_normalize(pred)), set(_normalize(gold))
    if not pred_toks or not gold_toks:
        return 0.0
    common = pred_toks & gold_toks
    if not common:
        return 0.0
    p, r = len(common) / len(pred_toks), len(common) / len(gold_toks)
    return 2 * p * r / (p + r)


def exact_match(pred: str, gold: str) -> float:
    return float(" ".join(_normalize(pred)) == " ".join(_normalize(gold)))


def f1_score(pred: str, gold: str) -> float:
    pred_toks, gold_toks = _normalize(pred), _normalize(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = set(pred_toks) & set(gold_toks)
    num_same = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in common)
    if num_same == 0:
        return 0.0
    precision, recall = num_same / len(pred_toks), num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


_JUDGE_SYSTEM = (
    "You are grading a QA system. Given the question, the gold answer and the "
    "predicted answer, reply with exactly 'correct' or 'incorrect'."
)


def llm_judge(llm: LLMClient, question: str, gold: str, pred: str) -> float:
    user = f"Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}"
    verdict = llm.chat(_JUDGE_SYSTEM, user).strip().lower()
    return float(verdict.startswith("correct"))


# ---------------------------------------------------------------------------
# LongMemEval official judge templates (src/evaluation/evaluate_qa.py).
# ---------------------------------------------------------------------------

_LME_TEMPLATE_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_TEMPLATE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. In addition, "
    "do not penalize off-by-one errors for the number of days. If the question asks for "
    "the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still correct. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_TEMPLATE_KUPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, "
    "the response should be considered as correct as long as the updated answer is the "
    "required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_TEMPLATE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired "
    "response. Otherwise, answer no. The model does not need to reflect all the points "
    "in the rubric. The response is correct as long as it recalls and utilizes the "
    "user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\n"
    "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_LME_TEMPLATE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a "
    "model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some "
    "other information is given but the asked information is not.\n\n"
    "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)


def longmemeval_judge_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool = False
) -> str:
    """Official LongMemEval answer-check prompt (per question_type / abstention)."""
    if abstention:
        return _LME_TEMPLATE_ABSTENTION.format(question, answer, response)
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return _LME_TEMPLATE_DEFAULT.format(question, answer, response)
    if task == "temporal-reasoning":
        return _LME_TEMPLATE_TEMPORAL.format(question, answer, response)
    if task == "knowledge-update":
        return _LME_TEMPLATE_KUPDATE.format(question, answer, response)
    if task == "single-session-preference":
        return _LME_TEMPLATE_PREFERENCE.format(question, answer, response)
    return _LME_TEMPLATE_DEFAULT.format(question, answer, response)
