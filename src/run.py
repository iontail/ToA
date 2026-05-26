import argparse
import json
import logging
import os
from datetime import datetime

from tqdm import tqdm

from agent import LocalLlama
from data_utils import Dataloader
from utils import (
    clean_answer,
    format_cards,
    normalize_card,
    normalize_path,
    p_data,
    parse_json,
    retrieve_card_ids,
    split_text_into_token_chunks,
    truncate_text_by_tokens,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run planned evidence-card reasoning.")
    parser.add_argument("--dataset", type=str, default="DetectiveQA")
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--agent_num", type=int, default=5, help="Number of document chunks")
    parser.add_argument("--repetition_num", type=int, default=1)
    parser.add_argument("--model_path", type=str, default=os.getenv("MODEL_PATH") or os.getenv("TOKENIZER_PATH"))
    parser.add_argument("--max_chunk_tokens", type=int, default=32768)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--verification_rounds", type=int, default=2)
    return parser.parse_args()


def make_agent(args):
    return LocalLlama(id=0, model=args.model_path, max_new_tokens=args.max_new_tokens)


def assign_docs(text, chunk_count, model_path):
    chunks, total_tokens = split_text_into_token_chunks(text, chunk_count, model_path)
    logging.info("Chunked document into %s chunks, total tokens: %s", len(chunks), total_tokens)
    return chunks


def extract_evidence_cards(agent, item, chunks, args):
    cards = []
    for idx, chunk in enumerate(chunks):
        content = (
            f"question: {item['question']}\n"
            f"options: {item['options']}\n"
            f"source_pointer: chunk_{idx}\n"
            f"document chunk:\n{truncate_text_by_tokens(chunk, args.max_chunk_tokens, args.model_path)}"
        )
        response = agent.generate_response([
            {"role": "system", "content": p_data("evidence_card")},
            {"role": "user", "content": content},
        ])
        card = normalize_card(parse_json(response), idx)
        cards.append(card)
        logging.info("Evidence card %s: %s", idx, card)
    return cards


def build_plan(agent, item, cards):
    content = (
        f"question: {item['question']}\n"
        f"options: {item['options']}\n"
        f"evidence cards:\n{format_cards(cards)}"
    )
    response = agent.generate_response([
        {"role": "system", "content": p_data("planner")},
        {"role": "user", "content": content},
    ])
    result = parse_json(response)
    path = normalize_path(result.get("path", []), len(cards), cards)
    logging.info("Traversal path: %s", path)
    return path


def execute_plan(agent, item, chunks, cards, path, args):
    state = {"evidence": "", "answer": "None", "reason": ""}
    trace = []

    for chunk_id in path:
        memory_ids = retrieve_card_ids(cards, chunk_id)
        content = (
            f"question: {item['question']}\n"
            f"options: {item['options']}\n"
            f"current_state: {json.dumps(state, ensure_ascii=False)}\n"
            f"current_card:\n{format_cards(cards, [chunk_id])}\n"
            f"retrieved_memory:\n{format_cards(cards, memory_ids)}\n"
            f"current_chunk:\n{truncate_text_by_tokens(chunks[chunk_id], args.max_chunk_tokens, args.model_path)}"
        )
        response = agent.generate_response([
            {"role": "system", "content": p_data("execute")},
            {"role": "user", "content": content},
        ])
        result = parse_json(response)
        state = {
            "evidence": result.get("evidence", state["evidence"]),
            "answer": clean_answer(result.get("answer", state["answer"])),
            "reason": result.get("reason", ""),
        }
        trace.append({"chunk": chunk_id, "memory": memory_ids, **state})
        logging.info("State after chunk %s: %s", chunk_id, state)

    return trace


def draft_answer(agent, item, cards, trace):
    content = (
        f"question: {item['question']}\n"
        f"options: {item['options']}\n"
        f"evidence cards:\n{format_cards(cards)}\n"
        f"execution trace:\n{json.dumps(trace, ensure_ascii=False)}"
    )
    response = agent.generate_response([
        {"role": "system", "content": p_data("draft")},
        {"role": "user", "content": content},
    ])
    result = parse_json(response)
    return {
        "answer": clean_answer(result.get("answer", "None")),
        "explanation": result.get("explanation", ""),
    }


def verify_answer(agent, item, cards, draft, rounds):
    for _ in range(rounds):
        content = (
            f"question: {item['question']}\n"
            f"options: {item['options']}\n"
            f"draft: {json.dumps(draft, ensure_ascii=False)}\n"
            f"evidence cards:\n{format_cards(cards)}"
        )
        response = agent.generate_response([
            {"role": "system", "content": p_data("verify")},
            {"role": "user", "content": content},
        ])
        result = parse_json(response)
        draft = {
            "answer": clean_answer(result.get("answer", draft["answer"])),
            "explanation": result.get("explanation", draft["explanation"]),
        }
        if str(result.get("status", "")).lower() == "pass":
            break
    return draft


def solve_sample(agent, item, args):
    chunks = assign_docs(item["context"], args.agent_num, args.model_path)
    cards = extract_evidence_cards(agent, item, chunks, args)
    path = build_plan(agent, item, cards)
    trace = execute_plan(agent, item, chunks, cards, path, args)
    draft = draft_answer(agent, item, cards, trace)
    final = verify_answer(agent, item, cards, draft, args.verification_rounds)
    return final, cards, path, trace


def prepare_outputs(args):
    log_dir = "../logs"
    result_dir = "../results"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"llama_{args.dataset}_{datetime.now().strftime('%m-%d-%H-%M')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    return result_dir


def main():
    args = parse_args()
    result_dir = prepare_outputs(args)
    data = Dataloader(args.dataset).get_data(args.sample_num)
    agent = make_agent(args)

    for run_idx in range(args.repetition_num):
        results = []
        result_path = os.path.join(result_dir, f"llama_{args.dataset}_{run_idx}.json")

        for idx, sample in enumerate(tqdm(data, desc="Processing Questions")):
            final, cards, path, trace = solve_sample(agent, sample, args)
            print(f"[Q{idx}] Final Decision: {final['answer']}")
            results.append({
                "question": idx,
                "final_decision": final["answer"],
                "explanation": final["explanation"],
                "label": sample["answer"],
                "path": path,
                "evidence_cards": cards,
                "trace": trace,
            })
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            agent.reset()


if __name__ == "__main__":
    main()
