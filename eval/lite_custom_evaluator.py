
import os
import json
import asyncio
import pprint
from google import genai

# Only import weave if/when needed
try:
    import weave
    from weave import EvaluationLogger
except ImportError:
    weave = None
    EvaluationLogger = None

GENAI_MODEL = "gemini-2.5-flash"

def safe_get_dict(attr, default=None):
    if attr is None:
        return default or {}
    if isinstance(attr, dict):
        return attr
    if hasattr(attr, 'dict'):
        return attr.dict()
    return default or {}

def safe_get_list(attr, default=None):
    if attr is None:
        return default or []
    if isinstance(attr, list):
        return attr
    return default or []

class LiteAgentEvaluator:
    _eval_logs = []

    @staticmethod
    def find_config_for_test_file(test_file: str):
        return {}

    @staticmethod
    async def llm_judge(gt_text, agent_text):
        prompt = (
            "You are a strict but format-agnostic grader for insurance claims. "
            "Given a GROUND TRUTH response and an AGENT response, is the agent "
            "response roughly similar the ground truth in terms of the answer? -- if some minor extra details are missing, that is ok. "
            "Ignore changes in format or wording. Respond YES or NO and justify.\n"
            f"\nGROUND TRUTH RESPONSE:\n{gt_text}\n"
            f"\nAGENT RESPONSE:\n{agent_text}\n"
            "\nResult:"
        )
        try:
            client = genai.Client()
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=GENAI_MODEL,
                contents=[prompt]
            )

            result = getattr(response, "text", None) or str(response)
        except Exception as e:
            result = f"LLM error: {e}"
        return result



    @staticmethod
    async def evaluate_eval_set(
        agent_module: str,
        eval_set: "EvalSet",
        criteria: dict,
        num_runs=1,
        agent_name=None,
        use_weave=True,
        weave_project="adk_eval"
    ):
        from google.adk.evaluation.evaluation_generator import EvaluationGenerator

        # ---- Optional Weave logger setup ----
        if use_weave:
            if weave is None or EvaluationLogger is None:
                raise ImportError("Weave is not installed. Run 'pip install weave'")
            weave.init(weave_project or "adk_eval")
            eval_logger = EvaluationLogger(
                model=f"{agent_name or agent_module}",
                dataset=f"{getattr(eval_set, 'name', 'unknown')}",
            )
        else:
            eval_logger = None

        def parts_to_text(parts):
            if parts is None:
                return ""
            if hasattr(parts, "parts"):
                parts = parts.parts
            elif isinstance(parts, dict):
                parts = parts.get("parts", [])
            if not parts:
                return ""
            out = []
            for p in parts:
                if isinstance(p, dict):
                    out.append(p.get("text", str(p)))
                elif hasattr(p, "text"):
                    out.append(p.text)
                else:
                    out.append(str(p))
            return "\n".join(out)

        eval_case_responses_list = await EvaluationGenerator.generate_responses(
            eval_set=eval_set,
            agent_module_path=agent_module,
            repeat_num=num_runs,
            agent_name=agent_name,
        )
        logs_this_eval = []

        for idx, eval_case_responses in enumerate(eval_case_responses_list):
            try:
                agent_run = eval_case_responses.responses[0][0]
                gt_case = eval_case_responses.eval_case.conversation[0]

                # 1. USER INPUT Extraction:
                # Robustly tries both dict and object style
                input_text = ""
                try:
                    user_content = gt_case.get("user_content") if isinstance(gt_case, dict) else getattr(gt_case, "user_content", None)
                    parts = user_content.get("parts") if isinstance(user_content, dict) else getattr(user_content, "parts", None)
                    if parts and isinstance(parts, list):
                        part0 = parts[0]
                        if isinstance(part0, dict):
                            input_text = part0.get("text", "")
                        elif hasattr(part0, "text"):
                            input_text = part0.text
                except Exception as parse_exc:
                    print("Failed to extract user input, DEBUG DUMP FOLLOWS ----------")
                    pprint.pprint(gt_case)
                    raise parse_exc

                gt_inter = getattr(gt_case, "intermediate_data", None)
                agent_inter = getattr(agent_run, "intermediate_data", None)
                gt_tool_uses = safe_get_list(safe_get_dict(gt_inter).get("tool_uses", []))
                agent_tool_uses = safe_get_list(safe_get_dict(agent_inter).get("tool_uses", []))

                gt_tools = set()
                for t in gt_tool_uses:
                    tool = t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
                    gt_tools.add(str(tool).lower())
                agent_tools = set()
                for t in agent_tool_uses:
                    tool = t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
                    agent_tools.add(str(tool).lower())
                all_gt_tools_used = gt_tools.issubset(agent_tools)
                missing_tools = list(gt_tools - agent_tools)
                extra_tools   = list(agent_tools - gt_tools)

                gt_final = getattr(gt_case, "final_response", None)
                agent_final = getattr(agent_run, "final_response", None)
                gt_text = parts_to_text(gt_final)
                agent_text = parts_to_text(agent_final)

                llm_judgement = await LiteAgentEvaluator.llm_judge(gt_text, agent_text)

                logs_this_eval.append({
                    "test_case_id": getattr(gt_case, "case_id", idx),
                    "all_gt_tools_used": all_gt_tools_used,
                    "missing_tools": missing_tools,
                    "extra_tools": extra_tools,
                    "llm_judgement": llm_judgement,
                    "gt_tools": list(gt_tools),
                    "agent_tools": list(agent_tools),
                    "gt_text": gt_text,
                    "agent_text": agent_text,
                    "input": input_text,
                })

                # ------ WEAVE Logging for this prediction -----
                if eval_logger:
                    pred_logger = eval_logger.log_prediction(
                        inputs={
                            "input": input_text,
                            "gt_tools": list(gt_tools),
                            "agent_tools": list(agent_tools),
                        },
                        output=agent_text
                    )
                    pred_logger.log_score(
                        scorer="llm_judgement",
                        score=llm_judgement
                    )
                    pred_logger.log_score(
                        scorer="llm_correctness",
                        score=llm_judgement.strip().upper().startswith("YES")
                    )
                    pred_logger.log_score(
                        scorer="tool_correctness",
                        score=all_gt_tools_used
                    )
                    if len(gt_tools):
                        tool_recall = len(gt_tools & agent_tools) / len(gt_tools)
                    else:
                        tool_recall = 1.0
                    pred_logger.log_score(
                        scorer="tool_recall",
                        score=tool_recall
                    )
                    pred_logger.finish()
                # ----------------------------------------------

            except Exception as e:
                logs_this_eval.append({
                    "test_case_id": f"error_case_{idx}",
                    "error": str(e),
                })

        # --- Optionally log summary to Weave ---
        if eval_logger:
            summary_stats = {
                "n_passed": sum(
                    1 for log in logs_this_eval
                    if isinstance(log.get("llm_judgement"), str) and log.get("llm_judgement", "").strip().upper().startswith("YES")
                ),
                "n_total": len(logs_this_eval)
            }
            eval_logger.log_summary()

        # --- Save results file as before ---
        with open("custom_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(logs_this_eval, f, indent=2, ensure_ascii=False)
        LiteAgentEvaluator._eval_logs.append(logs_this_eval)
        # Minimal reporting at end
        print(f"Wrote {len(logs_this_eval)} results to custom_eval_results.json")
        for log in logs_this_eval:
            tcid = log.get("test_case_id")
            res = log.get("llm_judgement", "error")
            print(f"Case {tcid}: Grade: {res if isinstance(res, str) else str(res)[:100]}")

    @staticmethod
    async def evaluate(
        agent_module: str,
        eval_dataset_file_path_or_dir: str,
        num_runs: int = 1,
        agent_name: str = None,
        initial_session_file: str = None,
        use_weave: bool = False,
    ):
        from google.adk.evaluation.eval_set import EvalSet
        with open(eval_dataset_file_path_or_dir, "r") as f:
            eval_set_json = json.load(f)
        eval_set = EvalSet.model_validate(eval_set_json)
        criteria = {}
        await LiteAgentEvaluator.evaluate_eval_set(
            agent_module, eval_set, criteria, num_runs, agent_name, use_weave=use_weave
        )