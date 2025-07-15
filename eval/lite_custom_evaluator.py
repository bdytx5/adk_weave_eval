
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

        # Optional Weave logger setup
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
            print("[parts_to_text] parts value:", parts, flush=True)
            if parts is None:
                return ""
            if hasattr(parts, "parts"):
                print("[parts_to_text] has .parts, value:", getattr(parts, 'parts', None), flush=True)
                parts = getattr(parts, 'parts', None)
            elif isinstance(parts, dict):
                print("[parts_to_text] is dict, has .get('parts'):", parts.get("parts"), flush=True)
                parts = parts.get("parts", [])
            if not parts:
                return ""
            out = []
            for p in parts:
                print("[parts_to_text] looping part p=", p, flush=True)
                if isinstance(p, dict):
                    out.append(p.get("text", str(p)))
                elif hasattr(p, "text"):
                    out.append(getattr(p, "text", str(p)))
                else:
                    out.append(str(p))
            joined = "\n".join(out)
            print("[parts_to_text] joined string:", joined, flush=True)
            return joined

        print("Calling EvaluationGenerator.generate_responses...", flush=True)
        eval_case_responses_list = await EvaluationGenerator.generate_responses(
            eval_set=eval_set,
            agent_module_path=agent_module,
            repeat_num=num_runs,
            agent_name=agent_name,
        )
        print("Got eval_case_responses_list, len =", len(eval_case_responses_list), flush=True)
        logs_this_eval = []

        for idx, eval_case_responses in enumerate(eval_case_responses_list):
            print(f"\n[loop] -------- eval_case_responses[{idx}] --------", flush=True)
            pprint.pprint(eval_case_responses, indent=2)
            gt_case = eval_case_responses.eval_case.conversation[0]
            print("[gt_case] =", flush=True)
            pprint.pprint(gt_case, indent=2)

            agent_runs = eval_case_responses.responses  # your shape: list of lists (1-element each)
            print(f"[agent_runs]: type={type(agent_runs)}, len={len(agent_runs)}", flush=True)
            for i, ar in enumerate(agent_runs): print(f"  [agent_runs][{i}] type={type(ar)}, len={len(ar)}", flush=True)
            
            user_input = ""
            try:
                user_content = gt_case.get("user_content") if isinstance(gt_case, dict) else getattr(gt_case, "user_content", None)
                print("[user_content] =", user_content, flush=True)
                parts = user_content.get("parts") if isinstance(user_content, dict) else getattr(user_content, "parts", None)
                print("[user_content.parts] =", parts, flush=True)
                if parts and isinstance(parts, list):
                    part0 = parts[0]
                    if isinstance(part0, dict):
                        user_input = part0.get("text", "")
                    elif hasattr(part0, "text"):
                        user_input = getattr(part0, "text", "")
                print("[user_input] =", user_input, flush=True)
            except Exception as exc:
                print(f"Failed to extract user input for idx {idx}, exception:", exc, flush=True)
                pprint.pprint(gt_case)

            # Now loop properly: agent_runs is a list of lists of Invocation
            for run_idx, agent_run_list in enumerate(agent_runs):
                print(f"\n  [run_idx={run_idx}] agent_run_list:", agent_run_list, flush=True)
                if not agent_run_list or not isinstance(agent_run_list, list):
                    print(f"  Empty or invalid agent_run_list at run_idx {run_idx}", flush=True)
                    continue
                agent_run = agent_run_list[0]
                print(f"    [agent_run]: {agent_run}", flush=True)
                try:
                    # Tool tracking
                    gt_inter = getattr(gt_case, "intermediate_data", None)
                    agent_inter = getattr(agent_run, "intermediate_data", None)
                    print(f"    [gt_inter]:", gt_inter, flush=True)
                    print(f"    [agent_inter]:", agent_inter, flush=True)
                    gt_tool_uses = safe_get_list(safe_get_dict(gt_inter).get("tool_uses", []))
                    agent_tool_uses = safe_get_list(safe_get_dict(agent_inter).get("tool_uses", []))
                    print(f"    [gt_tool_uses]:", gt_tool_uses, flush=True)
                    print(f"    [agent_tool_uses]:", agent_tool_uses, flush=True)

                    gt_tools = set()
                    for t in gt_tool_uses:
                        tool = t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
                        gt_tools.add(str(tool).lower())
                    agent_tools = set()
                    for t in agent_tool_uses:
                        tool = t.get("name", "") if isinstance(t, dict) else getattr(t, "name", "")
                        agent_tools.add(str(tool).lower())
                    print(f"    [gt_tools]:", gt_tools, flush=True)
                    print(f"    [agent_tools]:", agent_tools, flush=True)
                    all_gt_tools_used = gt_tools.issubset(agent_tools)
                    missing_tools = list(gt_tools - agent_tools)
                    extra_tools   = list(agent_tools - gt_tools)
                    print(f"    [all_gt_tools_used]:", all_gt_tools_used, flush=True)
                    print(f"    [missing_tools]:", missing_tools, flush=True)
                    print(f"    [extra_tools]:", extra_tools, flush=True)

                    gt_final = getattr(gt_case, "final_response", None)
                    print(f"    [gt_final]:", gt_final, flush=True)
                    agent_final = getattr(agent_run, "final_response", None)
                    print(f"    [agent_final]:", agent_final, flush=True)
                    gt_text = parts_to_text(gt_final)
                    agent_text = parts_to_text(agent_final)
                    print("    [gt_text]:", gt_text, flush=True)
                    print("    [agent_text]:", agent_text, flush=True)

                    llm_judgement = await LiteAgentEvaluator.llm_judge(gt_text, agent_text)
                    print("    [llm_judgement]:", llm_judgement, flush=True)

                    logs_this_eval.append({
                        "test_case_id": getattr(gt_case, "case_id", idx),
                        "run_idx": run_idx,
                        "all_gt_tools_used": all_gt_tools_used,
                        "missing_tools": missing_tools,
                        "extra_tools": extra_tools,
                        "llm_judgement": llm_judgement,
                        "gt_tools": list(gt_tools),
                        "agent_tools": list(agent_tools),
                        "gt_text": gt_text,
                        "agent_text": agent_text,
                        "input": user_input,
                        "output": agent_text,
                    })

                    # WEAVE logging: log each run!
                    if eval_logger:
                        pred_logger = eval_logger.log_prediction(
                            inputs={
                                "input": f"run_idx {run_idx}: {user_input}",
                                "gt_tools": list(gt_tools),
                                "agent_tools": list(agent_tools),
                                "run_idx": run_idx,
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

                except Exception as e:
                    print(f"    Error in run_idx={run_idx}:", str(e), flush=True)
                    logs_this_eval.append({
                        "test_case_id": f"error_case_{idx}_run{run_idx}",
                        "error": str(e),
                    })

        # --- Optionally log summary to Weave ---
        if eval_logger:
            print("[Weave] Logging summary ...", flush=True)
            eval_logger.log_summary()

        # --- Save results file as before ---
        print(f"\nWriting results to custom_eval_results.json ...", flush=True)
        with open("custom_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(logs_this_eval, f, indent=2, ensure_ascii=False)
        LiteAgentEvaluator._eval_logs.append(logs_this_eval)

        print(f"Wrote {len(logs_this_eval)} results to custom_eval_results.json", flush=True)
        for log in logs_this_eval:
            tcid = log.get("test_case_id")
            run_idx = log.get("run_idx", 0)
            res = log.get("llm_judgement", "error")
            print(f"Case {tcid} (run {run_idx}): Grade: {res if isinstance(res, str) else str(res)[:100]}", flush=True)



            
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