# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic evalualtion for Academic Research"""

import pathlib

import dotenv
import pytest
# from google.adk.evaluation.agent_evaluator import AgentEvaluator
# from custom_evaluator import AgentEvaluator

from lite_custom_evaluator import LiteAgentEvaluator

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_all():
    """Test the agent's basic ability on a few examples."""
    await LiteAgentEvaluator.evaluate(
        "seq_agent",
        "/Users/brettyoung/Desktop/dev25/tutorials/adk/adk-samples/python/agents/seq_agent/seq_agent/insurance_eval_set_cleaned.evalset.json",
        # "/Users/brettyoung/Desktop/dev25/tutorials/adk/adk-samples/python/agents/seq_agent/seq_agent/single_ex_eval.evalset.json",
        # "/Users/brettyoung/Desktop/dev25/tutorials/adk/adk-samples/python/agents/seq_agent/seq_agent/insurance_eval_set.evalset.json",
        # "/Users/brettyoung/Desktop/dev25/tutorials/adk/adk-samples/python/agents/seq_agent/seq_agent/evalset471f4d.evalset.json",
        num_runs=2,
        use_weave=True
    )
