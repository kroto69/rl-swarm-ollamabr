from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import re
import json
import ast
import concurrent.futures
import ollama
from transformers import AutoTokenizer
from code_gen_exp.src.utils.solver_utils import (
    check_eos,
    parse_python_fence,
    parse_response,
    get_solutions,
    get_unittests,
    get_questions,
    get_dataset,
)

@dataclass
class RewardsOllamaConfig:
    model: str = "qwen2.5-coder:1.5b-instruct"
    temperature: float = 0.0
    num_predict: int = 256 


class CodeGenerationRewards:
    def __init__(self, solver_tokenizer_path: str, solver_token_lim: int, ollama_config: RewardsOllamaConfig = RewardsOllamaConfig()):
        self.stage = 0
        self.model = ollama_config.model
        self.temperature = ollama_config.temperature
        self.num_predict = ollama_config.num_predict
        
        # [FIX] Load tokenizer with trust_remote_code for Qwen models
        self.tokenizer = AutoTokenizer.from_pretrained(
            solver_tokenizer_path, 
            padding_side="left", 
            trust_remote_code=True
        )
        
        # [CRITICAL FIX] Fix for Qwen/Llama models where pad_token is missing
        # This removes the "The attention mask is not set" warning
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.solver_token_lim = solver_token_lim

    def _check_syntax(self, code: str) -> bool:
        """
        Fast local check: Is the python code syntactically valid?
        Saves Ollama inference time for garbage code.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False

    def _build_prompt(self, dataset: str, solution_code: str, unit_tests: str, question: str) -> str:
        """
        Builds a prompt optimized for JSON output from the Judge Model.
        """
        base_instruction = (
            "You are an expert code reviewer. Your task is to verify if the provided Python solution logically solves the problem and passes the unit tests.\n"
            "Analyze the code logic carefully against the tests.\n\n"
            "RESPONSE FORMAT:\n"
            "You must output ONLY a JSON object with a single key 'is_correct'.\n"
            "Do not add explanations outside the JSON.\n"
            "```json\n"
            "{\"is_correct\": true}\n"
            "```\n"
            "OR\n"
            "```json\n"
            "{\"is_correct\": false}\n"
            "```"
        )
        
        content = (
            f"--- PROBLEM ---\n{question}\n\n"
            f"--- TESTS ---\n{unit_tests}\n\n"
            f"--- CANDIDATE SOLUTION ---\n{solution_code}"
        )
        
        return f"{base_instruction}\n\n{content}"

    def _extract_json_verdict(self, text: str) -> float:
        """
        Extract verdict from LLM response using regex. 
        Returns 1.0 for True, 0.0 for False/Error.
        """
        try:
            # 1. Try finding fenced JSON
            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
            # 2. Try finding raw JSON object
            if not match:
                match = re.search(r"(\{.*?\})", text, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                # Clean potential trailing commas or comments that break json.loads
                json_str = re.sub(r",\s*}", "}", json_str) 
                data = json.loads(json_str)
                
                # Handle boolean or string "true"/"false"
                val = data.get("is_correct")
                if isinstance(val, bool):
                    return 1.0 if val else 0.0
                if isinstance(val, str):
                    return 1.0 if val.lower() == 'true' else 0.0
                    
        except Exception:
            pass
        
        # Fallback: Keyword search
        text_lower = text.lower()
        if '"is_correct": true' in text_lower or "'is_correct': true" in text_lower:
            return 1.0
        return 0.0

    def _query_ollama_single(self, prompt: str) -> float:
        """Helper for ThreadPool execution"""
        try:
            response = ollama.generate(
                model=self.model, 
                prompt=prompt, 
                options={"temperature": self.temperature, "num_predict": self.num_predict}
            )
            return self._extract_json_verdict(response.response)
        except Exception:
            return 0.0

    def reward_fn(self, dataset, solutions, unittests, question) -> List[float]:
        rewards = []
        prompts_to_process = []
        indices_to_process = []

        # --- STEP 1: PRE-FILTERING (CPU) ---
        for i, solution in enumerate(solutions):
            # Default penalty
            current_reward = 0.0
            
            if not isinstance(solution, str):
                rewards.append(-1.5)
                continue
            
            # Clean and parse code
            parsed_code = parse_python_fence(solution)
            
            # Check EOS (End of Sentence) - Bonus if model finished writing
            eos_found = check_eos(solution, self.tokenizer, self.solver_token_lim)
            eos_bonus = 0.2 if eos_found else -0.2
            
            if not parsed_code:
                # No code found at all
                rewards.append(-1.0 + eos_bonus)
            elif not self._check_syntax(parsed_code):
                # Code exists but has Syntax Error
                rewards.append(-0.5 + eos_bonus)
            else:
                # Syntax is valid, prepare for Judge LLM
                # Initialize with just the EOS bonus, we will add Judge score later
                rewards.append(eos_bonus) 
                
                prompt = self._build_prompt(dataset, parsed_code, str(unittests), str(question))
                prompts_to_process.append(prompt)
                indices_to_process.append(i)

        # --- STEP 2: PARALLEL JUDGING (IO/Network) ---
        if prompts_to_process:
            # Limit max_workers to 4 to match your CPU constraint strategy
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                judge_results = list(executor.map(self._query_ollama_single, prompts_to_process))
            
            # Combine results
            for idx, judge_score in zip(indices_to_process, judge_results):
                # If Judge says Correct (1.0), we add +1.0. If False, +0.0
                rewards[idx] += judge_score

        return rewards

    def __call__(self, game_state):
        solutions_by_agent = get_solutions(game_state, self.stage)
        unittests_by_agent = get_unittests(game_state, self.stage)
        questions = get_questions(game_state, self.stage)
        datasets_by_agent = get_dataset(game_state, self.stage)
        
        rewards = {}  # Key per agent
        try:
            for agent in solutions_by_agent:
                rewards[agent] = {}  
                for batch_id in solutions_by_agent[agent]:
                    rewards[agent][batch_id] = []
                    
                    # GRPO structure: We usually have a list of generations for one problem node
                    # But the genrl structure might be [node1_gens, node2_gens...]
                    # We iterate nodes
                    
                    for node_idx, _ in enumerate(solutions_by_agent[agent][batch_id]):
                        # Extract data for this specific node/problem
                        sol_list = solutions_by_agent[agent][batch_id][node_idx] # List of K generations
                        
                        # Handle cases where metadata might be singular or list
                        # We assume unittests/questions are same for all K generations of this node
                        
                        curr_unittests = unittests_by_agent[agent][batch_id][node_idx]
                        curr_question = questions[agent][batch_id][node_idx]
                        curr_dataset = datasets_by_agent[agent][batch_id][node_idx]

                        # Calculate rewards for the whole list of K generations at once
                        node_rewards = self.reward_fn(
                            curr_dataset, 
                            sol_list, 
                            curr_unittests, 
                            curr_question
                        )
                        
                        rewards[agent][batch_id].append(node_rewards)
                        
            return rewards
        except Exception as e:
            # Log error properly in production, print for now
            print(f"[Reward Error]: {e}")
            return {}
