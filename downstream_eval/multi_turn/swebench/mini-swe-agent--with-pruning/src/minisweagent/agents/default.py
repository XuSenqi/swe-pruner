"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model

from typing import Any

from minisweagent.utils.cfq_generator import CFQGenerator, CFQGeneratorConfig
from minisweagent.utils.pruner import PrunerClient, PrunerConfig, PruneResponse, PrunerRequest

_READ_COMMAND_HINTS = ("cat ", "nl ", "grep ", "sed -n", "head ", "tail ")
_FILE_PATH_PATTERN = re.compile(r"(?:/testbed/)?[\w./-]+\.py")


def _resolve_env_placeholders(value: Any):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        return os.getenv(key, "")
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]
    return value

from minisweagent.utils.log import logger


def _is_read_command(command: str) -> bool:
    cmd = command.strip().split("&&")[0].strip()
    if cmd.startswith("sed -i") or cmd.startswith("python") or "pytest" in cmd:
        return False
    return any(hint in cmd for hint in _READ_COMMAND_HINTS)


def _extract_read_paths(command: str) -> list[str]:
    if not _is_read_command(command):
        return []
    paths = _FILE_PATH_PATTERN.findall(command)
    return list(dict.fromkeys(path.removeprefix("/testbed/") for path in paths))


def _prune_fallback_reason(
    result: PruneResponse,
    config: PrunerConfig,
    *,
    min_output_chars: int | None = None,
) -> str | None:
    if result.origin_token_cnt <= 0:
        return None
    threshold_chars = min_output_chars if min_output_chars is not None else config.min_output_chars
    min_origin_tokens = max(threshold_chars // 4, 1)
    if result.origin_token_cnt < min_origin_tokens:
        return "small_output"
    if result.score < config.threshold:
        return "low_score"
    if config.min_keep_ratio > 0:
        keep_ratio = result.left_token_cnt / result.origin_token_cnt
        if keep_ratio < config.min_keep_ratio:
            return "low_keep_ratio"
    return None


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0
    time_limit: float = 0.0
    """Wall-clock time limit in seconds for the whole run. 0 disables it."""
    pruner: dict[str, Any] | None = None
    cfq_generator: dict[str, Any] | None = None
    max_repeat_steps: int = 0


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class TimeLimitExceeded(TerminatingException):
    """Raised when the agent has run for longer than the configured wall-clock time limit."""


class RepeatedAction(TerminatingException):
    """Raised when the LM repeats the same action too many times (dead loop)."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.pruner_client: PrunerClient | None = None
        if self.config.pruner:
            print(f"Using Pruner Config: {self.config.pruner}")
            pruner_cfg = PrunerConfig(**{k: v for k, v in _resolve_env_placeholders(self.config.pruner).items()})
            print(f"Loaded Pruner Config: {pruner_cfg}")
            self.pruner_client = PrunerClient(pruner_cfg)
        self.cfq_generator: CFQGenerator | None = None
        if self.config.cfq_generator:
            cfq_cfg = CFQGeneratorConfig(**{k: v for k, v in _resolve_env_placeholders(self.config.cfq_generator).items()})
            print(f"Loaded CFQ Generator Config: {cfq_cfg}")
            self.cfq_generator = CFQGenerator(cfq_cfg)
        self._file_read_counts: dict[str, int] = {}
        self._last_action: str | None = None
        self._repeat_count: int = 0
        self._start_time: float = time.monotonic()

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self._file_read_counts = {}
        self._last_action = None
        self._repeat_count = 0
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        self._start_time = time.monotonic()
        unparsed_err_cnt = 0
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)
            except Exception as e:
                self.add_message("user", f"Error: {e}")
                unparsed_err_cnt += 1
                if unparsed_err_cnt >= 3:
                    return "Error", f"Unparsed error occurred {unparsed_err_cnt} times: {e}"

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        if 0 < self.config.time_limit and (time.monotonic() - self._start_time) > self.config.time_limit:
            raise TimeLimitExceeded(
                f"Exceeded wall-clock time limit of {self.config.time_limit:.0f}s "
                f"(elapsed {time.monotonic() - self._start_time:.0f}s)"
            )
        response = self.model.query(self.messages)
        
        # Log raw model response for debugging
        logger.debug(f"Raw model response (step {self.model.n_calls}):\n{response.get('content', '')[:2000]}")
        
        # Parse action to extract context_focus_question before adding to messages
        action = self.parse_action(response)
        # Log parsed action for debugging
        logger.debug(f"Parsed action: {action.get('action', '')[:200]}")
        logger.debug(f"Context focus question: {action.get('context_focus_question', 'None')}")
        
        # Add context_focus_question to response extra if present
        if action.get("context_focus_question"):
            if "extra" not in response:
                response["extra"] = {}
            if "parsed_action" not in response["extra"]:
                response["extra"]["parsed_action"] = {}
            response["extra"]["parsed_action"]["context_focus_question"] = action["context_focus_question"]
            
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        message_kwargs: dict[str, Any] = {}
        if output.get("pruned_stats"):
            message_kwargs["pruned_stats"] = output["pruned_stats"]
        if output.get("cfq_stats"):
            message_kwargs["cfq_stats"] = output["cfq_stats"]
        self.add_message("user", observation, **message_kwargs)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action with optional context_focus_question."""
        content = response["content"]
        
        # Extract bash command from code block first
        actions = re.findall(self.config.action_regex, content, re.DOTALL)
        if len(actions) != 1:
            raise FormatError(self.render_template(self.config.format_error_template, actions=actions))
        
        action_text = actions[0].strip()
        
        # Try to extract context_focus_question from HTML comment after bash code block
        context_focus_pattern = r"```\s*(?:bash)?\s*\n.*?\n```\s*(?:<context_focus_question>\s*(.*?)\s*</context_focus_question>)?"
        context_focus_match = re.search(context_focus_pattern, content, re.DOTALL | re.IGNORECASE)
        context_focus_question = None
        if context_focus_match:
            # Try to extract from the match group
            if context_focus_match.lastindex and context_focus_match.group(1):
                context_focus_question = context_focus_match.group(1).strip()
        
        if context_focus_question:
            # Try to extract from the response extra if present
            if "extra" in response and "parsed_action" in response["extra"]:
                if "context_focus_question" in response["extra"]["parsed_action"]:
                    context_focus_question = response["extra"]["parsed_action"]["context_focus_question"]
        if context_focus_question == "":
            context_focus_question = None
        
        return {"action": action_text, "context_focus_question": context_focus_question, **response}

    def _track_repeat(self, action: str) -> None:
        if self.config.max_repeat_steps <= 0:
            return
        if action == self._last_action:
            self._repeat_count += 1
            if self._repeat_count > self.config.max_repeat_steps:
                raise RepeatedAction(f"Repeated the same action {self._repeat_count} times: {action}")
        else:
            self._last_action = action
            self._repeat_count = 1

    def execute_action(self, action: dict) -> dict:
        self._track_repeat(action["action"])
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        self._apply_pruner(action, output)
        return output

    def _extract_reasoning_content(self, message: dict) -> str:
        try:
            msg = message["extra"]["response"]["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return ""
        reasoning = msg.get("reasoning_content") or ""
        if not reasoning:
            reasoning = (msg.get("provider_specific_fields") or {}).get("reasoning_content") or ""
        return reasoning.strip()

    def _collect_prior_reasoning(self) -> list[str]:
        max_steps = self.cfq_generator.config.max_prior_reasoning_steps if self.cfq_generator else 4
        prior_messages = [msg for msg in self.messages if msg["role"] == "assistant"][:-1]
        reasonings: list[str] = []
        for msg in reversed(prior_messages):
            reasoning = self._extract_reasoning_content(msg)
            if reasoning:
                reasonings.append(reasoning)
            if len(reasonings) >= max_steps:
                break
        reasonings.reverse()
        return reasonings

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))

    def _get_prune_skip_reason(self, command: str, text: str, *, min_output_chars: int) -> str | None:
        if not self.pruner_client:
            return None
        config = self.pruner_client.config
        if min_output_chars > 0 and len(text) < min_output_chars:
            return "small_output"
        if config.skip_prune_on_reread:
            for path in _extract_read_paths(command):
                if self._file_read_counts.get(path, 0) >= 1:
                    return "reread"
        return None

    def _record_file_reads(self, command: str) -> None:
        for path in _extract_read_paths(command):
            self._file_read_counts[path] = self._file_read_counts.get(path, 0) + 1

    def _apply_pruner(self, action: dict, output: dict[str, str]) -> None:
        if not self.pruner_client:
            return
        text = output.get("output")
        if not text:
            return
        
        context_focus_question = action.get("context_focus_question")
        cfq_source = "agent" if context_focus_question else None
        agent_min_output_chars = self.pruner_client.config.min_output_chars
        auto_cfq_min_output_chars = (
            self.cfq_generator.config.min_output_chars if self.cfq_generator else agent_min_output_chars
        )
        min_output_chars = agent_min_output_chars if context_focus_question else auto_cfq_min_output_chars

        skip_reason = self._get_prune_skip_reason(
            action["action"], text, min_output_chars=min_output_chars
        )
        if skip_reason in ("small_output", "reread"):
            output["output"] = text
            if context_focus_question:
                output["cfq_stats"] = {
                    "source": cfq_source,
                    "context_focus_question": context_focus_question,
                    "used_prior_context": False,
                    "prune_skipped": skip_reason,
                }
            logger.debug(
                "Skipping CFQ/prune (%s) for command: %s",
                skip_reason,
                action["action"][:120],
            )
            self._record_file_reads(action["action"])
            return

        reasoning_content = self._extract_reasoning_content(action)
        prior_reasoning = self._collect_prior_reasoning() if self.cfq_generator else []
        used_prior_context = False
        if (
            not context_focus_question
            and self.cfq_generator
            and (reasoning_content or prior_reasoning)
            and self.cfq_generator.should_generate(
                action["action"], text, min_output_chars=auto_cfq_min_output_chars
            )
        ):
            _, used_prior_context = self.cfq_generator.build_reasoning_block(
                reasoning_content,
                prior_reasoning,
                min_chars=self.cfq_generator.config.min_reasoning_chars,
            )
            context_focus_question = self.cfq_generator.generate(
                reasoning=reasoning_content,
                command=action["action"],
                prior_reasoning=prior_reasoning,
                existing_cfq=context_focus_question,
            )
            if context_focus_question:
                cfq_source = "cfq_generator"

        if context_focus_question:
            output["cfq_stats"] = {
                "source": cfq_source,
                "context_focus_question": context_focus_question,
                "used_prior_context": used_prior_context,
            }
            logger.debug("Using CFQ (%s): %s", cfq_source, context_focus_question)

        if not context_focus_question:
            output["output"] = text
            self._record_file_reads(action["action"])
            return

        req = PrunerRequest(
            code=text,
            query=context_focus_question,
            threshold=self.pruner_client.config.threshold,
            always_keep_first_frags=False,
            chunk_overlap_tokens=self.pruner_client.config.chunk_overlap_tokens,
        )
        pruned_result: PruneResponse = self.pruner_client.prune(req)
        output["pruned_stats"] = {
            "score": pruned_result.score,
            "origin_token_cnt": pruned_result.origin_token_cnt,
            "left_token_cnt": pruned_result.left_token_cnt,
            "model_input_token_cnt": pruned_result.model_input_token_cnt,
        }
        effective_min_output_chars = (
            agent_min_output_chars if cfq_source == "agent" else auto_cfq_min_output_chars
        )
        fallback_reason = _prune_fallback_reason(
            pruned_result,
            self.pruner_client.config,
            min_output_chars=effective_min_output_chars,
        )
        if pruned_result.error_msg:
            output["output"] = f"[Pruner Error]: {pruned_result.error_msg}\n\nOriginal Output:\n{text}"
        elif fallback_reason:
            output["output"] = text
            output["pruned_stats"]["fallback"] = True
            output["pruned_stats"]["fallback_reason"] = fallback_reason
            logger.debug(
                "Prune fallback (%s): score=%.3f origin=%s left=%s",
                fallback_reason,
                pruned_result.score,
                pruned_result.origin_token_cnt,
                pruned_result.left_token_cnt,
            )
        elif pruned_result.left_token_cnt == pruned_result.origin_token_cnt:
            output["output"] = "All outputs are judged as relevent! Output:\n" + text
        else:
            output["output"] = (
                "Filtered some unrelevant parts judged by your context_focus_question, good try! Filtered Output:\n"
                + pruned_result.pruned_code
            )
        self._record_file_reads(action["action"])