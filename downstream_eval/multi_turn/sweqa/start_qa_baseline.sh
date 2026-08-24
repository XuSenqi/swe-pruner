# --- 必填：模型与 API ---
export API_TYPE=openai                          # openai 或 azure
export OPENHANDS_MODEL_NAME=openai/deepseek-v4-pro-0813      # LiteLLM 要带 openai/ 前缀
export OPENAI_BASE_URL=https://api.modelverse.cn/v1
export OPENAI_API_KEY=4ZX0SV1e0dtdfiTw086b59Ef-3EdC-42E4-b280-Cb2d70Ad                 # 覆盖 .env 里的 key

# Azure 时改成下面这组，并设 API_TYPE=azure
# export AZURE_ENDPOINT=...
# export AZURE_API_VERSION=...
# export AZURE_API_KEY=...

# --- 跑哪些题、写到哪 ---
export OPENHANDS_REPOS=reflex                   # 或多个：reflex,streamlink,conan
export BASE_REPO_PATH=./swe-repos
export QUESTIONS_PATH=./questions
export ANSWER_OUTPUT_PATH=./answer-baseline
export TRAJ_OUTPUT_PATH=./traj-baseline
export EXPERIMENT_TYPE=baseline                 # baseline 或 pruner

# --- pruner 才需要；baseline 可以不写，会走 .env ---
#export PRUNER_URL=http://10.10.10.39:6001/prune
#export PRUNE_THRESHOLD=0.5

# --- 可选，不写就用代码默认值 ---
#export MAX_ITERATION_PER_RUN=50
#export MAX_TIME_PER_QUESTION=1800

nohup uv run python openhands-qa/main.py >> qa_baseline.log &
