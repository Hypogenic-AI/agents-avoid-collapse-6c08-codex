# Cloned Repositories

## Repo 1: self-correcting-self-consuming
- URL: https://github.com/nate-gillman/self-correcting-self-consuming
- Purpose: official implementation of `2402.07087`
- Location: `code/self-correcting-self-consuming/`
- Key files:
  - `exp_scripts/gaussian_toy_example.py`
  - `exp_scripts/mnist/self_consuming_ddpm_mini.py`
  - `exp_scripts/filter_dataset.py`
  - `exp_scripts/dataset_0064/train_iterative_finetuning_with_correction.sh`
- Notes:
  - Most complete collapse-focused codebase in the workspace.
  - Heavy reproduction path depends on conda, MuJoCo, SMPL/SMPL-X assets, HumanML3D, AMASS, and BMLMoVi.
  - Toy Gaussian and MNIST experiments are the fastest entry points for the experiment runner.

## Repo 2: CAMEL
- URL: https://github.com/camel-ai/camel
- Purpose: large-scale multi-agent LLM framework for role-playing, societies, and synthetic data generation
- Location: `code/camel/`
- Key files:
  - `examples/agents/create_chat_agent.py`
  - `examples/agents/single_agent.py`
  - `examples/observability/agentops_track_roleplaying_with_function.py`
  - `camel/societies/`
- Notes:
  - Useful for controlled sweeps over population size and role diversity.
  - README explicitly frames the project around scaling laws of agents.
  - Large repository with many optional integrations; use minimal examples first.

## Repo 3: generative_agents
- URL: https://github.com/joonspk-research/generative_agents
- Purpose: social simulation environment with persistent memory, planning, and reflection
- Location: `code/generative_agents/`
- Key files:
  - `README.md`
  - `reverie/backend_server/reverie.py`
  - `environment/frontend_server/`
- Notes:
  - Good fit for experiments where individuality is part of the definition of ecosystem health.
  - Requires an API-backed simulation and a browser-accessible environment.
  - Includes base simulations with 3-agent and 25-agent setups.

## Repo 4: lm-evaluation-harness
- URL: https://github.com/EleutherAI/lm-evaluation-harness
- Purpose: standardized evaluation harness for quality tracking across generations
- Location: `code/lm-evaluation-harness/`
- Key files:
  - `README.md`
  - `docs/interface.md`
  - `lm_eval/tasks/`
- Notes:
  - Best reusable evaluation layer for held-out task quality.
  - Supports Hugging Face, vLLM, and API-backed models.
  - Useful for measuring whether diversity-preserving strategies degrade benchmark quality.
