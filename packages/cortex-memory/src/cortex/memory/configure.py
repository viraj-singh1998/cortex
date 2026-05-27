"""Interactive CLI for selecting and persisting the LLM provider preference.

Run once after installation:
    cortex-memory-configure

Reconfigure at any time by running the same command, or override for a single
session by setting the CORTEX_LLM_PROVIDER environment variable.
"""

from ._client import write_config, read_config

_PROVIDERS = [
    ("openai",     "OpenAI",     "OPENAI_API_KEY"),
    ("anthropic",  "Anthropic",  "ANTHROPIC_API_KEY"),
    ("openrouter", "OpenRouter", "OPENROUTER_API_KEY — access 200+ models via one key"),
]


def run() -> None:
    current = read_config().get("provider")

    print("\ncortex-memory — LLM provider setup")
    print("─" * 38)
    if current:
        print(f"Current provider: {current}")
    print("\nSelect a provider:\n")
    for i, (_, label, env_var) in enumerate(_PROVIDERS, 1):
        print(f"  {i}. {label:<12} requires {env_var}")
    print()

    while True:
        raw = input(f"Choice [1-{len(_PROVIDERS)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(_PROVIDERS):
            break
        print(f"  Enter a number between 1 and {len(_PROVIDERS)}.")

    provider_id, label, env_var = _PROVIDERS[int(raw) - 1]
    write_config({**read_config(), "provider": provider_id})

    print(f"\n  Provider set to: {label}")
    print(f"  Make sure {env_var.split(' ')[0]} is set in your environment.\n")
    print("  To reconfigure:  cortex-memory-configure")
    print("  To override once: CORTEX_LLM_PROVIDER=<provider> your-command\n")


if __name__ == "__main__":
    run()
