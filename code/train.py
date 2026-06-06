"""Train PAC-LLM or a configured baseline on the chaotic benchmark."""

from run import train_main


if __name__ == "__main__":
    train_main(evaluate_after_training=False)

