#!/usr/bin/env python3
import argparse
import csv
import random
import time
from datetime import datetime, timedelta

from faker import Faker

# --- Config ---

MAX_DURATION_SECONDS = 30 * 60  # 30 minutes

fake = Faker()
fake_br = Faker("pt_BR")  # Brazilian names


OS_CHOICES = ["linux", "windows", "mac"]

RICH_VALUES = ["yes", "no", "Believe so", "not at all", "just crazy"]
INSANE_VALUES = ["yes", "no", "for sure", "couldn't be more lucid"]
IS_NICE_VALUES = ["yes", "no", "sometimes"]


# -------------------------
# FIELD GENERATORS
# -------------------------

def random_name() -> str:
    # 70% Brazilian names, 30% global
    if random.random() < 0.3:
        return fake_br.name()
    return fake.name()


def random_country() -> str:
    # 70% brazil, 30% other countries
    if random.random() < 0.3:
        return "brazil"
    return fake.country().lower()


def random_state(country: str) -> str:
    """
    Generate realistic states depending on the country.
    - Brazil -> use pt_BR states
    - USA -> US state
    - Else -> fallback to faker administrative region
    """
    if country == "brazil":
        return fake_br.estado_nome().lower()   # real Brazilian state
    elif country == "usa":
        return fake.state().lower()
    else:
        # fallback generic
        return fake.state().lower()


def random_age() -> int:
    return random.randint(18, 70)


def pick_insane(is_insane: bool) -> str:
    if is_insane:
        return random.choice(["yes", "for sure"])
    else:
        return random.choice(["no", "couldn't be more lucid"])


def pick_rich(os_value: str, force_rich: bool = False) -> str:
    if os_value == "mac":
        # mac: mostly NOT rich, sometimes rich
        if force_rich or random.random() < 0.15:
            return "yes"
        return random.choice(["no", "not at all", "just crazy", "Believe so"])
    else:
        # linux/windows: neutral distribution with slight bias
        if force_rich or random.random() < 0.2:
            return "yes"
        return random.choice(["no", "not at all", "just crazy", "Believe so"])


# -------------------------
# RANDOM REASON GENERATOR
# -------------------------

def generate_reason_template(os_value: str) -> str:
    action = fake.word()
    adjective = fake.word()
    tech = fake.word()
    short_phrase = fake.sentence(nb_words=4)
    medium_phrase = fake.sentence(nb_words=7)

    if os_value == "mac":
        template = random.choice([
            f"Chose mac because the {adjective} hardware felt inspiring during {action}",
            f"Uses mac due to {medium_phrase}",
            f"Mac seemed like a good idea after {short_phrase}",
            f"Believes mac helps with {tech}, for some reason",
            f"Mac felt more premium while dealing with {action}",
        ])
    elif os_value == "windows":
        template = random.choice([
            f"Uses windows for improved {action} and more {adjective} workflows",
            f"Windows provides better {tech} support according to {short_phrase}",
            f"Feels more productive on windows while handling {action}",
            f"Windows tooling fits the current {tech} stack",
        ])
    else:  # linux
        template = random.choice([
            f"Linux chosen for {tech} efficiency and stability",
            f"Believes linux improves {action} and reliability",
            f"Uses linux because {medium_phrase}",
            f"Linux gives more control over {tech} and system behavior",
        ])

    return template


# -------------------------
# BUILD 1 ROW
# -------------------------

def generate_row() -> dict:
    name = random_name()
    country = random_country()
    state = random_state(country)
    age = random_age()
    os_value = random.choice(OS_CHOICES)

    # defaults
    is_nice = "yes"
    insane_flag = False
    force_rich = False

    if os_value == "mac":
        insane_flag = True
        r = random.random()
        if r < 0.7:
            is_nice = "yes"
        elif r < 0.85:
            is_nice = "sometimes"
        else:
            is_nice = "no"

    elif os_value == "windows":
        is_nice = "yes"
        if random.random() < 0.2:  # 20% problematic
            if random.random() < 0.5:
                insane_flag = True
            if random.random() < 0.5:
                force_rich = True
            is_nice = "no"

    elif os_value == "linux":
        is_nice = "yes"
        if random.random() < 0.15:
            insane_flag = True

    is_insane = pick_insane(insane_flag)
    is_rich = pick_rich(os_value, force_rich)
    reason = generate_reason_template(os_value)

    return {
        "name": name,
        "country": country,
        "state": state,
        "age": age,
        "os": os_value,
        "is_rich": is_rich,
        "is_insane": is_insane,
        "is_nice": is_nice,
        "reason": reason,
    }


# -------------------------
# CSV HEADER HANDLING
# -------------------------

def ensure_header(writer: csv.DictWriter, file_obj):
    file_obj.seek(0, 2)  # move to end
    if file_obj.tell() == 0:
        writer.writeheader()
        file_obj.flush()


# -------------------------
# MAIN
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate funny OS dataset.")
    parser.add_argument(
        "output_file",
        help="CSV file to append data to (will be created if not exists)",
    )
    args = parser.parse_args()

    start = datetime.now()
    end = start + timedelta(seconds=MAX_DURATION_SECONDS)

    fieldnames = [
        "name",
        "country",
        "state",
        "age",
        "os",
        "is_rich",
        "is_insane",
        "is_nice",
        "reason",
    ]

    with open(args.output_file, "a+", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        ensure_header(writer, f)

        print(f"[INFO] Writing rows to {args.output_file}")
        print(f"[INFO] Running for up to {MAX_DURATION_SECONDS} seconds (~30min) or until Ctrl+C")

        try:
            while datetime.now() < end:
                row = generate_row()
                writer.writerow(row)
                f.flush()
                time.sleep(0.1)  # ~10 lines per second
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user (Ctrl+C).")

        print("[INFO] Done.")


if __name__ == "__main__":
    main()
