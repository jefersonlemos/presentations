#!/usr/bin/env python3
"""
Operating Systems Dataset Generator

Generates a synthetic dataset of OS users with personalized traits based on OS choice.
Rules:
- Mac users: always insane, mostly nice, not rich (sometimes), always use "bad reason mac"
- Windows users: sometimes insane/rich, sometimes not nice, use "good reason windows"
- Linux users: sometimes insane, mostly nice, use "good reason linux"
"""

import argparse
import csv
import random
import signal
import sys
import time
from faker import Faker

# Initialize faker
fake = Faker()

# Data from dataset_base.txt
OS_TYPES = ["linux", "windows", "mac"]

BAD_REASONS_MAC = [
    "newbies",
    "dummies",
    "don't know what's doing",
    "like the apple",
    "read steve jobs book",
    "feels good using an overpriced product",
    "because people say it's good"
]

GOOD_REASONS_WINDOWS = [
    "productivity",
    "compatibility",
    "gaming",
    "has linux",
    "has evolved since windows 10"
]

GOOD_REASONS_LINUX = [
    "like to develop their own drivers",
    "security",
    "for work",
    "lightweight",
    "GUI has evolved",
    "usability is nice",
    "the Original Linux"
]

YES_NO_VARIATIONS = ["yes", "no", "he/she think so", "not at all", "just crazy"]
INSANE_VARIATIONS = ["yes", "no", "for sure", "couldn't be more lucid"]
NICE_VARIATIONS = ["yes", "no", "sometimes"]


def generate_mac_user():
    """Generate a Mac user profile (always insane, bad reasons)"""
    return {
        "name": fake.name(),
        "country": "brazil" if random.random() < 0.8 else fake.country(),
        "state": fake.state() if random.random() < 0.7 else fake.state_abbr(),
        "age": random.randint(18, 75),
        "os": "mac",
        "rich": random.choices(["no", "yes"], weights=[0.7, 0.3])[0],
        "insane": "yes",  # Always insane
        "is_nice": random.choices(["yes", "no", "sometimes"], weights=[0.6, 0.25, 0.15])[0],
        "reason": random.choice(BAD_REASONS_MAC)
    }


def generate_windows_user():
    """Generate a Windows user profile (sometimes insane/rich, sometimes not nice)"""
    return {
        "name": fake.name(),
        "country": "brazil" if random.random() < 0.7 else fake.country(),
        "state": fake.state() if random.random() < 0.7 else fake.state_abbr(),
        "age": random.randint(18, 75),
        "os": "windows",
        "rich": random.choices(["no", "yes"], weights=[0.6, 0.4])[0],
        "insane": random.choices(["yes", "no"], weights=[0.3, 0.7])[0],
        "is_nice": random.choices(["yes", "no"], weights=[0.65, 0.35])[0],
        "reason": random.choice(GOOD_REASONS_WINDOWS)
    }


def generate_linux_user():
    """Generate a Linux user profile (sometimes insane, mostly nice and good reasons)"""
    return {
        "name": fake.name(),
        "country": "brazil" if random.random() < 0.7 else fake.country(),
        "state": fake.state() if random.random() < 0.7 else fake.state_abbr(),
        "age": random.randint(18, 75),
        "os": "linux",
        "rich": random.choices(["no", "yes"], weights=[0.5, 0.5])[0],
        "insane": random.choices(["yes", "no"], weights=[0.25, 0.75])[0],
        "is_nice": random.choices(["yes", "no", "sometimes"], weights=[0.7, 0.15, 0.15])[0],
        "reason": random.choice(GOOD_REASONS_LINUX)
    }


def generate_user():
    """Generate a user with OS-based personality"""
    os_choice = random.choices(
        OS_TYPES,
        weights=[0.35, 0.35, 0.3]  # Slight bias to Linux/Windows
    )[0]
    
    if os_choice == "mac":
        return generate_mac_user()
    elif os_choice == "windows":
        return generate_windows_user()
    else:  # linux
        return generate_linux_user()


def run_generator(filename, duration_minutes=30):
    """
    Generate dataset and append to file
    
    Args:
        filename: Output file path
        duration_minutes: Maximum duration in minutes (default 30)
    """
    start_time = time.time()
    max_duration = duration_minutes * 60  # Convert to seconds
    
    # Write header if file is empty
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            file_exists = len(content) > 0
    except FileNotFoundError:
        file_exists = False
    
    # Open file in append mode
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['name', 'country', 'state', 'age', 'os', 'rich', 'insane', 'is_nice', 'reason']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if needed
        if not file_exists:
            writer.writeheader()
        
        print(f"Starting dataset generation to '{filename}'")
        print(f"Press Ctrl+C to stop or wait for 30 minutes max")
        print("-" * 60)
        
        row_count = 0
        
        try:
            while True:
                # Check if time limit exceeded
                elapsed = time.time() - start_time
                if elapsed >= max_duration:
                    print(f"\nTime limit reached (30 minutes). Stopping.")
                    break
                
                # Generate and write user
                user = generate_user()
                writer.writerow(user)
                csvfile.flush()  # Ensure immediate write
                
                row_count += 1
                
                # Print progress every 100 rows
                if row_count % 100 == 0:
                    elapsed_min = elapsed / 60
                    print(f"Generated {row_count} rows ({elapsed_min:.1f} min elapsed)")
        
        except KeyboardInterrupt:
            print(f"\n\nInterrupted by user.")
    
    print(f"Total rows generated: {row_count}")
    print(f"File saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic OS user dataset"
    )
    parser.add_argument(
        "dataset_output_filename",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Maximum duration in minutes (default: 30)"
    )
    
    args = parser.parse_args()
    
    run_generator(args.dataset_output_filename, args.duration)


if __name__ == "__main__":
    main()
