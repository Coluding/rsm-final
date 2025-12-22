"""
Generate cross-product action space JSON files for different location counts.

This utility creates action space configuration files needed for training with
different numbers of locations (8, 15, or any custom number).

Usage:
    python src/utils/generate_action_space.py --num_locations 15
    python src/utils/generate_action_space.py --num_locations 8 --output custom_path.json
"""

import json
import argparse
import os
from pathlib import Path


class ActionSpaceGenerator:
    """Generates cross-product action space for variable location counts."""

    @staticmethod
    def generate(num_locations: int, output_path: str = None):
        """
        Generate action space JSON for specified number of locations.

        Args:
            num_locations: Number of server locations
            output_path: Optional custom output path

        Returns:
            Path to generated JSON file
        """
        # Calculate total actions: all (add, remove) pairs where add != remove, plus no-op
        total_actions = num_locations * (num_locations - 1) + 1

        action_space = list(range(total_actions))
        action_mapping = {}
        inv_action_mapping = {}

        counter = 0
        # Generate all valid (add, remove) pairs
        for add_loc in range(num_locations):
            for remove_loc in range(num_locations):
                if add_loc != remove_loc:
                    action_mapping[f"({add_loc}, {remove_loc})"] = counter
                    inv_action_mapping[str(counter)] = [add_loc, remove_loc]
                    counter += 1

        # Add no-op action
        action_mapping["(-1, -1)"] = counter
        inv_action_mapping[str(counter)] = [-1, -1]

        data = {
            "action_space": action_space,
            "action_mapping": action_mapping,
            "inv_action_mapping": inv_action_mapping,
            "num_location": num_locations
        }

        # Determine output path
        if output_path is None:
            output_path = f"src/data/action_space_{num_locations}.json"

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write JSON file
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Generated action space for {num_locations} locations:")
        print(f"  - Total actions: {total_actions}")
        print(f"  - Valid swap pairs: {num_locations * (num_locations - 1)}")
        print(f"  - No-op actions: 1")
        print(f"  - Output: {output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-product action space for RL training"
    )
    parser.add_argument(
        "--num_locations",
        type=int,
        required=True,
        help="Number of server locations (e.g., 8, 15)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON file (default: src/data/action_space_{num_locations}.json)"
    )

    args = parser.parse_args()

    if args.num_locations < 2:
        raise ValueError("Number of locations must be at least 2")

    ActionSpaceGenerator.generate(args.num_locations, args.output)


if __name__ == "__main__":
    main()