import json
from typing import List
import numpy as np
import random

class CrossProductActionSpace:
    def __init__(self, num_location: int):
        self.action_space = [i for i in range(num_location * (num_location - 1) + 1)]

        self.action_mapping = {}
        counter = 0
        for loc1 in range(num_location):
            for loc2 in range(num_location):
                if loc1 != loc2:
                    self.action_mapping[(loc1, loc2)] = counter
                    counter += 1

        self.action_mapping[(-1, -1)] = len(self.action_space) - 1

        self.inv_action_mapping = {v: k for k, v in self.action_mapping.items()}
        self.num_location = num_location

    def to_json(self):
        data = {"action_space": self.action_space,
                "action_mapping": {str(k): v for k, v in self.action_mapping.items()},
                "inv_action_mapping": self.inv_action_mapping,
                "num_location": self.num_location}

        with open("action_space.json", "w") as f:
            json.dump(data, f)

    @classmethod
    def from_json(cls, file):
        with open(file, "r") as f:
            action_mapping = json.load(f)

        obj = cls(action_mapping["num_location"])
        obj.action_space = action_mapping["action_space"]
        obj.action_mapping = {eval(k): v for k, v in action_mapping["action_mapping"].items()}
        obj.inv_location_mapping = {int(k): v for k, v in action_mapping["inv_action_mapping"].items()}
        return obj

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.action_mapping[item]

        return self.inv_action_mapping[item]

    def build_add_action_mask(self, addable_locations: List[int]):
        mask = []
        for key in sorted(self.inv_action_mapping.keys()):
            if self.inv_action_mapping[key][0] in addable_locations or self.inv_action_mapping[key][0] == -1:
                mask.append(0)
            else:
                mask.append(-float("inf"))
        return mask

    def build_remove_action_mask(self, removable_locations: List[int]):
        mask = []
        for key in sorted(self.inv_action_mapping.keys()):
            if self.inv_action_mapping[key][1] in removable_locations or self.inv_action_mapping[key][0] == -1:
                mask.append(0)
            else:
                mask.append(-float("inf"))
        return mask


def _estimate_mean_total_latency_from_dict(distance_latencies, active, passive, add_loc, remove_loc, alpha=0.8):
    """
    Approximates the combined mean latency across clients and replicas
    for a hypothetical configuration after swapping add_loc/remove_loc.

    Replica-centered view:
      - For each active replica, compute:
          (1) mean latency to all clients
          (2) mean latency to all other active replicas
      - Then average across replicas.
    """
    # --- Hypothetical configuration ---
    new_active = (active - {remove_loc}) | {add_loc}

    # --- Replica-centered client latencies ---
    replica_client_means = []
    for r in new_active:
        if r not in distance_latencies:
            continue
        client_lats = [lat for c, lat in distance_latencies[r].items() if "Client" in c]
        if client_lats:
            replica_client_means.append(np.mean(client_lats))

    mean_client_latency = np.mean(replica_client_means) if replica_client_means else np.inf

    # --- Replica-centered replica latencies ---
    replica_replica_means = []
    for r1 in new_active:
        if r1 not in distance_latencies:
            continue
        inter_replica_lats = [
            distance_latencies[r1][r2]
            for r2 in new_active
            if r1 != r2 and r2 in distance_latencies[r1]
        ]
        if inter_replica_lats:
            replica_replica_means.append(np.mean(inter_replica_lats))

    mean_replica_latency = np.mean(replica_replica_means) if replica_replica_means else 0.0

    # --- Weighted total latency ---
    total_latency = alpha * mean_client_latency + (1 - alpha) * mean_replica_latency
    return total_latency



def _sample_passive_rotation(env):
    """Randomly rotate one passive node to a new available one."""
    if not env.passive_locations or not env.available_locations:
        return (-1, -1)
    passive_remove = random.choice(list(env.passive_locations))
    passive_add = random.choice(list(env.available_locations - {passive_remove}))
    return (env.inv_loc_mapping[passive_add], env.inv_loc_mapping[passive_remove])
if __name__ == "__main__":
    ca = CrossProductActionSpace(8)
    ca.to_json()
    print(ca)