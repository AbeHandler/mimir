import datasets
from dataclasses import dataclass, field, asdict
from typing import List
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)  # Making the class immutable (strict)
class DatasetInstance:
    member: str = field(metadata={"type": str})
    nonmember: str = field(metadata={"type": str})
    member_neighbors: List[str] = field(default_factory=list, metadata={"type": list})
    nonmember_neighbors: List[str] = field(default_factory=list, metadata={"type": list})

    def __post_init__(self):
        # Enforce strict type checking
        if not isinstance(self.member, str):
            raise TypeError(f"Expected 'member' to be str, got {type(self.member).__name__}")
        if not isinstance(self.nonmember, str):
            raise TypeError(f"Expected 'nonmember' to be str, got {type(self.nonmember).__name__}")
        if not all(isinstance(n, str) for n in self.member_neighbors):
            raise TypeError("All 'member_neighbors' must be strings")
        if not all(isinstance(n, str) for n in self.nonmember_neighbors):
            raise TypeError("All 'nonmember_neighbors' must be strings")

    def __str__(self):
        return (f"Member: {self.member[:50]}...\n"
                f"Nonmember: {self.nonmember[:50]}...\n"
                f"Number of member neighbors: {len(self.member_neighbors)}\n"
                f"Number of nonmember neighbors: {len(self.nonmember_neighbors)}\n")

    def print_neighbors(self):
        print("Member Neighbors:")
        for neighbor in self.member_neighbors[:5]:  # Limit to first 5 for brevity
            print(f"- {neighbor[:50]}...")
        print("\nNonmember Neighbors:")
        for neighbor in self.nonmember_neighbors[:5]:  # Limit to first 5 for brevity
            print(f"- {neighbor[:50]}...")

    @classmethod
    def from_row(cls, row: dict):
        """
        Create a DatasetInstance from a single row.
        
        Args:
            row (dict): A single row from the dataset.
            
        Returns:
            DatasetInstance: An instance with populated fields.
        """
        try:
            instance = cls(
                member=row.get("member", ""),
                nonmember=row.get("nonmember", ""),
                member_neighbors=row.get("member_neighbors", []),
                nonmember_neighbors=row.get("nonmember_neighbors", [])
            )
            return instance
        except TypeError as e:
            raise ValueError(f"Invalid data format: {e}")

    @classmethod
    def to_hf_dataset(cls,
                      train_instances: List['DatasetInstance'], 
                      test_instances: List['DatasetInstance']):
        """
        Convert a list of DatasetInstance objects to a Hugging Face Dataset.

        Args:
            instances (List[DatasetInstance]): List of dataset instances.

        Returns:
            Dataset: A Hugging Face Dataset object.
        """
        train_instances = Dataset.from_list([asdict(instance) for instance in train_instances])
        test_instances = Dataset.from_list([asdict(instance) for instance in test_instances])
        
        return DatasetDict({"train": train_instances, 
                            "test":  test_instances})


if __name__ == "__main__":

    dataset_name = "iamgroot42/mimir"
    name = "github"
    split = "ngram_13_0.8"

    # Load the dataset as a stream
    ds = load_dataset(dataset_name, name=name, split=split, streaming=True)

    instances = []
    for i, row in enumerate(ds):
        instance = DatasetInstance.from_row(row)
        instances.append(instance)
        if i >= 10:  # Limit to 3 instances for demonstration
            break

    # Step 3: Convert the list of instances to a Hugging Face Dataset
    hf_dataset = DatasetInstance.to_hf_dataset(train_instances=instances[0:5],
                                               test_instances=instances[5:])

    repo_id = "abehandler/tmpdata"  # Replace with your username
    hf_dataset.push_to_hub(repo_id)
    print("\nHugging Face Dataset:")
    print(hf_dataset)
