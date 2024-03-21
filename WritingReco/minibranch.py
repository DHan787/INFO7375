import torch
import torch.nn as nn


class MiniBranchNetwork(nn.Module):
    def __init__(self, main_network, branches, merge_function):
        """
        Initializes the DNN with a mini-branch architecture.

        :param main_network: A module representing the main network.
        :param branches: A list of modules, each representing a branch.
        :param merge_function: A function to merge the outputs of the main network and branches.
        """
        super(MiniBranchNetwork, self).__init__()
        self.main_network = main_network
        self.branches = nn.ModuleList(branches)
        self.merge_function = merge_function

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: The output of the network after merging the main and branch pathways.
        """
        main_output = self.main_network(x)
        branch_outputs = [branch(x) for branch in self.branches]

        # Merge the main output and branch outputs
        output = self.merge_function(main_output, branch_outputs)
        return output


def example_merge_function(main_output, branch_outputs):
    """
    An example merge function that concatenates the main output with the sum of branch outputs.

    :param main_output: The output tensor from the main network.
    :param branch_outputs: A list of output tensors from the branches.
    :return: Merged output tensor.
    """
    # Summing the branch outputs
    sum_of_branches = torch.sum(torch.stack(branch_outputs), dim=0)
    # Concatenating the main output with the summed branch outputs
    merged_output = torch.cat([main_output, sum_of_branches], dim=1)
    return merged_output

# Example of how you might initialize and use the MiniBranchNetwork
# main_network = YourMainNetwork()
# branches = [YourBranch1(), YourBranch2(), ...]
# network = MiniBranchNetwork(main_network, branches, example_merge_function)
