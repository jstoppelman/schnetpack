from torch import nn as nn
import torch
import sys
from schnetpack import Properties

__all__ = ["AtomisticModel", "PairwiseModel"]


class ModelError(Exception):
    pass


class AtomisticModel(nn.Module):
    """
    Join a representation model with output modules.

    Args:
        representation (torch.nn.Module): Representation block of the model.
        output_modules (list or nn.ModuleList or spk.output_modules.Atomwise): Output
            block of the model. Needed for predicting properties.

    Returns:
         dict: property predictions
    """

    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()
        self.representation = representation
        if type(output_modules) not in [list, nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])
        # For stress tensor
        self.requires_stress = any([om.stress for om in self.output_modules])

    def forward(self, inputs):
        """
        Forward representation output through output modules.
        """
        if self.requires_dr:
            inputs[Properties.R].requires_grad_()
        if self.requires_stress:
            # Generate Cartesian displacement tensor
            displacement = torch.zeros_like(inputs[Properties.cell]).to(
                inputs[Properties.R].device
            )
            displacement.requires_grad = True
            inputs["displacement"] = displacement

            # Apply to coordinates and cell
            inputs[Properties.R] = inputs[Properties.R] + torch.matmul(
                inputs[Properties.R], displacement
            )
            inputs[Properties.cell] = inputs[Properties.cell] + torch.matmul(
                inputs[Properties.cell], displacement
            )

        inputs["representation"] = self.representation(inputs)
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        return outs

class PairwiseModel(nn.Module):
    """
    Join a representation model with output modules.

    Args:
        representation (torch.nn.Module): Representation block of the model.
        output_modules (list or nn.ModuleList or spk.output_modules.Atomwise): Output
            block of the model. Needed for predicting properties.

    Returns:
         dict: property predictions
    """

    def __init__(self, representation, output_modules):
        super(PairwiseModel, self).__init__()
        self.representation = representation
        if type(output_modules) not in [list, nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])
        # For stress tensor
        self.requires_stress = any([om.stress for om in self.output_modules])

    def forward(self, inputs):
        """
        Forward representation output through output modules.
        """
        if self.requires_dr:
            inputs[Properties.R].requires_grad_()
        if self.requires_stress:
            # Generate Cartesian displacement tensor
            displacement = torch.zeros_like(inputs[Properties.cell]).to(
                inputs[Properties.R].device
            )
            displacement.requires_grad = True
            inputs["displacement"] = displacement

            # Apply to coordinates and cell
            inputs[Properties.R] = inputs[Properties.R] + torch.matmul(
                inputs[Properties.R], displacement
            )
            inputs[Properties.cell] = inputs[Properties.cell] + torch.matmul(
                inputs[Properties.cell], displacement
            )

        radial_A, radial_B, ap_A, ap_B, pair_dists = self.representation(inputs)
        
        ZA, ZB = inputs['ZA'], inputs['ZB']

        ZA = ZA.unsqueeze(-1)
        ZA = torch.repeat_interleave(ZA, repeats=ZB.shape[1], dim=2)

        ZB = ZB.unsqueeze(1)
        ZB = torch.repeat_interleave(ZB, repeats=ZA.shape[1], dim=1)

        ZA = torch.reshape(ZA, (ZA.shape[0], ZA.shape[1]*ZA.shape[2]))
        ZB = torch.reshape(ZB, (ZB.shape[0], ZB.shape[1]*ZB.shape[2]))
        
        radial_A = radial_A.unsqueeze(2)
        radial_B = radial_B.unsqueeze(2)

        radial_A = torch.repeat_interleave(radial_A, repeats=radial_B.shape[1], dim=2)
        radial_B = torch.repeat_interleave(radial_B, repeats=radial_A.shape[1], dim=2)

        radial_A = torch.reshape(radial_A, (radial_A.shape[0], radial_A.shape[1]*radial_A.shape[2], radial_A.shape[3]))
        radial_B = torch.transpose(radial_B, 1, 2)
        radial_B = torch.reshape(radial_B, (radial_B.shape[0], radial_B.shape[1]*radial_B.shape[2], radial_B.shape[3]))

        ap_B = torch.transpose(ap_B, 1, 2)
        ap_A = torch.reshape(ap_A, (ap_A.shape[0], ap_A.shape[1]*ap_B.shape[2], ap_A.shape[3]))
        ap_B = torch.reshape(ap_B, (ap_B.shape[0], ap_B.shape[1]*ap_B.shape[2], ap_B.shape[3]))

        inputs["representation"] = (ZA, ZB, pair_dists, radial_A, radial_B, ap_A, ap_B)
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        return outs

