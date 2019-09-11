from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Variable

class RBFInterpolation(object):
    def rbf_interpolation_kernel(self, scaled_grid_dist, sigma=1.5):
        return (scaled_grid_dist/sigma).pow(2.).mul(-0.5).exp()
        
    
    def interpolate(self, x_grid, x_target, interp_points=range(-2, 3)):
        # The default interp_points interpretation in gpytorch is very illogical, flip it
        #interp_points = interp_points[::-1]
        
        # Do some boundary checking
        grid_mins = x_grid.min(1)[0]
        grid_maxs = x_grid.max(1)[0]
        x_target_min = x_target.min(0)[0]
        x_target_max = x_target.max(0)[0]
        lt_min_mask = (x_target_min - grid_mins).lt(-1e-7)
        gt_max_mask = (x_target_max - grid_maxs).gt(1e-7)
        if lt_min_mask.data.sum():
            first_out_of_range = lt_min_mask.nonzero().squeeze(1)[0].data
            raise RuntimeError(
                (
                    "Received data that was out of bounds for the specified grid. "
                    "Grid bounds were ({0:.3f}, {1:.3f}), but min = {2:.3f}, "
                    "max = {3:.3f}"
                ).format(
                    grid_mins[first_out_of_range].data[0],
                    grid_maxs[first_out_of_range].data[0],
                    x_target_min[first_out_of_range].data[0],
                    x_target_max[first_out_of_range].data[0],
                )
            )
        if gt_max_mask.data.sum():
            first_out_of_range = gt_max_mask.nonzero().squeeze(1)[0].data
            raise RuntimeError(
                (
                    "Received data that was out of bounds for the specified grid. "
                    "Grid bounds were ({0:.3f}, {1:.3f}), but min = {2:.3f}, "
                    "max = {3:.3f}"
                ).format(
                    grid_mins[first_out_of_range].data[0],
                    grid_maxs[first_out_of_range].data[0],
                    x_target_min[first_out_of_range].data[0],
                    x_target_max[first_out_of_range].data[0],
                )
            )

        # Now do interpolation
        interp_points_flip = Variable(x_grid.data.new(interp_points[::-1]))
        interp_points = Variable(x_grid.data.new(interp_points))

        num_grid_points = x_grid.size(1)
        num_target_points = x_target.size(0)
        num_dim = x_target.size(-1)
        num_coefficients = len(interp_points)

        interp_values = Variable(x_target.data.new(num_target_points, num_coefficients ** num_dim).fill_(1))
        interp_indices = Variable(x_grid.data.new(num_target_points, num_coefficients ** num_dim).zero_())

        for i in range(num_dim):
            grid_delta = x_grid[i, 1] - x_grid[i, 0]
            lower_grid_pt_idxs = torch.round((x_target[:, i] - x_grid[i, 0]) / grid_delta).squeeze()
            lower_pt_rel_dists = (x_target[:, i] - x_grid[i, 0]) / grid_delta - lower_grid_pt_idxs
            lower_grid_pt_idxs = lower_grid_pt_idxs - interp_points.max()
            lower_grid_pt_idxs.detach_()
            

            scaled_dist = lower_pt_rel_dists.unsqueeze(-1) + interp_points_flip.unsqueeze(-2)
            dim_interp_values = self.rbf_interpolation_kernel(scaled_dist)

            offset = (interp_points - interp_points.min()).unsqueeze(-2)
            dim_interp_indices = lower_grid_pt_idxs.unsqueeze(-1) + offset
            
            # Find points who's closest lower grid point is the first grid point
            # This corresponds to a boundary condition that we must fix manually.
            
            dim_interp_values[dim_interp_indices < 0.] = 0.
            dim_interp_indices[dim_interp_indices < 0.] = float('nan')
            
            dim_interp_values[dim_interp_indices >= num_grid_points] = 0.
            dim_interp_indices[dim_interp_indices >= num_grid_points] = float('nan')


            n_inner_repeat = num_coefficients ** i
            n_outer_repeat = num_coefficients ** (num_dim - i - 1)
            index_coeff = num_grid_points ** (num_dim - i - 1)
            dim_interp_indices = dim_interp_indices.unsqueeze(-1).repeat(1, n_inner_repeat, n_outer_repeat)
            dim_interp_values = dim_interp_values.unsqueeze(-1).repeat(1, n_inner_repeat, n_outer_repeat)
            interp_indices = interp_indices.add(dim_interp_indices.view(num_target_points, -1).mul(index_coeff))
            interp_values = interp_values.mul(dim_interp_values.view(num_target_points, -1))

            
        interp_indices[torch.isnan(interp_indices)] = 0.
        
        return interp_indices.long(), interp_values.div(interp_values.sum(-1).unsqueeze(-1)) #Renormalise interp_values to 1