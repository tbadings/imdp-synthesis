"""
Validation functions for compose_partitions.

This module provides functions to test and validate that a composed partition
reproduces the exact same properties as the original full partition.
"""

import numpy as np
import jax.numpy as jnp


def validate_composed_partition(original_partition, composed_partition, comp_partitions, 
                                independent_dimensions_x, tolerance=1e-5):
    """
    Validate that the composed partition matches the original partition.
    
    Tests include:
    - Total number of regions
    - Region centers and bounds
    - Partition grid structure
    - Cell widths and boundaries
    - Goal regions
    - Critical regions
    
    :param original_partition: The original full RectangularPartition object
    :param composed_partition: The ComposedPartition object from compose_partitions
    :param comp_partitions: List of component RectangularPartition objects
    :param independent_dimensions_x: List of state dimension indices for each component
    :param tolerance: Numerical tolerance for floating point comparisons
    :return: Dictionary with validation results and any errors
    """
    results = {
        'passed': True,
        'tests': {},
        'errors': []
    }
    
    # Test 1: Overall partition size
    test_name = 'total_regions'
    try:
        orig_num_regions = len(original_partition.regions['idxs'])
        comp_num_regions = composed_partition.size
        if orig_num_regions == comp_num_regions:
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'Total regions match: {orig_num_regions}',
                'original': orig_num_regions,
                'composed': comp_num_regions
            }
        else:
            raise AssertionError(f"Region count mismatch: {orig_num_regions} vs {comp_num_regions}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 2: Number of dimensions
    test_name = 'dimensions'
    try:
        orig_dims = original_partition.dimension
        comp_dims = composed_partition.dimension
        if orig_dims == comp_dims:
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'Dimensions match: {orig_dims}',
                'original': orig_dims,
                'composed': comp_dims
            }
        else:
            raise AssertionError(f"Dimension mismatch: {orig_dims} vs {comp_dims}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 3: Partition grid structure (number_per_dim)
    test_name = 'grid_structure'
    try:
        orig_npd = np.array(original_partition.number_per_dim)
        comp_npd = np.array(composed_partition.number_per_dim)
        if np.allclose(orig_npd, comp_npd):
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'Grid structure matches: {orig_npd.tolist()}',
                'original': orig_npd.tolist(),
                'composed': comp_npd.tolist()
            }
        else:
            raise AssertionError(f"Grid mismatch: {orig_npd} vs {comp_npd}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 4: Cell widths
    test_name = 'cell_widths'
    try:
        orig_cw = np.array(original_partition.cell_width)
        comp_cw = np.array(composed_partition.cell_width)
        if np.allclose(orig_cw, comp_cw, atol=tolerance):
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'Cell widths match',
                'original': orig_cw.tolist(),
                'composed': comp_cw.tolist()
            }
        else:
            raise AssertionError(f"Cell width mismatch")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 5: Partition boundaries
    test_name = 'boundaries'
    try:
        orig_lb = np.array(original_partition.boundary_lb)
        orig_ub = np.array(original_partition.boundary_ub)
        comp_lb = np.array(composed_partition.boundary_lb)
        comp_ub = np.array(composed_partition.boundary_ub)
        
        if np.allclose(orig_lb, comp_lb, atol=tolerance) and np.allclose(orig_ub, comp_ub, atol=tolerance):
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': 'Partition boundaries match',
                'original_lb': orig_lb.tolist(),
                'original_ub': orig_ub.tolist(),
                'composed_lb': comp_lb.tolist(),
                'composed_ub': comp_ub.tolist()
            }
        else:
            raise AssertionError("Boundary mismatch")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 6: Region centers consistency
    test_name = 'region_centers'
    try:
        orig_centers = original_partition.regions['centers']
        comp_centers = composed_partition.regions['centers']
        
        if np.allclose(orig_centers, comp_centers, atol=tolerance):
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'Region centers match (shape: {orig_centers.shape})',
                'shape': orig_centers.shape
            }
        else:
            max_diff = np.max(np.abs(orig_centers - comp_centers))
            raise AssertionError(f"Region centers differ, max diff: {max_diff}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 7: Region bounds
    test_name = 'region_bounds'
    try:
        orig_lb = original_partition.regions['lower_bounds']
        orig_ub = original_partition.regions['upper_bounds']
        comp_lb = composed_partition.regions['lower_bounds']
        comp_ub = composed_partition.regions['upper_bounds']
        
        if np.allclose(orig_lb, comp_lb, atol=tolerance) and np.allclose(orig_ub, comp_ub, atol=tolerance):
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'Region bounds match (shape: {orig_lb.shape})',
                'shape': orig_lb.shape
            }
        else:
            max_diff_lb = np.max(np.abs(orig_lb - comp_lb))
            max_diff_ub = np.max(np.abs(orig_ub - comp_ub))
            raise AssertionError(f"Region bounds differ, max diff lb: {max_diff_lb}, ub: {max_diff_ub}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 8: Goal regions
    test_name = 'goal_regions'
    try:
        orig_goal_idxs = original_partition.goal['idxs']
        comp_goal_idxs = composed_partition.goal['idxs']
        orig_goal_bools = original_partition.goal['bools']
        comp_goal_bools = composed_partition.goal['bools']
        
        # Convert to sets for comparison (handles lists, tuples, arrays)
        orig_goal_set = set(orig_goal_idxs) if orig_goal_idxs else set()
        comp_goal_set = set(comp_goal_idxs) if comp_goal_idxs else set()
        
        # Check count
        if len(orig_goal_set) != len(comp_goal_set):
            raise AssertionError(f"Goal region count mismatch: {len(orig_goal_set)} vs {len(comp_goal_set)}")
        
        # Check indices match
        if orig_goal_set != comp_goal_set:
            raise AssertionError(f"Goal region indices differ")
        
        # Check boolean arrays match
        if not np.array_equal(orig_goal_bools, comp_goal_bools):
            raise AssertionError(f"Goal region boolean arrays differ")
        
        results['tests'][test_name] = {
            'status': 'PASS',
            'message': f'Goal regions match: {len(orig_goal_set)} regions',
            'count': len(orig_goal_set)
        }
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 9: Critical regions
    test_name = 'critical_regions'
    try:
        orig_crit_idxs = original_partition.critical['idxs']
        comp_crit_idxs = composed_partition.critical['idxs']
        orig_crit_bools = original_partition.critical['bools']
        comp_crit_bools = composed_partition.critical['bools']
        
        # Convert to sets for comparison (handles lists, tuples, arrays)
        orig_crit_set = set(orig_crit_idxs) if orig_crit_idxs else set()
        comp_crit_set = set(comp_crit_idxs) if comp_crit_idxs else set()
        
        # Check count
        if len(orig_crit_set) != len(comp_crit_set):
            raise AssertionError(f"Critical region count mismatch: {len(orig_crit_set)} vs {len(comp_crit_set)}")
        
        # Check indices match
        if orig_crit_set != comp_crit_set:
            raise AssertionError(f"Critical region indices differ")
        
        # Check boolean arrays match
        if not np.array_equal(orig_crit_bools, comp_crit_bools):
            raise AssertionError(f"Critical region boolean arrays differ")
        
        results['tests'][test_name] = {
            'status': 'PASS',
            'message': f'Critical regions match: {len(orig_crit_set)} regions',
            'count': len(orig_crit_set)
        }
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 10: Verify rectangular property
    test_name = 'rectangular_property'
    try:
        if hasattr(composed_partition, 'rectangular') and composed_partition.rectangular:
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': 'Partition is rectangular'
            }
        else:
            raise AssertionError("Composed partition is not rectangular")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    return results


def print_partition_validation_report(results):
    """
    Print a formatted report of partition validation results.
    
    :param results: Results dictionary from validate_composed_partition
    """
    print("\n" + "="*70)
    print("PARTITION VALIDATION REPORT")
    print("="*70)
    
    if results['passed']:
        print("✓ ALL TESTS PASSED\n")
    else:
        print("✗ SOME TESTS FAILED\n")
    
    for test_name, test_result in results['tests'].items():
        status = test_result['status']
        status_symbol = "✓" if status == 'PASS' else "✗"
        print(f"{status_symbol} {test_name}: {test_result['message']}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("="*70 + "\n")
