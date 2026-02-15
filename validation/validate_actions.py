"""
Validation functions for compose_actions.

This module provides functions to test and validate that a composed actions object
matches the original actions object.
"""

import numpy as np
import jax.numpy as jnp


def validate_composed_actions(original_actions, composed_actions, tolerance=1e-5):
    """
    Validate that the composed actions object matches the original actions object.
    
    Compares the original actions (computed from the full partition) with the
    composed_actions (constructed from composing component actions). Tests include:
    - Inputs (control actions) are identical
    - FRS bounds (lower and upper) are identical
    - Index bounds are identical
    - max_slice is identical
    - Shapes are identical
    
    :param original_actions: The original RectangularForward object (full space)
    :param composed_actions: The ComposedActions object from compose_actions
    :param tolerance: Numerical tolerance for floating point comparisons
    :return: Dictionary with validation results and any errors
    """
    results = {
        'passed': True,
        'tests': {},
        'errors': [],
        'warnings': []
    }
    
    # Test 1: Inputs shape
    test_name = 'inputs_shape'
    try:
        orig_shape = original_actions.inputs.shape
        comp_shape = composed_actions.inputs.shape
        
        if orig_shape == comp_shape:
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'Input shapes match: {orig_shape}',
                'shape': orig_shape
            }
        else:
            raise AssertionError(f"Input shape mismatch: {orig_shape} vs {comp_shape}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 2: Inputs values
    test_name = 'inputs_values'
    try:
        if np.allclose(original_actions.inputs, composed_actions.inputs, atol=tolerance):
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'Input values are identical',
                'num_actions': len(original_actions.inputs)
            }
        else:
            max_diff = np.max(np.abs(original_actions.inputs - composed_actions.inputs))
            raise AssertionError(f"Input values differ, max difference: {max_diff}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 3: FRS bounds shapes
    test_name = 'frs_lb_shape'
    try:
        orig_shape = original_actions.frs_lb.shape
        comp_shape = composed_actions.frs_lb.shape
        
        if orig_shape == comp_shape:
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'FRS_lb shapes match: {orig_shape}',
                'shape': orig_shape
            }
        else:
            raise AssertionError(f"FRS_lb shape mismatch: {orig_shape} vs {comp_shape}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    test_name = 'frs_ub_shape'
    try:
        orig_shape = original_actions.frs_ub.shape
        comp_shape = composed_actions.frs_ub.shape
        
        if orig_shape == comp_shape:
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'FRS_ub shapes match: {orig_shape}',
                'shape': orig_shape
            }
        else:
            raise AssertionError(f"FRS_ub shape mismatch: {orig_shape} vs {comp_shape}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 4: FRS lower bounds values
    test_name = 'frs_lb_values'
    try:
        if np.allclose(original_actions.frs_lb, composed_actions.frs_lb, atol=tolerance):
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'FRS_lb values are identical',
                'num_states': original_actions.frs_lb.shape[0],
                'num_actions': original_actions.frs_lb.shape[1],
                'num_dims': original_actions.frs_lb.shape[2]
            }
        else:
            max_diff = np.max(np.abs(original_actions.frs_lb - composed_actions.frs_lb))
            mean_diff = np.mean(np.abs(original_actions.frs_lb - composed_actions.frs_lb))
            raise AssertionError(f"FRS_lb values differ, max: {max_diff}, mean: {mean_diff}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 5: FRS upper bounds values
    test_name = 'frs_ub_values'
    try:
        if np.allclose(original_actions.frs_ub, composed_actions.frs_ub, atol=tolerance):
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'FRS_ub values are identical',
                'num_states': original_actions.frs_ub.shape[0],
                'num_actions': original_actions.frs_ub.shape[1],
                'num_dims': original_actions.frs_ub.shape[2]
            }
        else:
            max_diff = np.max(np.abs(original_actions.frs_ub - composed_actions.frs_ub))
            mean_diff = np.mean(np.abs(original_actions.frs_ub - composed_actions.frs_ub))
            raise AssertionError(f"FRS_ub values differ, max: {max_diff}, mean: {mean_diff}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 6: FRS index lower bounds
    test_name = 'frs_idx_lb_values'
    try:
        if np.allclose(original_actions.frs_idx_lb, composed_actions.frs_idx_lb):
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'FRS_idx_lb values are identical'
            }
        else:
            num_diff = np.sum(original_actions.frs_idx_lb != composed_actions.frs_idx_lb)
            raise AssertionError(f"FRS_idx_lb values differ at {num_diff} locations")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 7: FRS index upper bounds
    test_name = 'frs_idx_ub_values'
    try:
        if np.allclose(original_actions.frs_idx_ub, composed_actions.frs_idx_ub):
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'FRS_idx_ub values are identical'
            }
        else:
            num_diff = np.sum(original_actions.frs_idx_ub != composed_actions.frs_idx_ub)
            raise AssertionError(f"FRS_idx_ub values differ at {num_diff} locations")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 8: max_slice
    test_name = 'max_slice'
    try:
        orig_max_slice = tuple(original_actions.max_slice) if hasattr(original_actions.max_slice, '__iter__') else original_actions.max_slice
        comp_max_slice = composed_actions.max_slice
        
        if orig_max_slice == comp_max_slice:
            results['tests'][test_name] = {
                'status': 'PASS',
                'message': f'max_slice values are identical: {comp_max_slice}',
                'value': comp_max_slice
            }
        else:
            raise AssertionError(f"max_slice mismatch: {orig_max_slice} vs {comp_max_slice}")
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
        results['passed'] = False
        results['errors'].append(f"{test_name}: {e}")
    
    # Test 9: idxs array (if it exists)
    test_name = 'idxs_array'
    try:
        if hasattr(original_actions, 'idxs') and hasattr(composed_actions, 'idxs'):
            if np.allclose(original_actions.idxs, composed_actions.idxs):
                results['tests'][test_name] = {
                    'status': 'PASS',
                    'message': f'idxs arrays are identical: {len(original_actions.idxs)} elements',
                    'length': len(original_actions.idxs)
                }
            else:
                raise AssertionError("idxs arrays differ")
        else:
            results['tests'][test_name] = {
                'status': 'SKIP',
                'message': 'idxs attribute not present in both objects'
            }
    except Exception as e:
        results['tests'][test_name] = {
            'status': 'FAIL',
            'message': str(e)
        }
    
    return results


def print_actions_validation_report(results):
    """
    Print a formatted report of actions validation results.
    
    :param results: Results dictionary from validate_composed_actions
    """
    print("\n" + "="*70)
    print("ACTIONS VALIDATION REPORT")
    print("="*70)
    
    if results['passed']:
        print("✓ ALL TESTS PASSED\n")
    else:
        print("✗ SOME TESTS FAILED\n")
    
    for test_name, test_result in results['tests'].items():
        status = test_result['status']
        if status == 'PASS':
            status_symbol = "✓"
        elif status == 'SKIP':
            status_symbol = "⊘"
        else:
            status_symbol = "✗"
        print(f"{status_symbol} {test_name}: {test_result['message']}")
    
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  ✗ {error}")
    
    print("="*70 + "\n")

