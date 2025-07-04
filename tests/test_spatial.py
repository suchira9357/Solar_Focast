import pytest
from shadow_calculator import ShadowCalculator
import math

@pytest.fixture
def shadow_calc():
    """Create a fixture that simulates one frame with two clouds"""
    # Create shadow calculator
    calculator = ShadowCalculator(domain_size=50000, area_size_km=50.0)
    
    # Create test clouds
    cloud_ellipses = [
        # Cloud 1: center, radius, rotation, opacity
        (25000, 25000, 3000, 2000, 0.0, 0.8),
        # Cloud 2: center, radius, rotation, opacity  
        (35000, 15000, 2000, 2000, 0.5, 0.6)
    ]
    
    # Simple solar position
    solar_position = {'elevation': 90.0, 'azimuth': 0.0}
    
    # Run project clouds to populate cloud_cells
    calculator._project_clouds_to_ground(cloud_ellipses, solar_position)
    
    return calculator

def test_cloud_cells_are_ints(shadow_calc):
    """Test that all cloud cell indices are integers"""
    # Verify cloud_cells exists and contains something
    assert shadow_calc.cloud_cells is not None
    assert len(shadow_calc.cloud_cells) > 0
    
    # Verify all elements in cloud_cells are integers
    assert all(isinstance(i, int)
               for v in shadow_calc.cloud_cells.values()
               for i in v)
    
    # Try adding a non-integer (should never happen in real code)
    try:
        key = next(iter(shadow_calc.cloud_cells.keys()))
        shadow_calc.cloud_cells[key].append("P001")  # Add a string
        
        # This assertion should fail if we've added a string
        assert all(isinstance(i, int)
                   for v in shadow_calc.cloud_cells.values()
                   for i in v)
        pytest.fail("Test should have failed with mixed types in cloud_cells")
    except AssertionError:
        # This is expected - test passed!
        pass

def test_cloud_cells_cleared_each_frame(shadow_calc):
    """Test that cloud_cells is cleared and rebuilt each frame"""
    # Capture the current state of cloud_cells
    original_cells = dict(shadow_calc.cloud_cells)
    assert len(original_cells) > 0
    
    # Clear cloud_cells by simulating a new frame with different clouds
    new_clouds = [
        (10000, 10000, 2000, 2000, 0.0, 0.5)  # Different cloud position
    ]
    solar_position = {'elevation': 90.0, 'azimuth': 0.0}
    
    # Project new clouds
    shadow_calc._project_clouds_to_ground(new_clouds, solar_position)
    
    # Check that the cloud_cells has changed
    assert shadow_calc.cloud_cells != original_cells
    
    # Make sure the old cloud isn't still in the cells
    for cell_key, cloud_indices in shadow_calc.cloud_cells.items():
        assert 1 not in cloud_indices  # Cloud 1 from original clouds
    
    # Test clearing with empty cloud list
    shadow_calc._project_clouds_to_ground([], solar_position)
    assert len(shadow_calc.cloud_cells) == 0

def test_type_safety_in_coverage_calculation():
    """Test that coverage calculation handles mixed types safely"""
    # Create a new shadow calculator
    calculator = ShadowCalculator(domain_size=50000, area_size_km=50.0)
    
    # Create test clouds
    cloud_ellipses = [
        (25000, 25000, 3000, 2000, 0.0, 0.8)  # One cloud
    ]
    
    # Mock panel data
    class MockPanelDF:
        def iterrows(self):
            yield None, {"panel_id": "P001", "x_km": 25.0, "y_km": 25.0}
    
    # Mock panel cells
    panel_cells = {(12, 12): ["P001"]}  # Cell coordinates for panel at (25,25)
    
    # Force a mixed type situation by adding a string to cloud cells
    calculator.cloud_cells = defaultdict(list)
    calculator.cloud_cells[(12, 12)].append(0)  # Valid integer index
    calculator.cloud_cells[(12, 12)].append("P001")  # Invalid string
    
    # Create a mock _project_clouds_to_ground that doesn't clear our test data
    def mock_project(clouds, solar):
        return [{"x": 25.0, "y": 25.0, "width": 3.0, "height": 2.0,
                 "rotation": 0.0, "opacity": 0.8, "altitude": 1.0}]
    
    calculator._project_clouds_to_ground = mock_project
    
    # This should not raise an exception despite the mixed types
    coverage = calculator.calculate_panel_coverage(
        cloud_ellipses, 
        MockPanelDF(), 
        {"elevation": 90.0, "azimuth": 0.0},
        panel_cells
    )
    
    # We should still get coverage results
    assert "P001" in coverage
    assert coverage["P001"] > 0

from collections import defaultdict