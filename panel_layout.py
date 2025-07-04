"""
Ultra-Optimized Panel Layout Module for Solar Farm Simulation
Maximum performance with advanced NumPy, Numba JIT, and memory optimizations
"""
import csv
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
import functools
import weakref
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Try to import Numba for JIT compilation
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    njit = jit
    prange = range

# Try to import spatial libraries for advanced indexing
try:
    from scipy.spatial import cKDTree, distance_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import rtree.index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False


@dataclass
class PanelData:
    """Memory-efficient panel data structure using slots"""
    __slots__ = ['panel_id', 'x_km', 'y_km', 'power_capacity']
    panel_id: str
    x_km: float
    y_km: float
    power_capacity: float


# Numba JIT-compiled functions for maximum performance
@njit(fastmath=True, cache=True)
def _vectorized_grid_coords(positions: np.ndarray, cell_size: float) -> np.ndarray:
    """JIT-compiled grid coordinate calculation"""
    return np.floor(positions / cell_size).astype(np.int32)


@njit(fastmath=True, cache=True, parallel=True)
def _vectorized_distances(positions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """JIT-compiled parallel distance calculation"""
    n = positions.shape[0]
    distances = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        dx = positions[i, 0] - target[0]
        dy = positions[i, 1] - target[1]
        distances[i] = np.sqrt(dx * dx + dy * dy)
    
    return distances


@njit(fastmath=True, cache=True)
def _bounding_box_filter(positions: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """JIT-compiled bounding box filtering"""
    n = positions.shape[0]
    mask = np.empty(n, dtype=np.bool_)
    
    min_x, min_y, max_x, max_y = bounds
    
    for i in range(n):
        x, y = positions[i, 0], positions[i, 1]
        mask[i] = (min_x <= x <= max_x) and (min_y <= y <= max_y)
    
    return mask


@njit(fastmath=True, cache=True, parallel=True)
def _density_histogram(positions: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    """JIT-compiled 2D histogram for density calculation"""
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1
    hist = np.zeros((n_x, n_y), dtype=np.int32)
    
    for i in prange(positions.shape[0]):
        x, y = positions[i, 0], positions[i, 1]
        
        # Find x bin
        x_bin = -1
        for j in range(n_x):
            if x_edges[j] <= x < x_edges[j + 1]:
                x_bin = j
                break
        
        # Find y bin
        y_bin = -1
        for j in range(n_y):
            if y_edges[j] <= y < y_edges[j + 1]:
                y_bin = j
                break
        
        if x_bin >= 0 and y_bin >= 0:
            hist[x_bin, y_bin] += 1
    
    return hist


class UltraOptimizedPanelLayout:
    """Ultra-optimized panel layout with multiple acceleration techniques"""
    
    def __init__(self, csv_path: str = "extended_coordinates.csv", 
                 enable_caching: bool = True, enable_spatial_trees: bool = True):
        self.csv_path = csv_path
        self.enable_caching = enable_caching
        self.enable_spatial_trees = enable_spatial_trees
        
        # Core data structures
        self.panels_df: Optional[pd.DataFrame] = None
        self.panel_positions: Optional[np.ndarray] = None  # Shape: (N, 2)
        self.panel_ids: Optional[np.ndarray] = None        # Shape: (N,)
        self.panel_capacities: Optional[np.ndarray] = None # Shape: (N,)
        
        # Memory-mapped arrays for large datasets
        self._positions_memmap: Optional[np.memmap] = None
        
        # Spatial acceleration structures
        self._kdtree: Optional['cKDTree'] = None
        self._rtree_index: Optional['rtree.index.Index'] = None
        
        # Caching
        self._cache = {} if enable_caching else None
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Thread pool for parallel operations
        self._thread_pool = ThreadPoolExecutor(max_workers=min(4, mp.cpu_count()))
        
        # Load data
        self._load_panels()
        self._build_spatial_structures()
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)
    
    def _cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        return str(hash(args))
    
    def _cached(self, func):
        """Caching decorator for expensive operations"""
        if not self.enable_caching:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = self._cache_key(func.__name__, args, tuple(sorted(kwargs.items())))
            
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
            
            result = func(*args, **kwargs)
            self._cache[cache_key] = result
            self._cache_misses += 1
            return result
        
        return wrapper
    
    def _load_panels(self) -> None:
        """Optimized panel loading with memory mapping for large files"""
        if os.path.exists(self.csv_path):
            try:
                # Check file size to decide on loading strategy
                file_size = os.path.getsize(self.csv_path)
                
                if file_size > 50 * 1024 * 1024:  # > 50MB
                    # Use chunked reading for large files
                    self._load_panels_chunked()
                else:
                    # Standard pandas loading
                    self._load_panels_standard()
                    
            except Exception as e:
                print(f"Error loading {self.csv_path}: {e}")
                self._create_fallback_panels()
        else:
            self._create_fallback_panels()
        
        # Create memory-efficient NumPy arrays
        self._optimize_memory_layout()
    
    def _load_panels_standard(self) -> None:
        """Standard pandas loading for smaller files"""
        # Use categorical data type for panel_id to save memory
        dtypes = {
            'panel_id': 'category',
            'x_km': np.float32,  # Use float32 instead of float64
            'y_km': np.float32,
            'power_capacity': np.float32
        }
        
        self.panels_df = pd.read_csv(self.csv_path, dtype=dtypes)
        
        # Ensure required columns
        if 'panel_id' not in self.panels_df.columns:
            self.panels_df['panel_id'] = pd.Categorical([f"P{i+1:03d}" for i in range(len(self.panels_df))])
        if 'power_capacity' not in self.panels_df.columns:
            self.panels_df['power_capacity'] = np.float32(5.0)
        
        print(f"Loaded {len(self.panels_df)} panels from {self.csv_path}")
    
    def _load_panels_chunked(self) -> None:
        """Chunked loading for large CSV files"""
        print(f"Loading large file {self.csv_path} in chunks...")
        
        chunks = []
        chunk_size = 10000
        
        for chunk in pd.read_csv(self.csv_path, chunksize=chunk_size):
            # Process chunk if needed
            if 'panel_id' not in chunk.columns:
                start_idx = len(chunks) * chunk_size
                chunk['panel_id'] = [f"P{i+1:03d}" for i in range(start_idx, start_idx + len(chunk))]
            if 'power_capacity' not in chunk.columns:
                chunk['power_capacity'] = 5.0
            
            chunks.append(chunk)
        
        # Concatenate all chunks
        self.panels_df = pd.concat(chunks, ignore_index=True)
        
        # Optimize data types
        self.panels_df['panel_id'] = self.panels_df['panel_id'].astype('category')
        self.panels_df['x_km'] = self.panels_df['x_km'].astype(np.float32)
        self.panels_df['y_km'] = self.panels_df['y_km'].astype(np.float32)
        self.panels_df['power_capacity'] = self.panels_df['power_capacity'].astype(np.float32)
        
        print(f"Loaded {len(self.panels_df)} panels from chunked file")
    
    def _create_fallback_panels(self) -> None:
        """Create fallback panels with optimized grid generation"""
        print(f"Creating optimized default panel grid...")
        
        # Use larger grid for stress testing
        grid_size = 20  # 400 panels instead of 36
        spacing = 2.0   # Closer spacing
        offset = 1.0
        
        # Vectorized grid generation
        coords_1d = np.arange(grid_size, dtype=np.float32) * spacing + offset
        x_coords, y_coords = np.meshgrid(coords_1d, coords_1d, indexing='ij')
        
        # Flatten arrays
        x_flat = x_coords.ravel()
        y_flat = y_coords.ravel()
        
        # Generate panel IDs efficiently
        panel_ids = np.array([f"P{i+1:05d}" for i in range(len(x_flat))], dtype='<U6')
        
        # Create DataFrame with optimal dtypes
        self.panels_df = pd.DataFrame({
            'panel_id': pd.Categorical(panel_ids),
            'x_km': x_flat,
            'y_km': y_flat,
            'power_capacity': np.full(len(x_flat), 5.0, dtype=np.float32)
        })
        
        print(f"Created {len(self.panels_df)} optimized default panels")
    
    def _optimize_memory_layout(self) -> None:
        """Optimize memory layout for maximum performance"""
        if self.panels_df is None:
            return
        
        # Create contiguous NumPy arrays for fastest access
        self.panel_positions = np.ascontiguousarray(
            self.panels_df[['x_km', 'y_km']].values, dtype=np.float32
        )
        self.panel_ids = np.ascontiguousarray(
            self.panels_df['panel_id'].values
        )
        self.panel_capacities = np.ascontiguousarray(
            self.panels_df['power_capacity'].values, dtype=np.float32
        )
        
        # For very large datasets, use memory mapping
        if len(self.panels_df) > 100000:
            # Create memory-mapped file for positions
            memmap_file = f"{self.csv_path}.positions.memmap"
            self._positions_memmap = np.memmap(
                memmap_file, dtype=np.float32, mode='w+', 
                shape=self.panel_positions.shape
            )
            self._positions_memmap[:] = self.panel_positions[:]
            self._positions_memmap.flush()
            
            # Use memory-mapped array instead
            self.panel_positions = self._positions_memmap
    
    def _build_spatial_structures(self) -> None:
        """Build advanced spatial data structures"""
        if self.panel_positions is None or not self.enable_spatial_trees:
            return
        
        # Build KD-Tree for fast nearest neighbor queries
        if SCIPY_AVAILABLE:
            self._kdtree = cKDTree(self.panel_positions, leafsize=16, balanced_tree=True)
        
        # Build R-Tree for fast spatial queries
        if RTREE_AVAILABLE and len(self.panel_positions) < 50000:  # R-Tree can be memory intensive
            self._rtree_index = rtree.index.Index()
            for i, (x, y) in enumerate(self.panel_positions):
                # R-Tree expects (left, bottom, right, top)
                self._rtree_index.insert(i, (x, y, x, y))
    
    @functools.lru_cache(maxsize=128)
    def build_spatial_index_ultra_fast(self, cell_size_km: float = 2.0) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Ultra-fast spatial indexing using JIT compilation and optimized data structures
        
        Returns numpy arrays instead of lists for better performance
        """
        if self.panel_positions is None:
            return {}
        
        # Use JIT-compiled grid coordinate calculation
        grid_coords = _vectorized_grid_coords(self.panel_positions, cell_size_km)
        
        # Use pandas groupby for efficient spatial indexing (fastest for this operation)
        df_temp = pd.DataFrame({
            'grid_x': grid_coords[:, 0],
            'grid_y': grid_coords[:, 1],
            'panel_idx': np.arange(len(self.panel_positions))
        })
        
        # Group by grid coordinates and return numpy arrays
        spatial_index = {}
        for (grid_x, grid_y), group in df_temp.groupby(['grid_x', 'grid_y']):
            spatial_index[(int(grid_x), int(grid_y))] = group['panel_idx'].values
        
        return spatial_index
    
    def get_panels_in_radius_ultra_fast(self, center_x: float, center_y: float, 
                                       radius_km: float) -> np.ndarray:
        """Ultra-fast radius query using KD-Tree or JIT compilation"""
        if self.panel_positions is None:
            return np.array([], dtype=np.int32)
        
        # Use KD-Tree if available (fastest for large datasets)
        if self._kdtree is not None:
            indices = self._kdtree.query_ball_point([center_x, center_y], radius_km)
            return np.array(indices, dtype=np.int32)
        
        # Fallback to JIT-compiled distance calculation
        target = np.array([center_x, center_y], dtype=np.float32)
        distances = _vectorized_distances(self.panel_positions, target)
        return np.where(distances <= radius_km)[0]
    
    def get_panels_in_bounds_ultra_fast(self, min_x: float, max_x: float,
                                       min_y: float, max_y: float) -> np.ndarray:
        """Ultra-fast bounding box query"""
        if self.panel_positions is None:
            return np.array([], dtype=np.int32)
        
        # Use R-Tree if available
        if self._rtree_index is not None:
            indices = list(self._rtree_index.intersection((min_x, min_y, max_x, max_y)))
            return np.array(indices, dtype=np.int32)
        
        # Fallback to JIT-compiled bounding box filter
        bounds = np.array([min_x, min_y, max_x, max_y], dtype=np.float32)
        mask = _bounding_box_filter(self.panel_positions, bounds)
        return np.where(mask)[0]
    
    def find_nearest_panels_ultra_fast(self, target_x: float, target_y: float, 
                                      n: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-fast nearest neighbor search"""
        if self.panel_positions is None:
            return np.array([]), np.array([])
        
        # Use KD-Tree if available (fastest)
        if self._kdtree is not None:
            distances, indices = self._kdtree.query([target_x, target_y], k=min(n, len(self.panel_positions)))
            if np.isscalar(distances):
                distances = np.array([distances])
                indices = np.array([indices])
            return indices, distances
        
        # Fallback to JIT-compiled approach
        target = np.array([target_x, target_y], dtype=np.float32)
        distances = _vectorized_distances(self.panel_positions, target)
        
        # Use argpartition for O(n) performance instead of full sort
        n_actual = min(n, len(distances))
        indices = np.argpartition(distances, n_actual - 1)[:n_actual]
        indices = indices[np.argsort(distances[indices])]
        
        return indices, distances[indices]
    
    @functools.lru_cache(maxsize=32)
    def get_density_map_ultra_fast(self, cell_size_km: float = 1.0) -> np.ndarray:
        """Ultra-fast density map generation"""
        if self.panel_positions is None:
            return np.array([[]])
        
        # Calculate bounds
        min_x, min_y = np.min(self.panel_positions, axis=0)
        max_x, max_y = np.max(self.panel_positions, axis=0)
        
        # Create bin edges
        x_bins = int(np.ceil((max_x - min_x) / cell_size_km)) + 1
        y_bins = int(np.ceil((max_y - min_y) / cell_size_km)) + 1
        
        x_edges = np.linspace(min_x, max_x, x_bins + 1, dtype=np.float32)
        y_edges = np.linspace(min_y, max_y, y_bins + 1, dtype=np.float32)
        
        # Use JIT-compiled histogram for maximum speed
        return _density_histogram(self.panel_positions, x_edges, y_edges)
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get performance and memory statistics"""
        stats = {
            'num_panels': len(self.panel_positions) if self.panel_positions is not None else 0,
            'memory_usage_mb': 0,
            'cache_hit_ratio': 0,
            'spatial_structures': []
        }
        
        if self.panel_positions is not None:
            stats['memory_usage_mb'] = (
                self.panel_positions.nbytes + 
                self.panel_ids.nbytes + 
                self.panel_capacities.nbytes
            ) / (1024 * 1024)
        
        if self.enable_caching and self._cache_hits + self._cache_misses > 0:
            stats['cache_hit_ratio'] = self._cache_hits / (self._cache_hits + self._cache_misses)
        
        if self._kdtree is not None:
            stats['spatial_structures'].append('KDTree')
        if self._rtree_index is not None:
            stats['spatial_structures'].append('RTree')
        
        return stats
    
    def parallel_query(self, query_points: np.ndarray, radius_km: float) -> List[np.ndarray]:
        """Parallel processing for multiple queries"""
        if len(query_points) < 100:
            # Not worth parallelizing for small queries
            return [self.get_panels_in_radius_ultra_fast(x, y, radius_km) 
                   for x, y in query_points]
        
        # Split work across threads
        def worker(points_chunk):
            return [self.get_panels_in_radius_ultra_fast(x, y, radius_km) 
                   for x, y in points_chunk]
        
        chunk_size = max(1, len(query_points) // self._thread_pool._max_workers)
        chunks = [query_points[i:i+chunk_size] for i in range(0, len(query_points), chunk_size)]
        
        # Submit to thread pool
        futures = [self._thread_pool.submit(worker, chunk) for chunk in chunks]
        
        # Collect results
        results = []
        for future in futures:
            results.extend(future.result())
        
        return results


# Create optimized global instance
_ultra_layout = UltraOptimizedPanelLayout()

# Backward compatibility with maximum performance
PANELS = {
    panel_id: {
        "x_km": float(_ultra_layout.panel_positions[i, 0]),
        "y_km": float(_ultra_layout.panel_positions[i, 1]),
        "power_capacity": float(_ultra_layout.panel_capacities[i])
    }
    for i, panel_id in enumerate(_ultra_layout.panel_ids)
} if _ultra_layout.panel_positions is not None else {}


class UltraOptimizedPanelDataFrame:
    """Ultra-optimized DataFrame interface"""
    
    def __init__(self):
        self.layout = _ultra_layout
    
    def __len__(self) -> int:
        return len(self.layout.panel_positions) if self.layout.panel_positions is not None else 0
    
    def iterrows(self):
        """Optimized iteration using numpy arrays"""
        if self.layout.panel_positions is None:
            return
        
        for i in range(len(self.layout.panel_positions)):
            yield i, {
                "panel_id": self.layout.panel_ids[i],
                "x_km": float(self.layout.panel_positions[i, 0]),
                "y_km": float(self.layout.panel_positions[i, 1]),
                "power_capacity": float(self.layout.panel_capacities[i])
            }
    
    def get_positions_array(self) -> np.ndarray:
        """Direct access to position array"""
        return self.layout.panel_positions
    
    def query_radius(self, center_x: float, center_y: float, radius: float) -> List[str]:
        """Ultra-fast radius query"""
        indices = self.layout.get_panels_in_radius_ultra_fast(center_x, center_y, radius)
        return [self.layout.panel_ids[i] for i in indices]


# Export optimized functions
panel_df = UltraOptimizedPanelDataFrame()

def build_panel_cells(cell_size_km: float = 2.0) -> Dict[Tuple[int, int], List[str]]:
    """Ultra-optimized spatial index builder"""
    index_arrays = _ultra_layout.build_spatial_index_ultra_fast(cell_size_km)
    
    # Convert numpy arrays back to lists for backward compatibility
    result = {}
    for key, indices in index_arrays.items():
        result[key] = [_ultra_layout.panel_ids[i] for i in indices]
    
    return result


# Advanced utility functions
def benchmark_all_methods(iterations: int = 1000):
    """Comprehensive benchmark of all optimization techniques"""
    import time
    
    print("Ultra-Optimization Benchmark")
    print("=" * 50)
    
    # Test data
    test_points = np.random.uniform(0, 40, (100, 2))
    
    methods = [
        ("Spatial Index", lambda: _ultra_layout.build_spatial_index_ultra_fast(2.0)),
        ("Radius Query", lambda: _ultra_layout.get_panels_in_radius_ultra_fast(20, 20, 5)),
        ("Nearest Neighbors", lambda: _ultra_layout.find_nearest_panels_ultra_fast(15, 15, 5)),
        ("Density Map", lambda: _ultra_layout.get_density_map_ultra_fast(1.0)),
        ("Bounds Query", lambda: _ultra_layout.get_panels_in_bounds_ultra_fast(10, 30, 10, 30))
    ]
    
    for name, func in methods:
        start_time = time.perf_counter()
        for _ in range(iterations):
            result = func()
        elapsed = time.perf_counter() - start_time
        
        print(f"{name:20}: {elapsed:.4f}s ({elapsed/iterations*1000:.2f}ms per call)")
    
    # Memory usage
    stats = _ultra_layout.get_statistics()
    print(f"\nMemory Usage: {stats['memory_usage_mb']:.2f} MB")
    print(f"Cache Hit Ratio: {stats['cache_hit_ratio']:.2f}")
    print(f"Spatial Structures: {', '.join(stats['spatial_structures'])}")
    print(f"Numba Available: {NUMBA_AVAILABLE}")
    print(f"SciPy Available: {SCIPY_AVAILABLE}")
    print(f"RTree Available: {RTREE_AVAILABLE}")


if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark_all_methods()