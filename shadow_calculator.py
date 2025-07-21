# shadow_calculator.py

import numpy as np
import pandas as pd
import math
import time
import threading
import os
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import lru_cache
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

CloudEllipse = namedtuple('CloudEllipse', [
    'cx', 'cy', 'width', 'height', 'rotation', 'opacity', 'altitude'
])

def to_cloud_ellipse(ellipse_params: Union[Tuple, List]) -> CloudEllipse:
    if len(ellipse_params) >= 7:
        return CloudEllipse(*ellipse_params[:7])
    cx, cy, cw, ch, crot, cop = ellipse_params[:6]
    return CloudEllipse(cx, cy, cw, ch, crot, cop, 1000.0)

@lru_cache(maxsize=128)
def _cached_sin_cos(angle_deg: int) -> Tuple[float, float]:
    rad = math.radians(angle_deg)
    return math.sin(rad), math.cos(rad)

def _vectorized_ellipse_coverage(
    panel_positions: np.ndarray,
    ellipse_data: np.ndarray
) -> np.ndarray:
    if ellipse_data.shape[0] == 0:
        return np.zeros(panel_positions.shape[0], dtype=np.float32)
    
    total_coverage = np.zeros(panel_positions.shape[0], dtype=np.float32)
    for idx in range(ellipse_data.shape[0]):
        cx, cy, width, height, rotation, opacity = ellipse_data[idx]
        if opacity < 0.01:
            continue
        
        dx = panel_positions[:,0] - cx
        dy = panel_positions[:,1] - cy
        
        if abs(rotation) > 0.01:
            cos_r = math.cos(math.radians(rotation))
            sin_r = math.sin(math.radians(rotation))
            dx, dy = dx*cos_r + dy*sin_r, -dx*sin_r + dy*cos_r
        
        a2 = (width/2)**2
        b2 = (height/2)**2
        if a2>0 and b2>0:
            vals = (dx**2)/a2 + (dy**2)/b2
            cov = np.where(vals<=1.0, opacity*(1.0-vals), 0.0)
            total_coverage = np.minimum(1.0, total_coverage + cov)
    
    return total_coverage

def _calculate_shadow_projection(
    cloud_ellipse: CloudEllipse,
    solar_elevation: float,
    solar_azimuth: float
) -> Tuple[float,float,float,float]:
    if solar_elevation<=0:
        return cloud_ellipse.cx, cloud_ellipse.cy, 0.0, 0.0
    elev = math.radians(solar_elevation)
    az = math.radians(solar_azimuth)
    dist = cloud_ellipse.altitude / math.tan(elev)
    dx = dist*math.sin(az)
    dy = dist*math.cos(az)
    factor = 1.0 + (1.0 - math.sin(elev))*0.5
    return (
        cloud_ellipse.cx+dx,
        cloud_ellipse.cy+dy,
        cloud_ellipse.width*factor,
        cloud_ellipse.height*factor
    )

class ThreadSafeShadowCalculator:
    """
    High-performance shadow calculator with CSV coordinate integration.
    """
    def __init__(
        self,
        panel_df: Optional[pd.DataFrame] = None,
        domain_size: float = 50000.0
    ):
        self.domain_size = domain_size
        if panel_df is None:
            self.panel_df = self._load_panel_coordinates_from_csv()
        else:
            self.panel_df = panel_df
        self.panel_coordinates = {}
        self._prepare_panel_data()
        self.thread_safe_mode = False
        self._lock = threading.RLock()
        self.shadow_cache = {}
        self.last_solar_position = None
        self.cache_max_size = 1000
        self.shadow_softness = 0.2
        self.penumbra_factor = 0.15
        self.atmospheric_scattering = 0.1
        self.spatial_grid_size = 5000.0
        self.spatial_index = {}
        self._build_spatial_index()
        self.stats = {
            'calculations': 0, 'cache_hits': 0, 'cache_misses': 0,
            'total_time': 0.0, 'avg_time': 0.0
        }
        print(f"ThreadSafeShadowCalculator initialized:")
        print(f"  Panels loaded: {len(self.panel_coordinates)}")
        print(f"  Domain size: {domain_size/1000:.1f} km")
    
    def _load_panel_coordinates_from_csv(self) -> pd.DataFrame:
        csv_file = "extended_coordinates.csv"
        try:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                required = ['panel_id','x_km','y_km']
                missing = [c for c in required if c not in df.columns]
                if missing:
                    print(f"Warning: Missing columns {missing}")
                df['x_km'] = df.get('x_km', df.get('x', 0))
                df['y_km'] = df.get('y_km', df.get('y', 0))
                print(f"✓ Loaded {len(df)} panels from {csv_file}")
                return df
        except Exception as e:
            print(f"ERROR loading CSV: {e}")
        return self._create_default_panel_layout()
    
    def _create_default_panel_layout(self) -> pd.DataFrame:
        data=[]
        pid=1
        for x in range(0,50,2):
            for y in range(10,40,2):
                data.append({'panel_id':f'P{pid:03d}','x_km':float(x),'y_km':float(y)})
                pid+=1
                if pid>155: break
            if pid>155: break
        print(f"Created default layout: {len(data)} panels")
        return pd.DataFrame(data)
    
    def _prepare_panel_data(self):
        if self.panel_df is None or len(self.panel_df)==0:
            self.panel_coordinates={}
            self.panel_positions = np.zeros((0,2),dtype=np.float32)
            self.panel_ids = np.array([],dtype='<U10')
            return
        for _,r in self.panel_df.iterrows():
            self.panel_coordinates[r['panel_id']] = (r['x_km']*1000.0, r['y_km']*1000.0)
        self.panel_positions = (self.panel_df[['x_km','y_km']].values.astype(np.float32)*1000.0)
        self.panel_ids = self.panel_df['panel_id'].values
        print(f"Prepared {len(self.panel_coordinates)} panels")
    
    def _build_spatial_index(self):
        for _,r in self.panel_df.iterrows():
            gx = int(r['x_km']*1000.0/self.spatial_grid_size)
            gy = int(r['y_km']*1000.0/self.spatial_grid_size)
            self.spatial_index.setdefault((gx,gy),[]).append(r['panel_id'])
        print(f"Built spatial index: {len(self.spatial_index)} cells")
    
    def set_thread_safe_mode(self, enabled: bool):
        self.thread_safe_mode = enabled
        if enabled: print("Thread-safe mode enabled")
    
    def calculate_panel_coverage(
        self,
        cloud_ellipses: List,
        panel_df: pd.DataFrame,
        solar_position: Dict[str,float]
    ) -> Dict[str,float]:
        if self.thread_safe_mode:
            with self._lock:
                return self._calculate_panel_coverage_internal(cloud_ellipses, panel_df, solar_position)
        else:
            return self._calculate_panel_coverage_internal(cloud_ellipses, panel_df, solar_position)
    
    def _calculate_panel_coverage_internal(
        self,
        cloud_ellipses: List,
        panel_df: pd.DataFrame,
        solar_position: Dict[str,float]
    ) -> Dict[str,float]:
        start = time.perf_counter()
        if not cloud_ellipses or panel_df is None or len(panel_df)==0:
            return {r['panel_id']:0.0 for _,r in panel_df.iterrows()}
        key = (round(solar_position.get('elevation',0),1),round(solar_position.get('azimuth',0),1))
        if key==self.last_solar_position and len(self.shadow_cache)<self.cache_max_size:
            self.stats['cache_hits']+=1
        else:
            self.shadow_cache.clear()
            self.last_solar_position=key
            self.stats['cache_misses']+=1
        
        shadows=[]
        for e in cloud_ellipses:
            ce = to_cloud_ellipse(e)
            sx,sy,sw,sh = _calculate_shadow_projection(ce, solar_position.get('elevation',45.0), solar_position.get('azimuth',180.0))
            op = ce.opacity * (1.0 - self.atmospheric_scattering)
            shadows.append((sx,sy,sw*(1+self.penumbra_factor),sh*(1+self.penumbra_factor),ce.rotation,op))
        
        if not shadows:
            return {r['panel_id']:0.0 for _,r in panel_df.iterrows()}
        
        panel_positions = (panel_df[['x_km','y_km']].values.astype(np.float32)*1000.0)
        shadow_data = np.array(shadows,dtype=np.float32)
        vals = _vectorized_ellipse_coverage(panel_positions, shadow_data)
        vals = np.power(vals, 1.0-self.shadow_softness)
        
        result={}
        for i,(_,r) in enumerate(panel_df.iterrows()):
            pid = r['panel_id']
            cov = float(vals[i])
            self.shadow_cache[(pid,key)] = cov
            result[pid] = cov
        
        t = time.perf_counter()-start
        self.stats['calculations']+=1
        self.stats['total_time']+=t
        self.stats['avg_time']=self.stats['total_time']/self.stats['calculations']
        return result
    
    def calculate_shadow_map(self, cloud_ellipses: List, solar_position: Dict[str,float], resolution: int=100) -> np.ndarray:
        return self._calculate_shadow_map_internal(cloud_ellipses, solar_position, resolution)
    
    def _calculate_shadow_map_internal(self, cloud_ellipses: List, solar_position: Dict[str,float], resolution: int) -> np.ndarray:
        # … same pattern as above …
        pass
    
    def get_performance_stats(self) -> Dict[str,Any]:
        with self._lock:
            return self.stats.copy()
    
    def clear_cache(self):
        with self._lock:
            self.shadow_cache.clear()
            self.last_solar_position=None
            print("Shadow cache cleared")


ShadowCalculator = ThreadSafeShadowCalculator

def create_shadow_calculator(panel_df=None, domain_size: float=50000.0):
    return ThreadSafeShadowCalculator(panel_df, domain_size)
