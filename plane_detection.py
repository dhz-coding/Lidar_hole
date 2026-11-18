"""Axis-independent plane extraction and parallel face pairing."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence
import numpy as np

AXIS_VECTORS = {
    'x': np.array([1.0, 0.0, 0.0]),
    'y': np.array([0.0, 1.0, 0.0]),
    'z': np.array([0.0, 0.0, 1.0]),
}


def normalize(v: np.ndarray) -> np.ndarray:
    v=np.asarray(v,dtype=np.float64); n=float(np.linalg.norm(v))
    if n<=1e-12: raise ValueError('零向量不能归一化')
    return v/n


def canonical_plane(model: Sequence[float]) -> np.ndarray:
    plane=np.asarray(model,dtype=np.float64)
    if plane.shape!=(4,): raise ValueError('平面模型应为4个参数')
    plane=plane/np.linalg.norm(plane[:3])
    pivot=int(np.argmax(np.abs(plane[:3])))
    if plane[pivot]<0: plane=-plane
    return plane


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    n=normalize(normal)
    reference=np.eye(3)[int(np.argmin(np.abs(np.eye(3) @ n)))]
    u=normalize(np.cross(n,reference))
    v=normalize(np.cross(n,u))
    return u,v


@dataclass
class PlaneCandidate:
    plane_id:int
    axis_name:str
    model:np.ndarray
    inlier_points:np.ndarray
    count:int
    centroid:np.ndarray
    projected_extent:np.ndarray
    residual_rms:float
    source_indices:np.ndarray
    @property
    def normal(self)->np.ndarray: return self.model[:3]
    @property
    def position(self)->float: return float(-self.model[3])
    @property
    def projected_area(self)->float: return float(np.prod(self.projected_extent))


@dataclass
class PlaneGroup:
    group_id:int
    axis_name:str='free_normal'
    members:list[PlaneCandidate]=field(default_factory=list)
    @property
    def normal(self)->np.ndarray:
        base=self.members[0].normal
        weighted=np.zeros(3,dtype=np.float64)
        for m in self.members:
            n=m.normal if np.dot(m.normal,base)>=0 else -m.normal
            weighted += n*float(m.count)
        return normalize(weighted)
    @property
    def axis(self)->np.ndarray: return self.normal
    @property
    def position(self)->float:
        return float(np.average([m.position for m in self.members],weights=[m.count for m in self.members]))
    @property
    def count(self)->int: return int(sum(m.count for m in self.members))
    @property
    def points(self)->np.ndarray:
        return self.members[0].inlier_points if len(self.members)==1 else np.vstack([m.inlier_points for m in self.members])
    @property
    def centroid(self)->np.ndarray:
        return np.average(np.vstack([m.centroid for m in self.members]),axis=0,weights=[m.count for m in self.members])
    @property
    def projected_bbox(self)->np.ndarray:
        u,v=plane_basis(self.normal); vals=np.column_stack((self.points@u,self.points@v)); return np.array([vals.min(axis=0),vals.max(axis=0)],dtype=np.float64)
    @property
    def projected_extent(self)->np.ndarray: return self.projected_bbox[1]-self.projected_bbox[0]
    @property
    def projected_area(self)->float: return float(np.prod(self.projected_extent))


@dataclass
class PartitionCandidate:
    axis_name:str
    group_ids:tuple[int,...]
    face_groups:tuple[PlaneGroup,...]
    score:float
    position:float
    thickness:float|None
    reason:str
    @property
    def normal(self)->np.ndarray:
        base=self.face_groups[0].normal
        weighted=np.zeros(3,dtype=np.float64)
        for g in self.face_groups:
            n=g.normal if np.dot(g.normal,base)>=0 else -g.normal
            weighted += n*float(g.count)
        return normalize(weighted)
    @property
    def axis(self)->np.ndarray: return self.normal
    @property
    def face_models(self)->list[np.ndarray]:
        return [member.model for group in self.face_groups for member in group.members]
    @property
    def boundary_points(self)->np.ndarray: return np.vstack([group.points for group in self.face_groups])


def _downsample_rows(a:np.ndarray,max_rows:int,seed:int)->tuple[np.ndarray,np.ndarray]:
    if len(a)<=max_rows: return a,np.arange(len(a),dtype=np.int64)
    rng=np.random.default_rng(seed); idx=np.sort(rng.choice(len(a),size=max_rows,replace=False)); return a[idx],idx


def _build_cloud(points:np.ndarray):
    try: import open3d as o3d
    except ImportError as exc: raise RuntimeError('平面检测需要 Open3D') from exc
    cloud=o3d.geometry.PointCloud(); cloud.points=o3d.utility.Vector3dVector(np.asarray(points,dtype=np.float64)); return cloud


def extract_planes(xyz:np.ndarray,normals:np.ndarray|None,config:Mapping[str,object])->list[PlaneCandidate]:
    max_input=int(config.get('max_input_points',config.get('sample_max_points',400000)))
    ransac_distance=float(config.get('ransac_distance_threshold',0.012))
    ransac_n=int(config.get('ransac_n',3)); iterations=int(config.get('num_iterations',3000))
    max_planes=int(config.get('max_planes',28)); min_points=int(config.get('min_plane_points',500)); max_face=int(config.get('max_face_points',30000)); seed=int(config.get('seed',7))
    try:
        import open3d as o3d; o3d.utility.random.seed(seed)
    except ImportError as exc: raise RuntimeError('平面检测需要 Open3D') from exc
    finite=np.isfinite(xyz).all(axis=1); points=np.asarray(xyz[finite],dtype=np.float64); src=np.flatnonzero(finite).astype(np.int64)
    points,rows=_downsample_rows(points,max_input,seed); src=src[rows]
    remaining=points.copy(); remaining_idx=src.copy(); planes=[]
    for k in range(max_planes):
        if len(remaining)<max(min_points,ransac_n): break
        cloud=_build_cloud(remaining); model,inliers=cloud.segment_plane(distance_threshold=ransac_distance,ransac_n=ransac_n,num_iterations=iterations)
        idx=np.asarray(inliers,dtype=np.int64)
        if len(idx)<min_points: break
        pp=remaining[idx]; model=canonical_plane(model); residual=pp@model[:3]+model[3]; center=pp.mean(axis=0); u,v=plane_basis(model[:3]); vals=np.column_stack((pp@u,pp@v)); extent=np.ptp(vals,axis=0)
        sample,sub=_downsample_rows(pp,max_face,seed+k)
        planes.append(PlaneCandidate(k+1,'free_normal',model,sample,int(len(pp)),center,extent,float(np.sqrt(np.mean(residual**2))),remaining_idx[idx][sub]))
        keep=np.ones(len(remaining),dtype=bool); keep[idx]=False; remaining=remaining[keep]; remaining_idx=remaining_idx[keep]
    return planes


def _bbox_overlap(a:np.ndarray,b:np.ndarray)->float:
    low=np.maximum(a[0],b[0]); high=np.minimum(a[1],b[1]); overlap=np.maximum(high-low,0.0); area=float(np.prod(overlap)); aa=float(np.prod(np.maximum(a[1]-a[0],0.0))); ab=float(np.prod(np.maximum(b[1]-b[0],0.0))); return area/max(min(aa,ab),1e-12)


def group_plane_faces(planes:Sequence[PlaneCandidate],face_merge_distance:float,min_bbox_overlap:float,normal_merge_angle_deg:float=8.0)->list[PlaneGroup]:
    groups=[]
    for plane in sorted(planes,key=lambda p:p.position):
        assigned=None
        pu,pv=plane_basis(plane.normal)
        projected=np.column_stack((plane.inlier_points@pu,plane.inlier_points@pv))
        pb=np.array([projected.min(axis=0),projected.max(axis=0)])
        for group in groups:
            angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(plane.normal,group.normal)),-1.0,1.0))))
            if angle>normal_merge_angle_deg:
                continue
            close=abs(plane.position-group.position)<=face_merge_distance
            overlap=_bbox_overlap(pb,group.projected_bbox)
            if close and overlap>=min_bbox_overlap: assigned=group; break
        if assigned is None: assigned=PlaneGroup(len(groups)+1); groups.append(assigned)
        assigned.members.append(plane)
    groups.sort(key=lambda g:g.position)
    for i,g in enumerate(groups,1): g.group_id=i
    return groups


def select_partition_candidates(groups:Sequence[PlaneGroup],config:Mapping[str,object])->list[PartitionCandidate]:
    if not groups: return []
    groups=sorted(groups,key=lambda g:g.position); min_sep=float(config.get('pair_min_separation',1.0)); max_sep=float(config.get('pair_max_separation',6.0)); min_overlap=float(config.get('pair_min_overlap',0.35)); max_angle=float(config.get('pair_normal_angle_deg',12.0)); max_candidates=int(config.get('max_partitions',8)); candidates=[]
    for i in range(len(groups)):
        for j in range(i+1,len(groups)):
            a,b=groups[i],groups[j]; angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(a.normal,b.normal)),-1.0,1.0))));
            if angle>max_angle: continue
            normal=a.normal; signed=float(np.dot(b.centroid-a.centroid,normal)); separation=abs(signed)
            if not(min_sep<=separation<=max_sep): continue
            basis=plane_basis(a.normal)
            av=np.column_stack((a.points@basis[0],a.points@basis[1]))
            bv=np.column_stack((b.points@basis[0],b.points@basis[1]))
            overlap=_bbox_overlap(np.array([av.min(axis=0),av.max(axis=0)]),np.array([bv.min(axis=0),bv.max(axis=0)]))
            if overlap<min_overlap: continue
            area=min(a.projected_area,b.projected_area); score=100.0*overlap+np.log1p(max(area,0.0))-angle
            candidates.append(PartitionCandidate('free_normal',(a.group_id,b.group_id),(a,b),float(score),float(0.5*(a.position+b.position)),float(separation),'parallel_plane_pair'))
    candidates.sort(key=lambda c:c.score,reverse=True); return candidates[:max_candidates]


def plane_to_dict(p:PlaneCandidate)->dict:
    return {'plane_id':int(p.plane_id),'axis_name':p.axis_name,'model':p.model.tolist(),'normal':p.normal.tolist(),'count':int(p.count),'centroid':p.centroid.tolist(),'position':float(p.position),'projected_extent':p.projected_extent.tolist(),'projected_area':float(p.projected_area),'residual_rms':float(p.residual_rms)}

def group_to_dict(g:PlaneGroup)->dict:
    return {'group_id':int(g.group_id),'axis_name':g.axis_name,'normal':g.normal.tolist(),'position':float(g.position),'count':int(g.count),'projected_extent':g.projected_extent.tolist(),'projected_area':float(g.projected_area),'member_plane_ids':[int(m.plane_id) for m in g.members]}
