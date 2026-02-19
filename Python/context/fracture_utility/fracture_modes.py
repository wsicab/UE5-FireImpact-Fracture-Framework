# Include existing libraries
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, kron, save_npz
from scipy.sparse.linalg import lsqr, spsolve
from scipy.sparse.csgraph import connected_components
# Libigl
from scipy.spatial import cKDTree
import igl
from scipy.stats import multivariate_normal
import polyscope as ps
from .fracture_modes_parameters import fracture_modes_parameters
from .massmatrix_tets import massmatrix_tets
from .compute_fracture_modes import compute_fracture_modes
from .tictoc import tic,toc

import sys
import os
from gpytoolbox.copyleft import mesh_boolean

# TODO: CHECK I DIDN'T BREAK 3D MODES
# TODO: Write unit tests for all dimensions and boolean options

class fracture_modes:
    impact_precomputed = False
    impact_projected = False
    def __init__(self,vertices,elements):
        # Initialize this class with an n by 3 matrix of vertices and an n by 4 integer matrix of tet indeces
        self.vertices = vertices
        self.elements = elements
        # 파괴된 에지(연결이 끊긴 조각들)를 추적하기 위한 마스크
        self.broken_edges = None

    def compute_modes(self,parameters = fracture_modes_parameters()):
        # This is just a call to compute_fracture_modes, saving all the information we will need for impact projection
        self.exploded_vertices,self.exploded_elements,self.modes, self.labels, self.tet_to_vertex_matrix,self.tet_neighbors,self.massmatrix,self.unexploded_to_exploded_matrix = compute_fracture_modes(self.vertices,self.elements,parameters)
        self.verbose = parameters.verbose

    def transfer_modes_to_3d(self):
        # Computing modes in 3D can be slow. One trick we can do for efficiency is compute the modes in 1D and then transfer them to 3D by taking every possible combination of every 1D mode in the x, y and z directions
        modes_3d = []
        labels_3d = []
        for k in range(self.modes.shape[1]):
            # We do the k=0 case differently, because we know these will be the x, y, z displacements.
            if k==0:
                label = self.labels[:,0] # should be all zeros
                for j in range(3):
                    mode_3d = np.zeros((3*self.elements.shape[0],1))
                    indeces = j*self.elements.shape[0] + np.linspace(0,self.elements.shape[0]-1,self.elements.shape[0],dtype=int)
                    mode_3d[indeces] = np.mean(self.modes[:,0])
                    modes_3d.append(mode_3d)
                    labels_3d.append(np.reshape(label,(-1,1)))
            else:
                # In this case, we can remove one degree of freedom because we know displacements will be in the span of the modes
                labels = self.labels[:,k]
                n_components = np.max(labels.astype(int))+1
                # Compute per-piece displacements and tet-to-piece indeces for each mode
                displacements = np.zeros(n_components)
                indeces = []
                for j in range(n_components):
                    displacements[j] = np.mean(self.modes[labels==j,k])
                    indeces.append(np.nonzero(labels==j))
                # This is how many 3D modes we'll get... (a lot!)
                multiplicity = 3**(n_components-1) 
                for i in range(multiplicity):
                    # We will do this in a cute way, by taking the ternary expansion of the number i and use each ternary digit to choose whether we displace the piece in the x, y or z directions.
                    mode_3d = np.zeros((3*self.elements.shape[0],1))
                    ter = ternary(i,n_components-1)
                    for j in range(n_components-1):
                        remainder = int(ter[j])
                        mode_3d[remainder*self.elements.shape[0] + indeces[j][0]] = displacements[j]
                    modes_3d.append(mode_3d)
                    labels_3d.append(np.reshape(labels,(-1,1)))
        # Stack everything into 3D modes
        modes_3d = np.hstack(modes_3d)
        labels_3d = np.hstack(labels_3d)
        # Overwrite our previous 1D modes with the current 3D ones:
        self.modes = modes_3d
        self.labels = labels_3d
        # We also need to repeat the mass matrix that we use later to make it 3D
        self.massmatrix = kron(eye(3),self.massmatrix)
        # Ta-dah! We have 3D modes :)
        # Please we have no proof that these are exactly the same modes as if you had computed the 3D modes directly. I *think* they are, but maybe they're not! 

    def impact_precomputation(self, v_fine=None, f_fine=None, wave_h=1/30, upper_envelope=False, smoothing_lambda=100,use_robust_method=True):
        # This is not strictly part of the mode computation but it can be
        # precomputed to make the impact projection as fast as possible:
        tic()
        dim = self.modes.shape[0]//self.elements.shape[0] # mode dimension
        # Do the kronecker product by these matrices to replicate the "tile" behaviour in matlab and the "blockdiag" behaviour
        blockdiag_mat = eye(dim)
        repmat_mat = np.ones((dim,1))
        def ind2dim(I): # This will take anything indexing elements and make it index dim x elements
            J = [] 
            for d in range(dim):
                J.append(I + d*self.elements.shape[0])        
            return np.concatenate(J)
        
        

        
        # Tet-tet adjacency matrix
        tet_tet_adjacency_matrix = csr_matrix((np.ones(self.tet_neighbors.shape[0]),(self.tet_neighbors[:,0],self.tet_neighbors[:,1])),shape=(self.exploded_elements.shape[0],self.exploded_elements.shape[0]),dtype=int)
        # For efficienty, we will later store and do math on *per-piece* impacts, instead of per-tet. For this to work, we need to identify all the possible pieces that break off and mappings between tets and pieces.
        
        tet_tet_distances_rep = np.abs(self.modes[ind2dim(self.tet_neighbors[:,0]),:] - self.modes[ind2dim(self.tet_neighbors[:,1]),:]) # This is a dim x num_neighbor_pairs by num_modes matrix
        
        # Need to turn this into L2 distances per tet
        tet_tet_distances = np.zeros((self.tet_neighbors.shape[0],self.modes.shape[1]))
        for d in range(dim):
            indeces = d*self.tet_neighbors.shape[0] + np.linspace(0,self.tet_neighbors.shape[0]-1,self.tet_neighbors.shape[0],dtype=int)
            tet_tet_distances = tet_tet_distances + (tet_tet_distances_rep[indeces,:]**2.0)
        tet_tet_distances = np.sqrt(tet_tet_distances)

        # These are the tets that are together in every mode, which means that no impact projected onto our modes can separate them
        always_neighbors = self.tet_neighbors[np.all(tet_tet_distances<0.1,axis=1),:]
        # In this matrix, two tets are connected if they are always neighbors
        always_adjacency_matrix = csr_matrix((np.ones(always_neighbors.shape[0]),(always_neighbors[:,0],always_neighbors[:,1])),shape=(self.exploded_elements.shape[0],self.exploded_elements.shape[0]),dtype=int)
        # Taking connected components lets us know all the pieces that can break off, and tet-to-piece labeling
        
        n_total,self.all_modes_labels = connected_components(always_adjacency_matrix,directed=False)
        self.precomputed_num_pieces = n_total
        # ^ This lets us now build a piece_to_tet matrix mapping values in one to the other.
        I = np.linspace(0,self.elements.shape[0]-1,self.elements.shape[0])
        J = self.all_modes_labels
        self.piece_to_tet_matrix = csr_matrix((np.ones(I.shape[0]),(I,J)),shape=(self.elements.shape[0],self.precomputed_num_pieces),dtype=int)
        # Then, a piece adjacency graph
        piece_piece_adjacency_matrix = coo_matrix(((self.piece_to_tet_matrix.T @ tet_tet_adjacency_matrix @ self.piece_to_tet_matrix)>0).astype(int))
        self.piece_neighbors = np.vstack((np.array(piece_piece_adjacency_matrix.row),np.array(piece_piece_adjacency_matrix.col))).T

        # Also need the modes and labels defined at pieces
        self.piece_modes = np.zeros((dim*self.precomputed_num_pieces,self.modes.shape[1]))
        self.piece_labels = np.zeros((self.precomputed_num_pieces,self.modes.shape[1]))
        for k in range(self.modes.shape[1]):
            self.piece_modes[:,k] = lsqr(kron(blockdiag_mat,self.piece_to_tet_matrix),self.modes[:,k])[0]
            self.piece_labels[:,k] = np.rint(lsqr(self.piece_to_tet_matrix,self.labels[:,k])[0]).astype(int)
            
        self.piece_massmatrix = kron(blockdiag_mat,self.piece_to_tet_matrix.T) @ self.massmatrix @ kron(blockdiag_mat,self.piece_to_tet_matrix)

        #  This precomputation will allow us to approximate the propagation of any impact with the wave equation without a linear solve at runtime.
        # At runtime, we will project an impact u into the best-fit (LS) per-piece impact. So, we will do
        # piece_impact = (piece_to_tet' M piece_to_tet)^{-1} piece_to_tet' u
        # So let's define ^-------------  tet_to_piece  ----------------^
        self.tet_to_piece_matrix = spsolve((kron(blockdiag_mat,self.piece_to_tet_matrix.T) @ self.massmatrix @ kron(blockdiag_mat,self.piece_to_tet_matrix)),kron(blockdiag_mat,self.piece_to_tet_matrix.T))
        # Now, say we have a contact point t[i] at runtime and d is the vector with all zeros except on the i-th position (called "onehot" later). Then, what we'd want to make the impact vector is
        # u = C (M - hL)^{-1} M d
        #       ^--A--^
        self.A = massmatrix_tets(self.vertices,self.elements) - wave_h*igl.cotmatrix(self.vertices,self.elements)
        self.M = massmatrix_tets(self.vertices,self.elements)
        # (C blurs per-unexploded-vertex values into tets)
        self.C = 0.25*(self.tet_to_vertex_matrix.T @ self.unexploded_to_exploded_matrix)

        # But then the full runtime computation will be
        # piece_impact = tet_to_piece * C * A^{-1} * M * d
        # So we might as well call 
        # wave_piece_lsqr' = tet_to_piece * C  * A^{-1} * M
        self.wave_piece_lsqr = spsolve(kron(blockdiag_mat,self.A.T), kron(blockdiag_mat,self.C.T) @ self.massmatrix.T @ self.tet_to_piece_matrix.T)
        # and then we no longer have to do a solve at runtime
        # we only need to do
        # piece_impact = wave_piece_lsqr' M d


        # We also may want to use a Gaussian, instead of a wave equation, to blur our impact from the contact point to the rest of the shape. In case we want to do this, we pre-build a normal distribution (not sure if this is really necessary)
        self.rv = multivariate_normal([0.0,0.0,0.0], [[0.01, 0.0, 0.0], [0.0,0.01, 0.0],[0.0,0.0,0.01]])

        # So far, we have precomputed everything we need to answer the question "which pieces will our input mesh break into given an impact". But, often, our input mesh is not the mesh we want to break; rather, it is a cage of a finer mesh, and we want a broken version of the latter to be the output. In that case, what we'll need to precompute are the possible fracture pieces *of the fine mesh* as well as a piece-to-fine-mesh-vertex mapping

        # We will be appending to these to stack later
        running_n = 0 # for combining meshes
        fine_piece_vertices = []
        fine_piece_triangles = []
        Js = []

        if v_fine is not None:
            print(f"\n[Fine Mesh Input]")
            print(f"  Fine mesh: {v_fine.shape[0]} vertices, {f_fine.shape[0]} faces")
            print(f"  Coarse mesh: {self.vertices.shape[0]} vertices")
            print(f"  Precomputed pieces: {self.precomputed_num_pieces}")
            print(f"  Method: {'KDTree-based (robust)' if use_robust_method else 'Boolean (may fail)'}")
            
            running_n = 0
            fine_piece_vertices = []
            fine_piece_triangles = []
            Js = []
            
            successful_pieces = 0
            
            # [1] KDTree 기반 매핑 (PWN 문제)
            if use_robust_method:
                print("\n[Using Robust KDTree Method]")
                
                # 전체 fine mesh를 한 번에 처리
                # 각 fine vertex가 어느 coarse piece에 속하는지 결정
                
                # 1. Coarse mesh 각 piece 중심 계산
                piece_centers = []
                piece_radii = []
                for i in range(self.precomputed_num_pieces):
                    piece_tets = self.elements[self.all_modes_labels == i]
                    if len(piece_tets) == 0:
                        piece_centers.append(None)
                        piece_radii.append(0)
                        continue
                    
                    piece_verts = self.vertices[np.unique(piece_tets.flatten())]
                    center = piece_verts.mean(axis=0)
                    radius = np.max(np.linalg.norm(piece_verts - center, axis=1))
                    piece_centers.append(center)
                    piece_radii.append(radius * 1.2)  # 여유 공간
                
                # 2. 각 fine vertex를 가장 가까운 coarse piece에 할당
                fine_to_piece = np.full(v_fine.shape[0], -1, dtype=int)
                
                for i in range(self.precomputed_num_pieces):
                    if piece_centers[i] is None:
                        continue
                    
                    # 거리 계산
                    dists = np.linalg.norm(v_fine - piece_centers[i], axis=1)
                    
                    # 반경 내의 정점만 선택
                    in_radius = dists < piece_radii[i]
                    
                    # 아직 할당되지 않았거나 더 가까운 경우 할당
                    for idx in np.where(in_radius)[0]:
                        if fine_to_piece[idx] == -1 or \
                        dists[idx] < np.linalg.norm(v_fine[idx] - piece_centers[fine_to_piece[idx]]):
                            fine_to_piece[idx] = i
                
                # 3. Piece별로 fine mesh 분리
                for i in range(self.precomputed_num_pieces):
                    vert_mask = (fine_to_piece == i)
                    if not np.any(vert_mask):
                        print(f"  [Piece {i}] No vertices assigned")
                        continue
                    
                    # 해당 piece의 정점을 포함하는 삼각형 추출
                    face_mask = np.all(vert_mask[f_fine], axis=1)
                    if not np.any(face_mask):
                        print(f"  [Piece {i}] No faces with all vertices")
                        continue
                    
                    # 참조되지 않는 정점 제거
                    vi, fi = igl.remove_unreferenced(
                        v_fine, 
                        f_fine[face_mask]
                    )[:2]
                    
                    if vi.shape[0] < 3 or fi.shape[0] < 1:
                        print(f"  [Piece {i}] Too small: {vi.shape[0]}v, {fi.shape[0]}f")
                        continue
                    
                    fine_piece_vertices.append(vi.copy())
                    fine_piece_triangles.append(fi + running_n)
                    running_n += vi.shape[0]
                    Js.append(i * np.ones(vi.shape[0], dtype=int))
                    successful_pieces += 1
                    print(f"  [Piece {i}] OK: {vi.shape[0]}v, {fi.shape[0]}f")
            
            # [방법 2] Boolean 연산 (원본, PWN 문제 있음)
            else:
                print("\n[Using Boolean Method (may fail on non-PWN)]")
                
                for i in range(self.precomputed_num_pieces):
                    try:
                        # Coarse piece boundary 추출
                        vi, ti = igl.remove_unreferenced(
                            self.vertices,
                            self.elements[self.all_modes_labels == i, :]
                        )[:2]
                        
                        if vi.shape[0] == 0:
                            continue
                        
                        fi = boundary_faces_fixed(ti)
                        fi = fi[:, [1, 0, 2]]
                        
                        # [시도 1] 기본 Boolean
                        try:
                            from gpytoolbox.copyleft import mesh_boolean
                            vi_fine, fi_fine = mesh_boolean(
                                v_fine, f_fine.astype(np.int32),
                                vi, fi.astype(np.int32),
                                boolean_type='intersection'
                            )
                            
                            if vi_fine.shape[0] > 0 and fi_fine.shape[0] > 0:
                                fine_piece_vertices.append(vi_fine.copy())
                                fine_piece_triangles.append(fi_fine + running_n)
                                running_n += vi_fine.shape[0]
                                Js.append(i * np.ones(vi_fine.shape[0], dtype=int))
                                successful_pieces += 1
                                continue
                        except Exception as e:
                            print(f"  [Piece {i}] Boolean failed: {e}")
                        
                        # [시도 2] Fallback - 중심점 거리 기반
                        print(f"  [Piece {i}] Using distance-based fallback")
                        piece_center = vi.mean(axis=0)
                        piece_radius = np.max(np.linalg.norm(vi - piece_center, axis=1)) * 1.2
                        
                        dists = np.linalg.norm(v_fine - piece_center, axis=1)
                        vert_mask = dists < piece_radius
                        
                        if not np.any(vert_mask):
                            continue
                        
                        face_mask = np.all(vert_mask[f_fine], axis=1)
                        if not np.any(face_mask):
                            continue
                        
                        vi_fine, fi_fine = igl.remove_unreferenced(
                            v_fine,
                            f_fine[face_mask]
                        )[:2]
                        
                        if vi_fine.shape[0] >= 3 and fi_fine.shape[0] >= 1:
                            fine_piece_vertices.append(vi_fine.copy())
                            fine_piece_triangles.append(fi_fine + running_n)
                            running_n += vi_fine.shape[0]
                            Js.append(i * np.ones(vi_fine.shape[0], dtype=int))
                            successful_pieces += 1
                    
                    except Exception as e:
                        print(f"  [Piece {i}] Error: {e}")
                        continue
            
            print(f"\n[Results] Successful: {successful_pieces}/{self.precomputed_num_pieces}")
            
            # 결과 검증
            if len(fine_piece_vertices) == 0:
                print("\n[CRITICAL ERROR] All methods failed!")
                print("\nTroubleshooting:")
                print("  1. Check mesh alignment:")
                print(f"     Fine bbox: {v_fine.min(axis=0)} ~ {v_fine.max(axis=0)}")
                print(f"     Coarse bbox: {self.vertices.min(axis=0)} ~ {self.vertices.max(axis=0)}")
                print("  2. Try: use_robust_method=True (default)")
                print("  3. Verify meshes are in same coordinate system")
                self.fine_vertices = None
                self.fine_triangles = None
                return
            
            # 성공: Fine mesh 생성
            self.fine_vertices = np.vstack(fine_piece_vertices)
            self.fine_triangles = np.vstack(fine_piece_triangles)
            J = np.concatenate(Js)
            I = np.arange(self.fine_vertices.shape[0], dtype=int)
            
            print(f"\n[Success] Fine mesh generated:")
            print(f"  Total vertices: {self.fine_vertices.shape[0]}")
            print(f"  Total triangles: {self.fine_triangles.shape[0]}")
            
            # Piece 매핑
            self.piece_to_fine_vertices_matrix = csr_matrix(
                (np.ones(I.shape[0]), (I, J)),
                shape=(self.fine_vertices.shape[0], self.precomputed_num_pieces),
                dtype=int
            )
            
            # Fine labels
            self.fine_labels = np.zeros((self.fine_vertices.shape[0], self.modes.shape[1]))
            for k in range(self.modes.shape[1]):
                self.fine_labels[:,k] = self.piece_to_fine_vertices_matrix @ \
                                        lsqr(self.piece_to_tet_matrix, self.labels[:,k])[0]
        else:
            print("[Info] No fine mesh provided")
            self.fine_vertices = None
            self.fine_triangles = None

        # Store and print timing details
        self.t_impact_pre = round(toc(silence=True),5)
        if self.verbose:
            print("Impact precomputation: ", self.t_impact_pre," seconds. Will produce a maximum of",self.precomputed_num_pieces,"pieces.")
        # This is a boolean that we'll check before projecting an impact
        self.impact_precomputed = True


    def impact_projection(self, contact_point=None, threshold=0.02, wave=True, 
                          direction=np.array([1]), impact=None, project_on_modes=False, 
                          num_modes_used=None, locality_radius=0.4, use_locality=True, vertex_weights=None):
                            
        if (num_modes_used is None):
            num_modes_used = self.modes.shape[1]
        
        if not self.impact_precomputed:
            self.impact_precomputation()
            
        if self.broken_edges is None:
            self.broken_edges = np.zeros(self.piece_neighbors.shape[0], dtype=bool)

        tic()
        dim = self.modes.shape[0]//self.elements.shape[0] 
        
        # 1. Impact Vector 계산
        if (impact is None):
            assert(direction.shape[0]==dim)
            
            # Contact Point 전처리
            vmin = self.vertices.min(axis=0)
            vmax = self.vertices.max(axis=0)
            bbox_center = 0.5*(vmin+vmax)
            bbox_half   = 0.5*(vmax-vmin)

            cp = np.asarray(contact_point, dtype=float)
            if np.any(np.abs(cp - bbox_center) > 3.0*np.maximum(bbox_half, 1e-6)):
                cp = np.minimum(np.maximum(cp, vmin), vmax)

            idx_snapped = np.argmin(np.linalg.norm(self.vertices - cp, axis=1))
            cp_used = self.vertices[idx_snapped]
            self._last_contact_point_used = cp_used

            if wave:
                onehot = np.zeros(self.vertices.shape[0])
                dists = np.linalg.norm(self.vertices - np.tile(cp_used,(self.vertices.shape[0],1)), axis=1)

                if use_locality:
                    # 지역성 모드: 반경 내 정점에 가우시안 분포로 충격 적용
                    mask = dists < locality_radius
                    
                    # 활성 정점이 없으면 반경 확장
                    if np.sum(mask) == 0:
                        for grow in (2.0, 4.0, 8.0):
                            grow_r = locality_radius * grow
                            mask = dists < grow_r
                            if np.sum(mask) > 0:
                                locality_radius = grow_r
                                break
                    
                    onehot[mask] = np.exp(- (dists[mask]**2) / max((locality_radius/2.5)**2, 1e-8))
                    if self.verbose:
                         print(f"[Locality] Active vertices: {np.sum(mask)}")
                else:
                    # 원본 모드: 단순 거리 컷
                    onehot[dists < 0.05] = 1.0

                # 충격 강제 증폭 (파괴 유도)
                base_strength = 10.0 
                
                if vertex_weights is not None:
                    # 가중치가 1.0에 가까울수록 threshold를 기본값의 10% 수준까지 낮춤
                    # 이를 통해 한꺼번에 깨지는 것이 아니라 탄 부위부터 순차적으로 파괴
                    dynamic_threshold = threshold * (1.1 - np.mean(vertex_weights))
                else:
                    dynamic_threshold = threshold

                impact_1d = onehot
            else:
                impact_1d = 1.0*self.rv.pdf(self.vertices[self.elements[:,0],:] - np.tile(np.reshape(cp_used,(1,3)),(self.elements.shape[0],1)))
            
            impacts_dim = []
            for d in range(dim):
                impacts_dim.append(direction[d]*impact_1d)
            self.impact = np.concatenate(impacts_dim)
        else:
            assert(impact.shape[0]==dim*self.vertices.shape[0])
            wave = False

        blockdiag_kron = eye(dim)

        # 2. Piece 단위 Impact 투영
        if wave:
            self.piece_impact = self.wave_piece_lsqr.T @ self.impact
        else:
            self.piece_impact = kron(blockdiag_kron,self.tet_to_piece_matrix @ self.massmatrix) @ self.impact
            

        # [Mode Locality] 모드별 지역성 가중치 적용 (먼 곳의 모드는 억제)
        if use_locality and contact_point is not None:
            # 전역 piece별 중심 거리 계산 (On-the-fly)
            if not hasattr(self, "_piece_centers"):
                self._piece_centers = np.zeros((self.precomputed_num_pieces, 3))
                self._piece_valid   = np.zeros(self.precomputed_num_pieces, dtype=bool)
                for p in range(self.precomputed_num_pieces):
                    tet_mask = (self.all_modes_labels == p)
                    if np.any(tet_mask):
                        tets = self.elements[tet_mask]
                        tet_centers = np.mean(self.vertices[tets], axis=1)
                        self._piece_centers[p] = tet_centers.mean(axis=0)
                        self._piece_valid[p]   = True
            
            cpw = getattr(self, "_last_contact_point_used", np.asarray(contact_point))
            piece_dists = np.full(self.precomputed_num_pieces, np.inf)
            piece_dists[self._piece_valid] = np.linalg.norm(self._piece_centers[self._piece_valid] - cpw[None, :], axis=1)

            mode_locality_weights = np.zeros(self.modes.shape[1])
            for k in range(self.modes.shape[1]):
                labels_k = self.piece_labels[:, k].astype(int)
                if labels_k.max() <= 0: continue
                
                p_near = int(np.argmin(piece_dists))
                label_near = labels_k[p_near]
                
                diff_mask = (labels_k != label_near) & self._piece_valid
                if not np.any(diff_mask): continue
                
                closest_dist = float(np.min(piece_dists[diff_mask]))
                mode_locality_weights[k] = np.exp(- (closest_dist ** 2) / (locality_radius ** 2))
            
            # 가중치 적용하여 재합성
            weighted_impact = np.zeros_like(self.piece_impact)
            for k in range(self.modes.shape[1]):
                if mode_locality_weights[k] <= 1e-6: continue
                coef = (self.piece_impact.T @ self.piece_massmatrix @ self.piece_modes[:, k])
                weighted_impact += coef * self.piece_modes[:, k] * mode_locality_weights[k]
            self.piece_impact = weighted_impact

        
        if project_on_modes:
            self.projected_impact = np.zeros((self.piece_impact.shape[0]))
            for k in range(num_modes_used):
                self.projected_impact = self.projected_impact + (self.piece_impact.T @ self.piece_massmatrix @ self.piece_modes[:,k])*self.piece_modes[:,k]
        else:
            self.projected_impact = self.piece_impact

        # 3. 조각 간 거리(변형률 차이) 계산
        piece_distances = np.linalg.norm(
            np.reshape(self.projected_impact,(-1,dim),order='F')[self.piece_neighbors[:,0],:] - 
            np.reshape(self.projected_impact,(-1,dim),order='F')[self.piece_neighbors[:,1],:],
            axis=1
        )

        # [수정] 고정된 threshold 대신 계산된 dynamic_threshold 사용
        current_breaks = piece_distances >= dynamic_threshold
        self.broken_edges = self.broken_edges | current_breaks

        # 4. 연결 성분 분석 (파괴 판정)
        def cc_with_persistence():
            keep_mask = ~self.broken_edges # 파괴되지 않은 에지만 유지
            piece_neighbors_filtered = self.piece_neighbors[keep_mask,:]
            
            if piece_neighbors_filtered.shape[0] == 0:
                adj = csr_matrix((self.precomputed_num_pieces, self.precomputed_num_pieces), dtype=int)
            else:
                adj = csr_matrix((np.ones(piece_neighbors_filtered.shape[0]),
                                (piece_neighbors_filtered[:,0], piece_neighbors_filtered[:,1])),
                                shape=(self.precomputed_num_pieces,self.precomputed_num_pieces),dtype=int)
            return connected_components(adj, directed=False)

        self.n_pieces_after_impact, self.piece_labels_after_impact = cc_with_persistence()
        
        self.impact_projected = True
        self.t_impact = round(toc(silence=True),5)
        
        if self.verbose:
            print("Impact projection: ", self.t_impact," seconds. Produced",self.n_pieces_after_impact, "pieces.")
        
        self.tet_labels_after_impact = self.piece_to_tet_matrix @ self.piece_labels_after_impact

        if(self.fine_vertices is not None):
            self.fine_vertex_labels_after_impact = self.piece_to_fine_vertices_matrix @ self.piece_labels_after_impact

        if wave:
            self.impact_vis = spsolve(kron(blockdiag_kron,self.A),kron(blockdiag_kron,self.M) @ self.impact)
        else:
            self.impact_vis = self.impact.copy()

        self.impact_vis = np.reshape(self.impact_vis,(-1,dim),order='F')
    
    def write_generic_data_compressed(self,filename):
        write_file_name = os.path.join(filename,"compressed_mesh.obj")
        write_data_name = os.path.join(filename,"compressed_data.npz")
        igl.write_obj(write_file_name,self.fine_vertices,self.fine_triangles)
        save_npz(write_data_name, self.piece_to_fine_vertices_matrix)

    def write_segmented_output_compressed(self,filename = None):
        write_fracture_name = os.path.join(filename,"compressed_fracture.npy")
        np.save(write_fracture_name,self.piece_labels_after_impact)

    def write_segmented_modes_compressed(self,filename = None):
        for j in range(self.modes.shape[1]):
            new_dir = os.path.join(filename,"mode_"+str(j))
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            write_fracture_name = os.path.join(new_dir,"compressed_fracture.npy")
            mode_labels = self.piece_labels[:,j]
            np.save(write_fracture_name,mode_labels)

    def return_ui_gi(self, pieces = True):
        print(f"=== new_return_ui_gi_ 시작 ===")
        print(f"n_pieces_after_impact: {self.n_pieces_after_impact}")
    
        if not hasattr(self, 'tet_labels_after_impact'):
            print("ERROR: tet_labels_after_impact가 없습니다!")
            return 0, [], []
    
        print(f"tet_labels_after_impact 범위: {np.min(self.tet_labels_after_impact)} ~ {np.max(self.tet_labels_after_impact)}")
        
        Vs = []
        Fs = []
        running_n = 0 
    
        # 각 조각별로 처리
        for i in range(self.n_pieces_after_impact):
            # print(f"\n--- 조각 {i} 처리 (simple coarse) ---")
        
            try:
            # 해당 조각에 속하는 tetrahedra 찾기
                piece_mask = (self.tet_labels_after_impact == i)
                piece_tets = self.elements[piece_mask, :]
                # print(f"조각 {i}에 속하는 tet 개수: {len(piece_tets)}")
            
                if len(piece_tets) == 0:
                    continue
            
                # [수정] 최소 크기 체크 완화 (1개 이상이면 허용)
                if len(piece_tets) < 1:
                    continue
            
            # 참조되지 않는 정점 제거
                try:
                    vi, ti = igl.remove_unreferenced(self.vertices, piece_tets)[:2]
                except Exception as e:
                    print(f"조각 {i}: remove_unreferenced 실패 - {e}")
                    continue
            
                if len(vi) == 0 or len(ti) == 0:
                    continue
            
            # Tetrahedra의 surface faces 추출
                try:
                    fi = boundary_faces_fixed(ti)
                except Exception as e:
                    print(f"조각 {i}: boundary_faces 실패 - {e}")
                    continue
            
                if len(fi) == 0:
                    continue
                
            # libigl 순서 수정 
                try:
                    if fi.shape[1] >= 3:
                        fi = fi[:, [1, 0, 2]]  # 순서 변경
                    else:
                        continue
                except Exception as e:
                    continue
            
            # 중복 정점 제거 
                try:
                    ui, I, J, _ = igl.remove_duplicate_vertices(vi, fi, 1e-10)
                
                # J 유효한지 확인
                    if len(J) != len(vi):
                        continue
                    
                # Face 인덱스 재매핑
                    gi = J[fi]
                
                # 유효하지 않은 face 제거 (-1 인덱스 등)
                    valid_face_mask = (gi >= 0).all(axis=1) & (gi < len(ui)).all(axis=1)
                    gi = gi[valid_face_mask]
                
                except Exception as e:
                    continue
            
            # [수정] 최종 유효성 검사 완화 (최소 3개 정점, 1개 면)
            # 납작한 조각(Triangle)도 허용
                if len(ui) >= 3 and len(gi) >= 1:  
                # 면 인덱스가 정점 범위 내에 있는지 확인
                    if np.max(gi) < len(ui):
                        Vs.append(ui.copy())  # 복사본 저장
                        Fs.append(gi.copy()) 
                        # print(f"조각 {i} 성공: {len(ui)} 정점, {len(gi)} 면")
                    else:
                        print(f"조각 {i}: 면 인덱스 오류")
                else:
                    pass # 너무 작은 조각 무시
                
            except Exception as e:
                print(f"조각 {i} 전체 처리 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
    
        print(f"\n=== 최종 결과 ===")
        print(f"총 {len(Vs)}개의 유효한 조각 생성됨")
    
    # 결과 검증
        for i, (v, f) in enumerate(zip(Vs, Fs)):
            # print(f"조각 {i}: {len(v)} 정점, {len(f)} 면")
            if len(v) > 0:
                pass
                # print(f"  정점 범위: [{np.min(v, axis=0)}, {np.max(v, axis=0)}]")
    
        return len(Vs), Vs, Fs
    
    def new_return_ui_gi_v6(self):
        getter_names = ("new_return_ui_gi_v2", "new_return_ui_gi_", "new_return_ui_gi")
        for name in getter_names:
            fn = getattr(self, name, None)
            if fn is None:
                continue
            try:
                n, Vs, Fs = fn()
                if n and sum(len(v) for v in Vs) > 0:
                    return n, Vs, Fs
            except Exception as e:
                print(f"[warn] {name} failed in v6: {e}")

       
        if hasattr(self, "_extract_coarse_pieces"):
            return self._extract_coarse_pieces()
        return 0, [], []

    def new_return_ui_gi(self):
        # ... (생략, 원본 유지) ...
        return self.new_return_ui_gi_v2()
   
    def new_return_ui_gi_v2(self):
        if not self.impact_projected:
            print("[ERROR] Impact not projected yet.")
            return 0, [], []
        
        # Fine mesh 필수 체크
        if self.fine_vertices is None or self.fine_triangles is None:
            print("\n[CRITICAL ERROR] Fine mesh not available!")
            print("Did you call: fm.impact_precomputation(v_fine=..., f_fine=...)?")
            return 0, [], []
        
        print(f"\n[Extracting Fine Mesh Pieces]")
        print(f"  Fine mesh: {self.fine_vertices.shape[0]} vertices")
        print(f"  Expected pieces: {self.n_pieces_after_impact}")
        
        self.fine_vertex_labels_after_impact = \
            self.piece_to_fine_vertices_matrix @ self.piece_labels_after_impact
        
        Ui = []
        Gi = []
        running_n = 0
        
        for i in range(self.n_pieces_after_impact):
            try:
                # Fine mesh에서 해당 piece 추출
                tri_labels = self.fine_vertex_labels_after_impact[self.fine_triangles[:,0]]
                piece_mask = (tri_labels == i)
                
                if not np.any(piece_mask):
                    print(f"  [Piece {i}] No triangles (piece might be too small)")
                    continue
                
                # 참조되지 않는 정점 제거
                vi, fi = igl.remove_unreferenced(
                    self.fine_vertices,
                    self.fine_triangles[piece_mask, :]
                )[:2]
                
                # 최소 크기 검증
                if vi.shape[0] < 3 or fi.shape[0] < 1:
                    print(f"  [Piece {i}] Too small: {vi.shape[0]}v, {fi.shape[0]}f")
                    continue
                
                # 중복 제거
                ui, _, J, _ = igl.remove_duplicate_vertices(vi, fi, 1e-10)
                gi = J[fi]
                
                # 인덱스 유효성 검증
                if gi.shape[0] == 0 or np.max(gi) >= len(ui):
                    print(f"  [Piece {i}] Invalid face indices")
                    continue
                
                # 추가
                Ui.append(ui.copy())
                Gi.append(gi.copy() + running_n)
                running_n += ui.shape[0]
                print(f"  [Piece {i}] OK: {ui.shape[0]}v, {gi.shape[0]}f")
                
            except Exception as e:
                print(f"  [Piece {i}] Error: {e}")
                continue
        
        if len(Ui) == 0:
            print("\n[ERROR] No valid pieces extracted from fine mesh!")
            return 0, [], []
        
        print(f"\n[Success] Extracted {len(Ui)} pieces from FINE mesh")
        return len(Ui), Ui, Gi


    def write_segmented_output(self,filename = None,pieces=False):
        # All this routine is doing is write the fractured output, as a triangle mesh with num_broken_pieces connected components, so you can load it into an animation in another software. 
        # If you gave our algorithm a fine mesh, it will write the fractured fine mesh directly.
        # What variables do I need for this:
        # General, per-object data:
        # self.fine_vertices, self.fine_triangles
        # self.piece_to_fine_vertices_matrix
        # Per-impact data:
        # self.piece_labels_after_impact

        assert(self.impact_projected)
        self.fine_vertex_labels_after_impact = self.piece_to_fine_vertices_matrix @ self.piece_labels_after_impact
        Vs = []
        Fs = []
        running_n = 0 # for combining meshes
        for i in range(self.n_pieces_after_impact):
                if (self.fine_vertices is not None):
                    tri_labels = self.fine_vertex_labels_after_impact[self.fine_triangles[:,0]]
                    if np.any(tri_labels==i):
                        vi, fi = igl.remove_unreferenced(self.fine_vertices,self.fine_triangles[tri_labels==i,:])[:2]
                    else:
                        continue
                else:
                    vi, ti = igl.remove_unreferenced(self.vertices,self.elements[self.tet_labels_after_impact==i,:])[:2]
                    fi = boundary_faces_fixed(ti)
                ui, I, J, _ = igl.remove_duplicate_vertices(vi,fi,1e-10)
                gi = J[fi]
                
                if pieces:    
                    write_file_name = os.path.join(filename,"piece_" + str(i) + ".obj")
                    igl.write_obj(write_file_name,ui,gi)
                Vs.append(ui)
                Fs.append(gi.copy()) 
                running_n = running_n + ui.shape[0]
        self.mesh_to_write_vertices = np.vstack(Vs)
        self.mesh_to_write_triangles = np.vstack(Fs)
        if (filename is not None):
            if (not pieces):
                igl.write_obj(filename,self.mesh_to_write_vertices,self.mesh_to_write_triangles)
    
    def write_segmented_modes(self,filename = None,pieces=False):

        for j in range(self.modes.shape[1]):
            Vs = []
            Fs = []
            self.fine_labels = self.piece_to_fine_vertices_matrix @ self.piece_labels
            if pieces:
                new_dir = os.path.join(filename,"mode_"+str(j))
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)
            running_n = 0 # for combining meshes
            self.fine_labels = self.fine_labels.astype(int)
            for i in range(np.max(self.fine_labels[:,j])+1): # 
                
                # Double check this loop limit
                if (self.fine_vertices is not None):
                    tri_labels = self.fine_labels[self.fine_triangles[:,0],j]
                    if np.any(tri_labels==i):
                        vi, fi = igl.remove_unreferenced(self.fine_vertices,self.fine_triangles[tri_labels==i,:])[:2]
                        
                    else:
                        continue
                ui, I, J, _ = igl.remove_duplicate_vertices(vi,fi,1e-10)
                gi = J[fi]
                if pieces:
                    write_file_name = os.path.join(new_dir,"piece_" + str(i) + ".obj")
                    igl.write_obj(write_file_name,ui,gi)
                Vs.append(ui)
                Fs.append(gi.copy())
                running_n = running_n + ui.shape[0]
            self.mesh_to_write_vertices = np.vstack(Vs)
            self.mesh_to_write_triangles = np.vstack(Fs)
            if (filename is not None):
                if (not pieces):
                    igl.write_obj(filename + "_mode_" + str(j) + ".obj",self.mesh_to_write_vertices,self.mesh_to_write_triangles)

# [수정됨] 튜플 반환 오류 수정 함수
def boundary_faces_fixed(ti):
    if ti.shape[0] == 0:
        return np.array([], dtype=np.int32).reshape(0, 3)

    ti = np.reshape(ti, (-1, 4))
    out = igl.boundary_facets(ti)

    # igl 버전이나 바인딩에 따라 튜플을 반환하는 경우가 있어 체크
    if isinstance(out, tuple):
        if len(out) > 0:
            return out[0]
        else:
            return np.array([], dtype=np.int32).reshape(0, 3)
    
    return out

def blur_onto_vertices(F,f_vals):
    v_vals = np.zeros((np.max(F.astype(int)) + 1,f_vals.shape[1]))
    valences = np.zeros((np.max(F.astype(int)) + 1,1))
    vec_kron = np.ones((F.shape[1],1))
    for i in range(F.shape[0]):
        valences[F[i,:]] = valences[F[i,:]] + 1
        v_vals[F[i,:],:] = v_vals[F[i,:],:] + f_vals[i,:]
    return v_vals/np.tile(valences,(1,f_vals.shape[1]))

def ternary(n,m):
    # if n == 0:
    #     return '0'
    nums = []
    for it in range(m):
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))