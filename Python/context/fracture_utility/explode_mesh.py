# Include existing libraries
import numpy as np
from scipy.sparse import csr_matrix

# Libigl
import igl

import numpy as np

def tet_tet_adjacency_like_igl(T: np.ndarray):
   
    T = np.asarray(T)
    assert T.ndim == 2 and T.shape[1] == 4, "T must be (m,4) int array"
    m = T.shape[0]
    TT  = np.full((m, 4), -1, dtype=np.int64)
    TTi = np.full((m, 4), -1, dtype=np.int64)

    # libigl의 면 정의(위 순서를 정확히 따름)
    faces_local = np.array([
        [0,1,2],  # face 0
        [0,1,3],  # face 1
        [1,2,3],  # face 2
        [2,0,3],  # face 3
    ], dtype=np.int64)

    # 무방향 면 키 -> (tet_index, local_face_index)
    face_map = {}

    for ti in range(m):
        tet = T[ti]
        for fj in range(4):
            face = tet[faces_local[fj]]
            # 공유면 매칭을 위해 정렬된 튜플을 키로 사용 (libigl의 면 공유 원리와 동일)
            key = tuple(sorted(face.tolist()))
            if key in face_map:
                ti2, fj2 = face_map[key]
                # 상호 매칭
                TT[ti,  fj]  = ti2
                TTi[ti, fj]  = fj2
                TT[ti2, fj2] = ti
                TTi[ti2, fj2]= fj
            else:
                face_map[key] = (ti, fj)

    return TT, TTi


# @profile
def explode_mesh(vertices,elements,num_quad=3):
    # Vertices will have the following order:
    # [tet_1_vert_1,tet_1_vert_2,....,tet_n_vert_3,tet_n_vert_4]
    # We can get these indeces by reshaping (row-first) the elements
    vert_indeces = np.squeeze(np.reshape(elements,(-1,1)))
    exploded_vert_indeces = np.linspace(0,4*elements.shape[0]-1,4*elements.shape[0],dtype=int)
    exploded_elements = np.reshape(exploded_vert_indeces,(-1,4))
    exploded_vertices = vertices[vert_indeces,:]

    # Build the matrix that takes any scalar-valued function on tets and maps it to the four vertices of the tet in the exploded mesh
    I = exploded_vert_indeces
    J = np.kron(np.linspace(0,elements.shape[0]-1,elements.shape[0],dtype=int),np.array([1.0,1.0,1.0,1.0],dtype=int))
    vals = np.ones(I.shape)
    tet_to_vertex_matrix = csr_matrix((vals,(I,J)),shape=(4*elements.shape[0],elements.shape[0]))

    # Make unexploded to exploded matrix
    J = vert_indeces
    I = exploded_vert_indeces
    vals = np.ones((I.shape[0]))
    unexploded_to_exploded_matrix = csr_matrix((vals,(I,J)),shape=(4*elements.shape[0],vertices.shape[0]))
   
    # Make discontinuity matrix (use quadrature weights)
    if num_quad==3:
        quad_weights = np.array([[2/3,1/3,1/3],[1/3,2/3,1/3],[1/3,1/3,2/3]])
    elif num_quad==1:
        quad_weights = np.array([[1.0,0.0,0.0]])
    # There will be three nodes per internal face
    TT, TTi = tet_tet_adjacency_like_igl(elements)
    # TT #T by #4 adjacency matrix, the element i,j is the id of the tet adjacent to the j face of tet i
    # TTi #T by #4 adjacency matrix, the element i,j is the id of face of the tet TT(i,j) that is adjacent to tet i
    # the first face of a tet is [0,1,2], the second [0,1,3], the third [1,2,3], and the fourth [2,0,3].

    # Quickly let's get indeces of neighboring tets.
    tet_neighbors_j = np.reshape(TT,(-1,1))
    tet_neighbors_i = np.reshape(np.kron(np.linspace(0,elements.shape[0]-1,elements.shape[0],dtype=int),np.array([1.0,1.0,1.0,1.0],dtype=int)),(-1,1))
    tet_neighbors = np.hstack((tet_neighbors_i,tet_neighbors_j))
    tet_neighbors = tet_neighbors[tet_neighbors_j[:,0]>-1,:]
    # that's it, keep going building D

    I = np.zeros(6*num_quad*4*exploded_elements.shape[0])
    J = np.zeros(6*num_quad*4*exploded_elements.shape[0])
    vals = np.zeros(6*num_quad*4*exploded_elements.shape[0])
    num_tets = elements.shape[0]
    # the matrix row ordering will be 
    # I = 4*num_tets*qi + 4*i + nn

    # Now we have to figure out which *vertices* match
    tet_face_ordering = np.array([[0,1,2],
                                  [0,1,3],
                                  [1,2,3],
                                  [2,0,3]])
    all_faces = np.vstack((elements[:,[0,1,2]],elements[:,[0,1,3]],elements[:,[1,2,3]],elements[:,[2,0,3]]))

    areas = igl.doublearea(vertices,all_faces)

    #areas[vertices[all_faces[:,0],2]>0] = 100*areas[vertices[all_faces[:,0],2]>0]
    # This for loop is obviously suboptimal, should be vectorized. However,
    # it is far from the bottleneck (dominated by eigsh call), so it is not
    # very important to do it.
    for i in range(elements.shape[0]):
        tet_1 = i
        for nn in range(4):
            tet_2 = TT[i,nn]
            if tet_2>-1 and tet_1>tet_2: # internal faces only, and only once
                face_area = areas[nn*elements.shape[0] + i]
                # tet_1 and tet_2 are neighbors. The face nn of tet_1 neighbors the face TTi[i,j] of tet_2
                # So the six exploded vertices we are dealing with are 4*tet_1 + tet_face_ordering[nn,:] and 4*tet_2 + tet_face_ordering[TTi[i,nn],:]
                # What we want to know is which of these six vertices are duplicates
                # In the unexploded mesh, these vertices are (in the same order) elements[tet_1,tet_face_ordering[nn,:]] and elements[tet_2,tet_face_ordering[TTi[i,nn],:]]
                unexploded_indeces_tet_1 = elements[tet_1,tet_face_ordering[nn,:]]
                unexploded_indeces_tet_2 = elements[tet_2,tet_face_ordering[TTi[i,nn],:]]
                exploded_indeces_tet_1 = 4*tet_1 + tet_face_ordering[nn,:]
                exploded_indeces_tet_2 = 4*tet_2 + tet_face_ordering[TTi[i,nn],:]

                # Then all we need is a mapping giving the equality relationship between unexploded_indeces_tet_1 and 2. We can do this by sorting them
                argsort_1 = np.argsort(unexploded_indeces_tet_1)
                argsort_2 = np.argsort(unexploded_indeces_tet_2)

                # There should be 3*6 new non-zero entries in the matrix from this internal face, in 3 different rows
                for qi in range(num_quad):
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 0] = 4*num_tets*qi + 4*i + nn
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 1] = 4*num_tets*qi + 4*i + nn
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 2] = 4*num_tets*qi + 4*i + nn
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 3] = 4*num_tets*qi + 4*i + nn
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 4] = 4*num_tets*qi + 4*i + nn
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 5] = 4*num_tets*qi + 4*i + nn
                    J[ 6*(qi + num_quad*nn + num_quad*4*i) + 0] = exploded_indeces_tet_1[argsort_1[0]]
                    J[ 6*(qi + num_quad*nn + num_quad*4*i) + 1] = exploded_indeces_tet_2[argsort_2[0]]
                    J[ 6*(qi + num_quad*nn + num_quad*4*i) + 2] = exploded_indeces_tet_1[argsort_1[1]]
                    J[ 6*(qi + num_quad*nn + num_quad*4*i) + 3] = exploded_indeces_tet_2[argsort_2[1]]
                    J[ 6*(qi + num_quad*nn + num_quad*4*i) + 4] = exploded_indeces_tet_1[argsort_1[2]]
                    J[ 6*(qi + num_quad*nn + num_quad*4*i) + 5] = exploded_indeces_tet_2[argsort_2[2]]
                    vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 0] = quad_weights[qi,0]*face_area
                    vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 1] = -quad_weights[qi,0]*face_area
                    vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 2] = quad_weights[qi,1]*face_area
                    vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 3] = -quad_weights[qi,1]*face_area
                    vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 4] = quad_weights[qi,2]*face_area
                    vals[ 6*(qi + num_quad*nn + num_quad*4*i) + 5] = -quad_weights[qi,2]*face_area
            else:
                for qi in range(num_quad):
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 0] = -1
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 1] = -1
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 2] = -1
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 3] = -1
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 4] = -1
                    I[ 6*(qi + num_quad*nn + num_quad*4*i) + 5] = -1

    J = J[I>-1]
    vals = vals[I>-1]
    I = I[I>-1]

    discontinuity_matrix = csr_matrix((vals,(I,J)),shape=(num_quad*4*exploded_elements.shape[0],exploded_vertices.shape[0]))

    num_nonzeros = np.diff(discontinuity_matrix.indptr)
    discontinuity_matrix =  discontinuity_matrix[num_nonzeros != 0]

    return exploded_vertices, exploded_elements, discontinuity_matrix, unexploded_to_exploded_matrix, tet_to_vertex_matrix, tet_neighbors