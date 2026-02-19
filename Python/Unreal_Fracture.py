import numpy as np
import igl
import tetgen
from context import fracture_utility as fracture
import gpytoolbox
from gpytoolbox.copyleft import lazy_cage
import sys
import os

model1 = "C:/Users/holykite/Documents/Unreal Projects/FireImpact_Fracture/Content/Models/armadillo_Type1.obj"  
model2 = "C:/Users/holykite/Documents/Unreal Projects/FireImpact_Fracture/Content/Models/bunny_oded.obj"
model3 = "C:/Users/holykite/Documents/Unreal Projects/FireImpact_Fracture/Content/Models/octopus.obj"
model4 = "C:/Users/holykite/Documents/Unreal Projects/FireImpact_Fracture/Content/Models/rocking_chair.obj"
model5 = "C:/Users/holykite/Documents/Unreal Projects/FireImpact_Fracture/Content/Models/chair.obj"
#model6 = "C:/Users/holykite/Documents/Unreal Projects/FireImpact_Fracture/Content/Models/old_stool.obj"
#model7 = "C:/Users/holykite/Documents/Unreal Projects/FireImpact_Fracture/Content/Models/propeller.obj"
#model8 = "C:/Users/holykite/Documents/Unreal Projects/FireImpact_Fracture/Content/Models/tower_op.obj"
model9 = "C:/Users/holykite/Documents/Unreal Projects/FireImpact_Fracture/Content/Models/rehand.obj"


class Unreal_Fracture:
    def __init__(self):
        print("Loading mesh...")
        self.v_fine, self.f_fine = igl.read_triangle_mesh(model9)
        self.v_fine *= 0.01
        
        print("Normalizing points...")
        self.v_fine = gpytoolbox.normalize_points(self.v_fine)
        
        print("Creating lazy cage...")
        self.v, self.f = lazy_cage(self.v_fine, self.f_fine, num_faces=2000)

        print("Creating tetrahedral mesh...")
        self.tgen = tetgen.TetGen(self.v, self.f)
        self.nodes, self.elements = self.tgen.tetrahedralize()

        print("Initializing fracture modes...")
        self.modes = fracture.fracture_modes(self.nodes, self.elements)
        
        self.params = fracture.fracture_modes_parameters(num_modes=8, verbose=True, d=1)

        self.contact_point = self.nodes[1,:]

        print("Computing modes...")
        self.modes.compute_modes(parameters=self.params)
        
        print("Impact precomputation...")
        # v_fine, f_fine 
        self.modes.impact_precomputation(v_fine=self.v_fine, f_fine=self.f_fine)
        
        # Fine mesh 
        if hasattr(self.modes, 'fine_vertices') and self.modes.fine_vertices is not None:
            print(f"Fine mesh generate : {self.modes.fine_vertices.shape[0]} vertices")
        else:
            print("WARNING: Fine mesh failed. - use coarse mesh ")
        
        print("Initial impact projection...")
        self.modes.impact_projection(contact_point=self.contact_point)
        
        self.initialize_fragments()
        
        self.vertex_weights = np.ones(self.nodes.shape[0])  
        print(f"Vertex weights initialized: {len(self.vertex_weights)} vertices")
        
        print("Initialization complete!")
    
    def initialize_fragments(self):
        try:
            self.pieces, self.Vs, self.Fs = self.modes.new_return_ui_gi()
            print(f"debris : {self.pieces}pieces")

            if self.pieces > 0 and len(self.Vs) > 0:
                for i, v in enumerate(self.Vs):
                    print(f"  fragment {i}: {len(v)} vertex")
        except Exception as e:
            print(f"debris error: {e}")
            self.pieces, self.Vs, self.Fs = 1, [self.v_fine], [self.f_fine]

    def runtime_impact_projection(self, impact_point, vertex_weights=None):
        print(f"\n{'='*60}")
        print(f"Fire simulation start")
        print(f"{'='*60}")
        print(f"point (input): {impact_point}")
        
        try:
            print("\n[1/5] ")
            impact_array = np.array(impact_point, dtype=float)
            
            v_min = self.nodes.min(axis=0)
            v_max = self.nodes.max(axis=0)
            v_center = (v_min + v_max) / 2.0
            v_extent = v_max - v_min
            
            # Unreal cm → m 
            impact_array = impact_array / 100.0
            
            impact_relative = impact_array - v_center
            
            # extent
            impact_normalized = impact_relative / (v_extent / 2.0)
            
            # radius
            impact_normalized = np.clip(impact_normalized, -1.5, 1.5)

            impact_final = impact_normalized * (v_extent / 2.0) + v_center
            
            print(f"   impact point  (cm→m): {impact_array}")
            print(f"   mesh location : {impact_final}")
            
            print("\n[2/5] ...")
            if vertex_weights is not None:
                if len(vertex_weights) != len(self.vertex_weights):
                    avg_weight = np.mean(vertex_weights)
                    self.vertex_weights = self.vertex_weights * avg_weight
                    print(f"   error →  {avg_weight:.3f} used ")
                else:
                    self.vertex_weights = self.vertex_weights * vertex_weights
                
                print(f"   weight radius : {np.min(self.vertex_weights):.3f} ~ {np.max(self.vertex_weights):.3f}")
                weak_vertices = np.sum(self.vertex_weights < 0.5)
                print(f"   weaken vertex: {weak_vertices} ({weak_vertices/len(self.vertex_weights)*100:.1f}%)")
            else:
                print("   there is no fire-based weight.")
            
            print("\n[3/5] ")
            direction = np.array([1])
            
            self.modes.impact_projection(
                contact_point=impact_final,  
                threshold=0.4,
                direction=direction,
                wave=True,
                use_locality=True,
                locality_radius=0.6,
                vertex_weights=self.vertex_weights
            )
            

            print("\n[4/5] generating fragment...")
            self.pieces, self.Vs, self.Fs = self.modes.new_return_ui_gi_v2()
            
            print(f"   pieces : {self.pieces}")
            for i in range(min(3, len(self.Vs))):
                if i < len(self.Vs) and len(self.Vs[i]) > 0:
                    print(f"   - pieces  {i}: {len(self.Vs[i])}vertex")
            
     
            print("\n[5/5] ")
            if self.pieces == 0 or len(self.Vs) == 0:
                print("   no valid fragment. using original mesh")
                self.pieces = 1
                self.Vs = [self.v_fine]
                self.Fs = [self.f_fine]
            
            print(f"\n{'='*60}")
            print(f"fracture complete : {self.pieces}")
            print(f"{'='*60}\n")
                    
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"error_fracture: {e}")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()

            self.pieces = 1
            self.Vs = [self.v_fine]
            self.Fs = [self.f_fine]


    
    def get_str_Vs(self, v):
        tempStr = ""
        for vertex_array in v:
            for vertex in vertex_array:
                tempStr += f"{vertex:.3f}@"
            tempStr += "#"
        return tempStr

    def get_str_Fs(self, f):
        tempStr = ""
        for face_array in f:
            for face in face_array:
                tempStr += f"{face}@"
            tempStr += "#"
        return tempStr

    def first_send(self):
        vertex = self.get_str_Vs(self.v_fine)
        face = self.get_str_Fs(self.f_fine)
        return vertex, face

    def get_num_str_vs(self, num_pieces, v):

        print(f"\n get_num_str_vs debug:")
        print(f"   num_pieces: {num_pieces}")
        print(f"   len(v): {len(v)}")
        
        temp = ""
        for idx in range(num_pieces):
            if idx >= len(v):
                print(f"   piece {idx}: over")
                break

            if len(v[idx]) == 0:
                print(f"   ! {idx}: empty vertex array")
                temp += "%"
                continue
            
            vertex_count = 0
            for iv in v[idx]:
                temp += f"{iv[0]}@{iv[1]}@{iv[2]}@#"
                vertex_count += 1
            temp += "%"
            
            if idx < 3:
                print(f"   pieces {idx}: {vertex_count}add vertex")
        
        percent_count = temp.count('%')
        print(f"   vertex data : '%' num = {percent_count}")
        
        return temp

    def get_num_str_fs(self, num_pieces, f):
        temp = ""
        empty_count = 0
        
        for idx in range(num_pieces):
            if idx >= len(f): 
                temp += "%"
                empty_count += 1
                continue
                
            piece_faces = f[idx] 
                
            if len(f[idx]) == 0:
                temp += "%"
                empty_count += 1
                continue
            
            if piece_faces.ndim == 1:
                for iif in piece_faces: 
                    temp += f"{iif}@"
                temp += "#" 

            elif piece_faces.ndim == 2:
                for i_f in piece_faces: 
                    for iif in i_f: 
                        temp += f"{iif}@"
                    temp += "#" 

            temp += "%" 
        
        if empty_count > 0:
            print(f"   empty fragment : {empty_count} ")
        
        return temp
    
    def set_fire_damage(self, fire_center, fire_radius):

        fire_center = np.array(fire_center)

        distances = np.linalg.norm(self.nodes - fire_center, axis=1)
        

        affected = distances < fire_radius
        
        for i in range(len(self.vertex_weights)):
            if affected[i]:
         
                ratio = distances[i] / fire_radius

                damage = 0.3 + (1.0 - 0.3) * ratio
                self.vertex_weights[i] = min(self.vertex_weights[i], damage)
        
        num_affected = np.sum(affected)
        avg_weight = np.mean(self.vertex_weights[affected]) if num_affected > 0 else 1.0
        
        print(f"fire setting complete:")
        print(f"   - center: {fire_center}")
        print(f"   - radius: {fire_radius}")
        print(f"   - vertex: {num_affected}개")
        print(f"   - aver weight: {avg_weight:.3f}")