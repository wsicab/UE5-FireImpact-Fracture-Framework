
import socket
import Unreal_Fracture as u
import numpy as np

# init 
Fracture_modes = u.Unreal_Fracture()

if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 1111

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    server.listen(1)
    
    print("=" * 60)
    print("server start")
    print("=" * 60)
    print(f"address: {ip}:{port}")
    print("  1. FIRE:x/y/z/radius   - setting")
    print("  2. x/y/z               - impact")
    print("  3. STATUS              - weight check")
    print("  4. RESET               - weight reset")
    print("=" * 60)
    
    while True:
        client, address = server.accept()
        print(f"\n✅ connected - {address[0]}:{address[1]}")

        # first send 
        vertex, face = Fracture_modes.first_send()
        senddata = vertex+"%"+face+"$"
        client.send(senddata.encode())
        print("first send complete")

        # receive data
        data = client.recv(2048)
        data = data.decode("utf-8")
        print(f"set data : {data}")
        
        # 1. fire setting
        if data.startswith("FIRE:"):
            print("\nfire data solving...")
            try:
                # "FIRE:x/y/z/radius" 
                params = data[5:].split("/")
                
                fire_x = float(params[0])
                fire_y = float(params[1])
                fire_z = float(params[2])
                fire_radius = float(params[3])
                
                print(f"   center : [{fire_x}, {fire_y}, {fire_z}]")
                print(f"   radius : {fire_radius}")
        
                Fracture_modes.set_fire_damage([fire_x, fire_y, fire_z], fire_radius)

                weights = Fracture_modes.vertex_weights
                print(f"\n weight:")
                print(f"   min: {np.min(weights):.3f}")
                print(f"   max: {np.max(weights):.3f}")
                print(f"   aver: {np.mean(weights):.3f}")
                
                weak_count = np.sum(weights < 0.5)
                print(f"   weak vertex (50% down): {weak_count}")
                
                # message send
                client.send("FIRE_OK".encode())
                print("setting complete!")
                
            except Exception as e:
                print(f"error: {e}")
                client.send("FIRE_ERROR".encode())
        
        # 2. status chck
        elif data == "STATUS":
            print("\nstatus check...")
            weights = Fracture_modes.vertex_weights
            
            status_msg = (
                f"MIN:{np.min(weights):.3f}/"
                f"MAX:{np.max(weights):.3f}/"
                f"AVG:{np.mean(weights):.3f}/"
                f"WEAK:{np.sum(weights < 0.5)}"
            )
            
            print(f"   min weight : {np.min(weights):.3f}")
            print(f"   max weight : {np.max(weights):.3f}")
            print(f"   aver weight : {np.mean(weights):.3f}")
            print(f"   weaken vertex : {np.sum(weights < 0.5)}")
            
            client.send(status_msg.encode())
            print("send status finished")
        
        # 3. init weight
        elif data == "RESET":
            print("\n init weight...")
            Fracture_modes.vertex_weights = np.ones(len(Fracture_modes.vertex_weights))
            print(f"   restore all weight as 1.0")
            client.send("RESET_OK".encode())
            print("init complete")
        
        # 4. impact
        elif data != "ok":
            print("\nimpact data...")
            try:
                impact = []
                temp = data.split("/")
                for t in temp:
                    impact.append(float(t))
                
                print(f"   location: {impact}")
                
                weights = Fracture_modes.vertex_weights
                print(f"   weight radius: {np.min(weights):.3f} ~ {np.max(weights):.3f}")

                Fracture_modes.runtime_impact_projection(impact)

                # result
                strVS = Fracture_modes.get_num_str_vs(Fracture_modes.pieces, Fracture_modes.Vs)
                strFS = Fracture_modes.get_num_str_fs(Fracture_modes.pieces, Fracture_modes.Fs)
                
                # send data
                senddata = str(Fracture_modes.pieces) + "&" + strVS + "$" + strFS
                client.send(senddata.encode())
                
                print(f"send result : {Fracture_modes.pieces} pieces")
                
            except Exception as e:
                print(f"impact error: {e}")
                import traceback
                traceback.print_exc()

        client.close()
        print("end\n")
